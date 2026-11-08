"""Durable C08 completion runner: wait, validate, real Kaggle smoke, submit ONCE.

The user explicitly requested this sequence. An execution/audit failure stops
the campaign for inspection. Training recovery remains the pinned supervisor.
No app automation, source edits, extra scoring trials or automatic refits.
"""
import argparse,csv,ctypes,json,os,re,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta

ROOT=Path(__file__).resolve().parents[1];CLI=ROOT/'.venv/Scripts/python.exe';GPU=ROOT/'.venv-gpu/Scripts/python.exe'
COMP='enveda-CASMI26-molecule-id-mass-spectra';KERNEL='clarkkitchen/casmi26-p15-compatible-c08'
VALIDATION_OUTPUTS=r'^(c08_validation_(p01\.csv|p01_metrics\.csv|final_evidence\.parquet|queries\.parquet|report\.json)|neural_candidate_features\.parquet|outsider_neural_features\.parquet)$'
SMOKE_OUTPUTS=r'^(p15_qualification_receipt\.json|c08_model_audit\.json|c08_p01_submission\.csv|c08_candidate_scores\.parquet(\.json)?|submission\.csv(\.c08\.json)?|effective_query_spectra\.parquet|neural_(public|ours|candidate_features)\.parquet)$'


class Paused(Exception):pass


def read(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return default


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def status_name(value):
    if hasattr(value,'status'):value=value.status
    elif hasattr(value,'to_json'):value=value.to_json()
    if isinstance(value,str):
        try:value=json.loads(value)
        except (ValueError,TypeError):pass
    if isinstance(value,dict):value=value.get('status',value)
    return str(value).split('.')[-1].upper()


def register(run_id,platform,kind,ref,purpose):
    with (ROOT/'runs.csv').open(newline='',encoding='utf-8') as stream:existing={r['id'] for r in csv.DictReader(stream)}
    if run_id not in existing:
        values=[str(CLI),str(ROOT/'scripts/add_run.py'),run_id,platform,kind,ref,purpose]
        if platform.startswith('local'):values.append('1')
        subprocess.run(values,cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    config=ROOT/'configs/compatible-campaign-20261108.json';campaign=read(config);auth=campaign['authorization']
    assert auth['maximum_scoring_submissions']==1 and auth['marker']=='C08-P15-20261108'
    run=ROOT/campaign['training_output'];out=run/'campaign';out.mkdir(exist_ok=True)
    required=['package_compatible_models.py','select_compatible_candidate.py','build_compatible_submission.py','audit_compatible_smoke.py','submit_compatible_once.py']
    assert all((ROOT/'scripts'/name).is_file() for name in required)
    if args.check:
        print(json.dumps(dict(ready=True,training_state=read(run/'status.json',{}).get('state'),maximum_scoring_submissions=1,
            validation_kernel=campaign['validation_kernel'],hourly_tracking=True,execution_checks_required=True,starts_training=False)));return
    import msvcrt
    lock=(out/'runner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    launch=read(run/'launch.json');assert launch
    deadline=datetime.fromisoformat(launch['started_utc'])+timedelta(hours=36)
    def guard():
        if (ROOT/'.local/compute-paused.json').exists():raise Paused('Project pause requested')
        assert datetime.now(timezone.utc)<deadline,'36-hour training-plus-postprocessing wall deadline reached'
    previous=None
    def state(stage,**values):
        nonlocal previous
        row=dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=deadline.isoformat(),
                 authorized_scoring_submissions=1,marker=auth['marker']);row.update(values)
        write(out/'status.json',row)
        if stage!=previous:print(json.dumps(row),flush=True);previous=stage
    def command(name,arguments,python=CLI,cwd=ROOT,seconds=7200):
        guard();state(name);log=out/(name+'.log');err=out/(name+'.err');start=time.monotonic()
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
        with log.open('w',encoding='utf-8') as stdout,err.open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen([str(python)]+[str(v) for v in arguments],cwd=cwd,env=env,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            try:
                while child.poll() is None:
                    guard();assert time.monotonic()-start<seconds,f'{name} exceeded its wall budget';time.sleep(2)
                assert child.returncode==0,f'{name} failed; inspect {err}'
            finally:
                if child.poll() is None:subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=30)
        return log.read_text(encoding='utf-8',errors='replace')
    # Reversible sleep inhibition; display timeout/locking remain available.
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    assert awake,'Could not request system-awake state'
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();last_remote=0;remote='UNKNOWN'
    try:
        while True:
            guard();current=read(config);training=read(run/'status.json',{});watchdog=read(run/'watchdog/status.json',{})
            if time.monotonic()-last_remote>=900:
                try:remote=status_name(api.kernels_status(current['validation_kernel']))
                except Exception as exc:remote='UNKNOWN';print('VALIDATION POLL',type(exc).__name__,flush=True)
                last_remote=time.monotonic()
            state('waiting_for_training_and_validation',training_state=training.get('state'),validation_state=remote,
                  training_arm=training.get('arm'),training_fold=training.get('fold'),training_step=training.get('step'))
            assert watchdog.get('action') not in ('needs_attention','budget_reached','paused'),'Training supervisor needs inspection'
            assert training.get('state') not in ('budget_or_resource_stop','paused'),'Training stopped before all models finished'
            if training.get('state')=='validation_pending':
                assert remote not in ('ERROR','FAILED','CANCELLED'),'External validation failed; inspect its log'
                if remote=='COMPLETE':break
            time.sleep(30)
        validation=ROOT/'outputs/forks/c08-external-validation'
        if not (validation/'c08_validation_report.json').exists():
            command('fetch_validation',[ROOT/'scripts/fetch_output.py','c08-novel-external-validation',validation,
                VALIDATION_OUTPUTS])
        bundle=ROOT/'outputs/publish/c08-models-20261108'
        if not (bundle/'c08-manifest.json').exists():command('package_models',[ROOT/'scripts/package_compatible_models.py'],GPU)
        register('C08Q','local-2070','diag','outputs/local/long-compatible-20261108/campaign/select_candidate.log',
            'Development-only C08 arm/weight choice on128 fixed novel full-pipeline queries;64 reserved reported once; explicit trial, no statistical promotion claim')
        if not (run/'qualification/report.json').exists():command('select_candidate',[ROOT/'scripts/select_compatible_candidate.py'],GPU)
        publish=out/'publish.json'
        if not publish.exists():
            command('publish_models',['-m','kaggle','datasets','create','-p','.'],cwd=bundle,seconds=1800)
            write(publish,dict(dataset='clarkkitchen/casmi26-c08-models-20261108',created_utc=datetime.now(timezone.utc).isoformat()))
        command('build_submission',[ROOT/'scripts/build_compatible_submission.py'])
        pushed=out/'push.json'
        if not pushed.exists():
            register('P15','kaggle-gpu','commit','p15-compatible-c08 v1','One authorized C08 trial; required real12 P01+tail GPU smoke, checksum/arithmetic/identity/portability audits before submission')
            output=command('push_smoke',['-m','kaggle','kernels','push','-p','kaggle/forks/p15-compatible-c08','--accelerator','NvidiaTeslaT4','--timeout','32400'],seconds=1800)
            matched=re.search(r'Kernel version (\d+) successfully pushed',output);assert matched and KERNEL in output,output[-1000:]
            write(pushed,dict(kernel=KERNEL,version=int(matched.group(1)),pushed_utc=datetime.now(timezone.utc).isoformat()))
        version=read(pushed)['version'];state('waiting_for_real_Kaggle_smoke',kernel=KERNEL,version=version)
        while True:
            guard();remote=status_name(api.kernels_status(KERNEL))
            if remote=='COMPLETE':break
            assert remote not in ('ERROR','FAILED','CANCELLED'),f'P15 real smoke failed: {remote}'
            time.sleep(180)
        folder=ROOT/'outputs/forks/p15-compatible-c08'
        if not (folder/'p15_qualification_receipt.json').exists():
            command('fetch_smoke',[ROOT/'scripts/fetch_output.py','p15-compatible-c08',folder,
                SMOKE_OUTPUTS])
        if not (folder/'independent_c08_smoke_audit.json').exists():command('audit_smoke',[ROOT/'scripts/audit_compatible_smoke.py',folder],GPU)
        command('submit_once',[ROOT/'scripts/submit_compatible_once.py','--version',str(version)],seconds=1800)
        receipt=read(ROOT/'docs/evidence/compatible-2026-11-08/p15_submission.json');assert receipt and receipt['reference']>0
        state('submitted',**receipt)
        command('refresh_runs',[ROOT/'scripts/update_runs.py'],seconds=180)
        state('submitted',**receipt);print('DONE C08 CAMPAIGN SUBMISSION',json.dumps(receipt),flush=True)
    except Paused as exc:state('paused',reason=str(exc));raise SystemExit(75)
    except BaseException as exc:state('needs_attention',error=repr(exc));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
