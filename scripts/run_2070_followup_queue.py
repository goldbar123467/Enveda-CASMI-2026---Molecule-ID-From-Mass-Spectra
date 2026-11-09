"""Keep one useful RTX2070 workload active: matched control, then mined negatives.

Waits for the already launched control; never restarts it or submits to Kaggle.
The longer experiment requires a real learning/resource/checkpoint pilot first.
"""
import argparse,csv,hashlib,json,math,os,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from training_monitor import process_alive

ROOT=Path(__file__).resolve().parents[1];CLI=ROOT/'.venv/Scripts/python.exe';GPU=ROOT/'.venv-gpu/Scripts/python.exe'
QUEUE=ROOT/'outputs/local/c09-2070-queue-20261109'
CONTROL=ROOT/'outputs/local/c09-control-20261109'
MINED=ROOT/'outputs/local/c09-mined-20261109'
POINTER=QUEUE/'active-run.json'
SOURCES=['scripts/c09_train.py','scripts/c09_control_train.py','scripts/c09_hard_train.py','scripts/online_hard_negatives.py',
    'scripts/build_c09_cosine_control.py','scripts/build_c09_hard_mining.py','scripts/supervise_local_continuation.py',
    'configs/c09-control-20261109.json','configs/c09-mined-20261109.json']
def stamp():return datetime.now(timezone.utc).isoformat()
def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def register(run_id,kind,ref,purpose):
    with (ROOT/'runs.csv').open(newline='',encoding='utf-8') as stream:ids={r['id'] for r in csv.DictReader(stream)}
    if run_id not in ids:subprocess.run([str(CLI),'scripts/add_run.py',run_id,'local-2070',kind,ref,purpose,'1'],cwd=ROOT,check=True,timeout=300)
def completed(run_id,result,ended=None):
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf-8') as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    row=next(r for r in rows if r['id']==run_id);row.update(status='complete',ended_utc=ended or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),result=result)
    temporary=path.with_suffix('.queue.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    QUEUE.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(QUEUE/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pins={p:digest(ROOT/p) for p in SOURCES}
    config=read(ROOT/'configs/c09-mined-20261109.json')
    assert config['training']['steps']==20000 and config['cycle_steps']==5000 and config['folds']==[0,1]
    assert config['mining']['sampled_negatives']==128 and config['training']['hard_negatives']==48
    assert not (ROOT/'.local/compute-paused.json').exists()
    assert (CONTROL/'source-contract.json').exists()
    if args.check:
        print(json.dumps(dict(ready=True,current_control=read(CONTROL/'status.json',{}).get('state'),
            next_run='C09H',models=2,planned_updates=40000,pilot_updates=250,maximum_hours=6,source_pins=len(pins),automatic_submission=False)),flush=True);return
    contract=read(QUEUE/'contract.json')
    if contract:assert contract['source_sha256']==pins,'Queue sources changed'
    else:write(QUEUE/'contract.json',dict(source_sha256=pins,created_utc=stamp(),authorization='User9Nov requested more2070 training and prefers useful queued GPU work',maximum_local_gpu_workloads=1,automatic_submission=False))
    deadline=datetime.now(timezone.utc)+timedelta(hours=10)
    def guard():
        assert datetime.now(timezone.utc)<deadline and not (ROOT/'.local/compute-paused.json').exists()
        assert all(digest(ROOT/p)==value for p,value in pins.items()),'Pinned queue source changed'
    def status(stage,**values):
        value=dict(stage=stage,updated_utc=stamp(),pid=os.getpid(),**values);write(QUEUE/'status.json',value);print(json.dumps(value),flush=True)
    def wait_exit(run):
        while process_alive(read(run/'status.json',{}).get('pid')):guard();time.sleep(2)
    def command(label,arguments):
        with (MINED/(label+'.log')).open('w') as stdout,(MINED/(label+'.err')).open('w') as stderr:
            child=subprocess.Popen(arguments,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        while child.poll() is None:
            try:guard()
            except BaseException:
                subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=30);raise
            time.sleep(5)
        assert child.returncode==0,f'{label} failed with{child.returncode}; inspect preserved logs'
    awake=None
    try:
        import ctypes
        awake=ctypes.windll.kernel32.SetThreadExecutionState;awake(0x80000001)
        write(POINTER,dict(run=str(CONTROL.relative_to(ROOT)),updated_utc=stamp()))
        status('waiting_for_matched_control')
        while not (CONTROL/'results.json').exists():
            guard();assert read(CONTROL/'watchdog/status.json',{}).get('action')!='needs_attention','Control supervisor needs attention';time.sleep(5)
        wait_exit(CONTROL)
        # Let the control supervisor finish its ledger refresh before this
        # queue registers another run, avoiding concurrent CSV writers.
        while process_alive(read(CONTROL/'watchdog/status.json',{}).get('pid')):
            guard();time.sleep(2)
        control=read(CONTROL/'results.json')['models'];assert len(control)==2 and all(r['continuation_steps']==5000 and r['amp_skips']==0 for r in control)
        MINED.mkdir(parents=True,exist_ok=True)
        if not (MINED/'results.json').exists():
            register('C09HP','diag','outputs/local/c09-mined-20261109/pilot.log',
                'Online hard-negative pilot: current model selects48 of128 train-only candidates; exclude positive/equal-fingerprint negatives;200+50 checkpoint recovery;2 CPU threads,70%VRAM cap')
            status('mining_pilot')
            command('pilot',[str(GPU),'-u','scripts/c09_hard_train.py','--limit-updates','200'])
            write(POINTER,dict(run=str(MINED.relative_to(ROOT)),updated_utc=stamp()))
            first=read(MINED/'status.json');assert first['state']=='pilot_checkpoint' and first['step']==200 and first['resources']['amp_skips']==0
            command('resume-pilot',[str(GPU),'-u','scripts/c09_hard_train.py','--limit-updates','50'])
            resumed=read(MINED/'status.json');assert resumed['state']=='pilot_checkpoint' and resumed['step']==250 and resumed['resources']['amp_skips']==0
            history=read(MINED/'models/mined128-served/fold-0/history.json')
            assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) and r['gradient_norm']>0 and r['mining_gain']>=0 for r in history)
            assert any(r['mining_gain']>0 for r in history)
            peak=max(first['resources']['gpu_peak_reserved_gib'],resumed['resources']['gpu_peak_reserved_gib'])
            rate=sum(r['updates_per_second'] for r in history[-3:])/min(3,len(history))
            readiness=dict(ready=True,pilot_updates=250,checkpoint_resume=True,amp_skips=0,peak_reserved_gib=peak,
                measured_updates_per_second=rate,remaining_updates=39750,optimizer_eta_hours=39750/rate/3600,
                mean_hardness_gain=sum(r['mining_gain'] for r in history)/len(history),input_errors=0,quality_claim=False)
            write(MINED/'pilot-readiness.json',readiness)
            completed('C09HP',f"250 successful updates incl200-to250 fresh-process checkpoint recovery;0 AMP skips;peak{peak:.4f}GiB;{rate:.3f}updates/s;positive hardness gain, finite gradients/loss; execution/learning only",resumed['updated_utc'][:16].replace('T',' '))
            register('C09H','train','outputs/local/c09-mined-20261109/train-0.log',
                '2 folds x20000 online-hard updates (4x5000 cyclic schedule, firstcycle matchesC09T);48 hardest of128 fromtrain-only512 pool;milestones5k/10k/15k/20k;6h cap,no submission')
            status('mining_full_run',**readiness)
            command('supervisor',[str(CLI),'-u','scripts/supervise_local_continuation.py','--plan','configs/c09-mined-20261109.json','--trainer','scripts/c09_hard_train.py'])
        result=read(MINED/'results.json');assert len(result['models'])==2
        status('training_complete_validation_pending',models=2,continuation_updates=40000,automatic_submission=False)
    except BaseException as exc:
        status('needs_attention',error=repr(exc));raise
    finally:
        if awake:awake(0x80000000)


if __name__=='__main__':main()
