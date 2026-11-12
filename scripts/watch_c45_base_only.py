"""Collect the existing C41R3 base inference only; never evaluate or submit.

C45 repairs the missing P16 external-reference validation stage. The already
running expensive core remains useful, but its P01 output cannot qualify the
production pair until the complete parent is replayed and audited separately.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c45-complete-parent-validation-20261112'
KERNEL='clarkkitchen/casmi26-c41r3-full-masked-validation'
CLOUD=ROOT/'outputs/forks/c41r3-full-masked-validation-v1'


def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name,value):
    path=OUT/name;temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8');temporary.replace(path)


def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'base-only.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not (OUT/'base-only-launch.json').exists(),'Reconcile the existing exact-job observer before restarting'
        api=KaggleApi();api.authenticate();deadline=datetime.now(timezone.utc)+timedelta(hours=8)
        files=[Path(__file__),ROOT/'scripts/collect_c41_validation.py',ROOT/'scripts/audit_c39_acquisitions.py',
            ROOT/'kaggle/forks/c41r3-full-masked-validation/validation-pin.json']
        sealed={str(p.relative_to(ROOT)):sha(p) for p in files}
        write('base-only-launch.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),
            existing_kernel=KERNEL,source_seal=sealed,new_cloud_run=False,quality_evaluation=False,C13_read=False,submission=False))
        def check():
            auth=json.loads((ROOT/'.local/c38-active-goal.json').read_text())
            assert auth['active'] and auth['submissions_approved']==2
            for name,pin in sealed.items():assert sha(ROOT/name)==pin,name
        def status(stage,**extra):
            row=dict(checked_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),**extra)
            write('base-only-status.json',row);print('C45 BASE ONLY',json.dumps(row),flush=True)
        while datetime.now(timezone.utc)<deadline:
            check()
            try:state=json.loads(str(api.kernels_status(KERNEL)))
            except Exception as error:
                status('same_job_observation_unavailable',error_type=type(error).__name__,terminal_state_inferred=False)
                time.sleep(60);continue
            status('waiting_same_existing_base_inference',kernel=KERNEL,cloud_status=state['status'])
            if state['status']=='COMPLETE':break
            assert state['status'] not in ['ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'],state
            time.sleep(60)
        else:raise TimeoutError('Observation expired, not job termination; reconcile the same handle, never requeue automatically')
        for filename,command in [
            ('base-collection.log',[sys.executable,'scripts/collect_c41_validation.py','c41r3-full-masked-validation']),
            ('base-acquisition-fidelity.log',[str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c39_acquisitions.py','--cloud',str(CLOUD/'c41_queries.parquet')])]:
            check();status('collecting_and_auditing_base_only',step=filename)
            target=CLOUD/('collection-audit.json' if filename=='base-collection.log' else 'acquisition-fidelity-audit.json')
            if not target.exists():
                with (OUT/filename).open('w',encoding='utf8') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                assert result.returncode==0,filename+' failed; inspect the retained log'
        collection=json.loads((CLOUD/'collection-audit.json').read_text())
        fidelity=json.loads((CLOUD/'acquisition-fidelity-audit.json').read_text())
        assert collection['queries']==352 and collection['forbidden_reference_rows']==0
        assert fidelity['all_original_acquisitions_retained'] and fidelity['acquisitions']==1292
        record=dict(completed_utc=stamp(),existing_kernel=KERNEL,queries=352,acquisitions=1292,
            collection_sha256=sha(CLOUD/'collection-audit.json'),fidelity_sha256=sha(CLOUD/'acquisition-fidelity-audit.json'),
            quality_evaluated=False,C13_read=False,scoring_entries_released=0,
            next_action='Replay and audit the exact P16 external-reference stage with validation-only reference exclusions before scoring either frozen head or opening quality labels.')
        write('base-only-complete.json',record);status('base_collected_only',**record)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():write('base-only-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),automatic_retry=False))
        raise
