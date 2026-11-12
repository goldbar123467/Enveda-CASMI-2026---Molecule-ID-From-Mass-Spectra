"""Observe/collect one already-released auxiliary kernel; never score or train."""
import json,os,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from release_c57_target import ROOT,OUT,KERNEL,sha,read,write,stamp


def main():
    import msvcrt
    lock=(OUT/'base-only.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'base-only-launch.json').exists()
    push=read(OUT/'kernel-push-result.json');assert push['version']==1 and push['kernel']==KERNEL
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();deadline=datetime.now(timezone.utc)+timedelta(hours=8)
    paths=[Path(__file__),ROOT/'scripts/collect_c57_validation.py',ROOT/'scripts/audit_c57_acquisitions.py',ROOT/'kaggle/forks/c57-full-masked-validation/validation-pin.json']
    sealed={str(path.relative_to(ROOT)):sha(path) for path in paths}
    write(OUT/'base-only-launch.json',dict(started_utc=stamp(),pid=os.getpid(),kernel=KERNEL,version=1,deadline_utc=deadline.isoformat(),source_seal=sealed,
        can_push_or_requeue=False,can_score_or_read_quality=False,C13_read=False))
    def check():
        auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_approved']==2
        for name,pin in sealed.items():assert sha(ROOT/name)==pin,name
    def status(stage,**kwargs):
        row=dict(checked_utc=stamp(),pid=os.getpid(),stage=stage,kernel=KERNEL,version=1,deadline_utc=deadline.isoformat(),**kwargs)
        write(OUT/'base-only-status.json',row);print('C57 BASE ONLY',json.dumps(row),flush=True)
    while datetime.now(timezone.utc)<deadline:
        check()
        try:state=json.loads(str(api.kernels_status(KERNEL)))
        except Exception as error:status('status_observation_unavailable',error_type=type(error).__name__,termination_inferred=False);time.sleep(60);continue
        status('same_job_observed',actual_cloud_status=state['status'])
        if state['status']=='COMPLETE':break
        assert state['status'] not in ['ERROR','CANCELED','CANCELLED','CANCEL_ACKNOWLEDGED'],state
        time.sleep(60)
    else:raise TimeoutError('Observation deadline reached; inspect SAME handle, never infer termination or requeue')
    cloud=ROOT/'outputs/forks/c57-full-masked-validation-v1'
    for name,command,artifact in [('collection',[sys.executable,'scripts/collect_c57_validation.py'],'collection-audit.json'),
        ('fidelity',[str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c57_acquisitions.py'],'acquisition-fidelity-audit.json')]:
        check();status('collecting_base_only',step=name)
        if not (cloud/artifact).exists():
            with (OUT/(name+'.log')).open('w',encoding='utf8') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    collection=read(cloud/'collection-audit.json');fidelity=read(cloud/'acquisition-fidelity-audit.json')
    assert collection['queries']==482 and collection['forbidden_reference_rows']==0 and fidelity['all_original_acquisitions_retained']
    result=dict(completed_utc=stamp(),kernel=KERNEL,version=1,queries=482,acquisitions=2125,
        collection_sha256=sha(cloud/'collection-audit.json'),fidelity_sha256=sha(cloud/'acquisition-fidelity-audit.json'),
        auxiliary_quality_evaluated=False,C13_read=False,scored_entries_released=0,
        next_action='Replay/audit the complete P16 stage, freeze both source predictions and independently check all admissions before any C13 or scored entry.')
    write(OUT/'base-only-complete.json',result);status('base_collected_only',completion_sha256=sha(OUT/'base-only-complete.json'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(OUT/'base-only-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),automatic_retry=False));raise
