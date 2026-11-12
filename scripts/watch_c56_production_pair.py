"""Collect/audit CPU preview, release/audit its frozen GPU mate; never score."""
import json,os,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from release_c56_production_smoke import ROOT,OUT,read,write,sha,stamp


def main():
    import msvcrt
    lock=(OUT/'production-pair-watch.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'production-pair-watch-launch.json').exists()
    family=read(OUT/'complete-candidate-freeze.json');deadline=datetime.now(timezone.utc)+timedelta(hours=8)
    assert read(OUT/'used-auxiliary-results.json')['combined_used_development_pair_passed']
    assert read(OUT/'cpu-private-push-result.json')['version']==1
    paths=[Path(__file__),OUT/'complete-candidate-freeze.json',ROOT/'configs/c56-bounded-semantic-pair-20261112.json']
    paths.extend(ROOT/'scripts'/name for name in ['release_c56_production_smoke.py','inspect_c56_remote.py','audit_c56_production.py','audit_c56_ranking.py'])
    for entry in family['candidates'].values():paths.extend([ROOT/entry['path']/'notebook.ipynb',ROOT/entry['path']/'production-pin.json',ROOT/entry['bundle']/'head-manifest.json'])
    sealed={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(OUT/'production-pair-watch-launch.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),source_seal=sealed,
        can_release_only_frozen_P31_private_v1=True,can_score=False,can_read_quality=False,C13_read=False))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    def check():
        auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_approved']==2
        for name,pin in sealed.items():assert sha(ROOT/name)==pin,name
    def status(stage,**kwargs):
        row=dict(checked_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),**kwargs)
        write(OUT/'production-pair-watch-status.json',row);print('C56 PRIVATE PAIR WATCH',row,flush=True)
    for trial in ['cpu','gpu']:
        entry=family['candidates'][trial];check()
        if trial=='gpu' and not (OUT/'gpu-private-push-result.json').exists():
            assert not (OUT/'gpu-private-push-intent.json').exists(),'Unreconciled push intent; do not retry'
            status('releasing_frozen_private_GPU_only',kernel=entry['kernel'])
            with (OUT/'gpu-private-release-watch.log').open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,'scripts/release_c56_production_smoke.py','gpu','push'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        while datetime.now(timezone.utc)<deadline:
            check()
            try:state=json.loads(str(api.kernels_status(entry['kernel'])))
            except Exception as error:status('status_observation_unavailable',trial=trial,error_type=type(error).__name__,termination_inferred=False);time.sleep(60);continue
            status('same_private_job_observed',trial=trial,kernel=entry['kernel'],actual_cloud_status=state['status'])
            if state['status']=='COMPLETE':break
            assert state['status'] not in ['ERROR','CANCELED','CANCELLED','CANCEL_ACKNOWLEDGED'],state
            time.sleep(60)
        else:raise TimeoutError('Observe/reconcile same exact handle; do not infer terminal state or blindly requeue')
        dest=ROOT/'outputs/forks'/(Path(entry['path']).name+'-v1');audit_path=dest/'independent-production-audit.json'
        if not audit_path.exists():
            check();status('auditing_private_preview',trial=trial,kernel=entry['kernel'])
            with (OUT/(trial+'-production-watch-audit.log')).open('w',encoding='utf8') as log:
                subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c56_production.py',trial],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        audit=read(audit_path);assert audit['execution_qualified'] and audit['local_cloud_final_lists_exact'] and audit['notebook_sha256']==entry['notebook_sha256']
        assert audit['head_manifest_sha256']==entry['head_manifest_sha256'] and audit['source_sha256']==sha(ROOT/'scripts/audit_c56_production.py')
        status('private_preview_audited',trial=trial,production_audit_sha256=sha(audit_path))
    write(OUT/'production-pair-watch-complete.json',dict(completed_utc=stamp(),both_private_previews_audited=True,scoring_submissions=0,C53_quality_read=False,C13_read=False,
        next_action='Complete and jointly seal all482P16/head source predictions; only then run the predeclared fresh-source evaluation.'))
    status('both_private_previews_audited')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(OUT/'production-pair-watch-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),automatic_retry=False,scoring_submissions=0));raise
