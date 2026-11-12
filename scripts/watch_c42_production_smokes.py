"""Observe/audit the exact CPU preview, then the GPU preview; never score LB.

These are engineering tests only. C39 admissions and any paired C13 confirmation
remain separate, mandatory qualification work before either approved submission.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c42-two-new-submissions-20261111'


def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi
    lock=(OUT/'preview-worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'preview-worker-complete.json').exists()
    api=KaggleApi();api.authenticate();deadline=datetime.now(timezone.utc)+timedelta(hours=6)
    slots={'cpu':'p20-c42-cpu-alignment','gpu':'p21-c42-gpu-alignment'}
    files=[Path(__file__),ROOT/'scripts/audit_c42_production.py',ROOT/'scripts/audit_c42_ranking.py',ROOT/'scripts/inspect_c42_remote.py',
        ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111/head-manifest.json']
    for slug in slots.values():files += [ROOT/'kaggle/forks'/slug/n for n in ['notebook.ipynb','kernel-metadata.json','production-pin.json']]
    sealed={str(p.relative_to(ROOT)):sha(p) for p in files}
    def check():
        a=json.loads((ROOT/'.local/c38-active-goal.json').read_text());assert a['active'] and a['submissions_approved']==2
        for name,h in sealed.items():assert sha(ROOT/name)==h,name
    def status(stage,**kw):
        row=dict(updated_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),**kw)
        write('preview-worker-status.json',row);print('C42 PRIVATE PREVIEW',json.dumps(row),flush=True)
    write('preview-worker-launch.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),source_seal=sealed,
        purpose='Private source/artifact engineering tests in the second free Kaggle slot while C39 runs',leaderboard_submission=False,C13_read=False))
    audits={}
    for trial,slug in slots.items():
        check();receipt_path=OUT/(trial+'-smoke-push-receipt.json')
        if not receipt_path.exists():
            assert trial=='gpu','CPU preview must already have an exact accepted handle'
            assert not (OUT/'gpu-smoke-push-intent.json').exists(),'Ambiguous private push intent; reconcile the exact handle'
            folder=ROOT/'kaggle/forks'/slug;pin=json.loads((folder/'production-pin.json').read_text())
            write('gpu-smoke-push-intent.json',dict(created_utc=stamp(),kind='Private engineering preview only; no score or labels',notebook_sha256=pin['notebook_sha256'],qualified_for_scoring=False))
            result=api.kernels_push(str(folder.resolve()),timeout='32400',acc='NvidiaTeslaT4')
            write('gpu-smoke-push-receipt.json',dict(response=str(result),url=result.url,version=result.version_number,error=result.error,submitted_utc=stamp(),scoring_submission=False))
            assert not result.error and result.version_number==1
        receipt=json.loads(receipt_path.read_text());assert receipt['version']==1 and not receipt['error']
        kernel='clarkkitchen/casmi26-'+slug
        while datetime.now(timezone.utc)<deadline:
            check();state=json.loads(str(api.kernels_status(kernel)));status(trial+'_preview_wait',kernel=kernel,cloud_status=state['status'])
            if state['status']=='COMPLETE':break
            assert state['status'] not in ['ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'],state
            time.sleep(60)
        else:raise TimeoutError('Exact private preview observation expired; do not infer termination or requeue')
        audit_path=ROOT/'outputs/forks'/(slug+'-v1')/'independent-production-audit.json'
        if not audit_path.exists():
            status(trial+'_preview_local_cloud_audit',kernel=kernel)
            with (OUT/(trial+'-production-audit.log')).open('w',encoding='utf8') as stream:
                result=subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c42_production.py',trial],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            assert result.returncode==0,trial+' production audit failed; inspect exact log'
        audit=json.loads(audit_path.read_text());assert audit['execution_qualified'] and audit['trial']==trial
        audits[trial]=dict(path=str(audit_path.relative_to(ROOT)),sha256=sha(audit_path))
        status(trial+'_engineering_passed',audit_sha256=sha(audit_path),scoring_qualified=False,C13_read=False)
    record=dict(completed_utc=stamp(),audits=audits,leaderboard_submissions=0,C13_read=False,qualified_for_scoring=False,
        next_action='Require actual C39 fixed gates and separately frozen C13 decision plus complete qualification before the two approved sequential scoring entries')
    write('preview-worker-complete.json',record);status('both_private_engineering_smokes_complete',**record)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write('preview-worker-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),automatic_retry=False,leaderboard_submissions=0,C13_read=False))
        raise
