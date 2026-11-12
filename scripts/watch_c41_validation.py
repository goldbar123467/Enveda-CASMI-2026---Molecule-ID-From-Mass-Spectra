"""Bounded exact-job continuation: smoke -> full validation -> fixed evaluation.

This worker belongs to the active C38 goal. It never submits to the leaderboard,
restarts an old campaign, starts an hourly automation or reads C13 labels.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c41r3-masked-pipeline-validation-20261111'
SCORES=ROOT/'outputs/local/c42-fixed-c39-validation-20261111'


def stamp():return datetime.now(timezone.utc).isoformat()


def write(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorized():
    auth=json.loads((ROOT/'.local/c38-active-goal.json').read_text())
    assert auth['active'] and auth['submissions_approved']==2 and not auth['hourly_automation']
    assert not auth.get('scientific_gate_hold'),'C45 complete-parent repair holds the retired automatic evaluation path'


def run(args,log):
    authorized()
    with (OUT/log).open('w',encoding='utf8') as stream:
        result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,log+' failed; inspect exact retained log'


def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi
    lock=(OUT/'continuation.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'continuation-complete.json').exists()
    authorized();api=KaggleApi();api.authenticate();deadline=datetime.now(timezone.utc)+timedelta(hours=6)
    sealed={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/collect_c41_validation.py',ROOT/'scripts/evaluate_c42_fixed.py',
        ROOT/'kaggle/forks/c41r3-smoke-masked-validation/validation-pin.json',ROOT/'kaggle/forks/c41r3-full-masked-validation/validation-pin.json',
        ROOT/'scripts/audit_c39_acquisitions.py',
        ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111/head-manifest.json',ROOT/'outputs/local/c42-crossmodal-engineering-20261111/production-source-audit.json']}
    write('continuation-launch.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),source_seal=sealed,
        scoring_submission=False,C13_read=False,old_campaigns_resumed=False))
    def status(stage,**kw):
        row=dict(updated_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),**kw)
        write('continuation-status.json',row);print('C41 CONTINUATION',json.dumps(row),flush=True)
    def check_seal():
        authorized()
        for name,h in sealed.items():assert sha(ROOT/name)==h,name
    def wait(slot):
        while datetime.now(timezone.utc)<deadline:
            check_seal();state=json.loads(str(api.kernels_status('clarkkitchen/casmi26-'+slot)))
            status(slot,cloud_status=state['status'])
            if state['status']=='COMPLETE':return
            assert state['status'] not in ['ERROR','CANCELLED','CANCELED','CANCEL_ACKNOWLEDGED'],state
            time.sleep(60)
        raise TimeoutError('Bounded exact-job continuation deadline reached; inspect retained run, do not requeue')
    smoke='c41r3-smoke-masked-validation';full='c41r3-full-masked-validation'
    wait(smoke)
    audit=ROOT/'outputs/forks'/(smoke+'-v1')/'collection-audit.json'
    if not audit.exists():run([sys.executable,'scripts/collect_c41_validation.py',smoke],'smoke-collection.log')
    proof=json.loads(audit.read_text());assert proof['queries']==12 and proof['actual_two_T4s'] and proof['forbidden_reference_rows']==0
    pin=lambda slot:json.loads((ROOT/'kaggle/forks'/slot/'validation-pin.json').read_text())
    assert pin(smoke)['frozen_scientific_payload_sha256']==pin(full)['frozen_scientific_payload_sha256']
    assert pin(full)['queries']==352 and not pin(full)['labels_mounted']
    status('smoke_verified_full_release_ready',smoke_audit_sha256=sha(audit))
    response_path=OUT/'full-push-receipt.json'
    if not response_path.exists():
        assert not (OUT/'full-push-intent.json').exists(),'Ambiguous push intent; reconcile before releasing anything'
        check_seal();assert str(api.dataset_status('clarkkitchen/casmi26-c41r3-full-masked-validation-20261111'))=='ready'
        write('full-push-intent.json',dict(created_utc=stamp(),notebook_sha256=pin(full)['notebook_sha256'],smoke_audit_sha256=sha(audit)))
        response=api.kernels_push(str((ROOT/'kaggle/forks'/full).resolve()),timeout='32400',acc='NvidiaTeslaT4')
        write('full-push-receipt.json',dict(response=str(response),url=response.url,version=response.version_number,error=response.error,submitted_utc=stamp()))
        assert response.version_number==1 and not response.error
    else:assert json.loads(response_path.read_text())['version']==1
    wait(full)
    audit=ROOT/'outputs/forks'/(full+'-v1')/'collection-audit.json'
    if not audit.exists():run([sys.executable,'scripts/collect_c41_validation.py',full],'full-collection.log')
    check_seal();proof=json.loads(audit.read_text());assert proof['queries']==352 and proof['forbidden_reference_rows']==0
    status('full_verified_encoding_fixed_candidates',full_audit_sha256=sha(audit))
    head=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111';cloud=audit.parent
    if not (cloud/'acquisition-fidelity-audit.json').exists():
        run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/audit_c39_acquisitions.py','--cloud',str(cloud/'c41_queries.parquet')],'full-acquisition-fidelity.log')
    if not (SCORES/'score-receipt.json').exists():
        assert not SCORES.exists(),'Partial scoring directory; inspect rather than overwrite'
        run([str(ROOT/'.venv-encoders/Scripts/python.exe'),str(head/'score_c42_crossmodal.py'),str(head),
             str(ROOT/'outputs/publish/frozen-p14-20261107'),str(cloud/'c41_p01.csv'),str(cloud/'c41_queries.parquet'),str(SCORES)],'C42-fixed-scoring.log')
    check_seal();status('fixed_predictions_frozen_evaluating_C39_only')
    if not (SCORES/'quality-results.json').exists():run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/evaluate_c42_fixed.py',str(SCORES.relative_to(ROOT))],'C42-fixed-evaluation.log')
    result=json.loads((SCORES/'quality-results.json').read_text())
    record=dict(completed_utc=stamp(),state='fixed_C39_evaluation_complete',quality_results_sha256=sha(SCORES/'quality-results.json'),
        admissions={k:result['results'][k]['admission'] for k in ['cpu','gpu']},C13_read=False,scoring_submissions=0,
        next_action='Review actual fixed quality and qualify or reject; this worker cannot release any scoring entry.')
    write('continuation-complete.json',record);status('complete',**record)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write('continuation-attention.json',dict(checked_utc=stamp(),error_type=type(error).__name__,reason=str(error),automatic_retry=False,scoring_submissions=0,C13_read=False))
        raise
