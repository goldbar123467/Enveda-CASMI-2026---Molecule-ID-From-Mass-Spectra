"""Collect the exact returned C29 handle, bounded to the campaign; never submit."""
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c29-public-replay-20261110'
DEST=ROOT/'outputs/forks/c29-public-replay-v1'


def write(name,value):
    path=OUT/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def stamp():return datetime.now(timezone.utc).isoformat()


def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    from finish_compatible_campaign import status_name
    from submit_c09_external_once import scientific_sha,sha
    lock=(OUT/'collector.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'collector-complete.json').exists(),'Already collected; never restart completed work'
    receipt=json.loads((OUT/'push-receipt.json').read_text());kernel=receipt['kernel'];slug=kernel.split('/')[1]
    assert slug=='casmi26-c29-fixed-public-replay-development'
    deadline=datetime.fromisoformat('2026-11-10T22:08:41+00:00')
    api=KaggleApi();api.authenticate()
    while datetime.now(timezone.utc)<deadline:
        try:state=status_name(api.kernels_status(kernel))
        except Exception as exc:
            write('collector-status.json',dict(checked_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),
                state='observing',poll_error=type(exc).__name__,terminal_state_inferred=False));time.sleep(60);continue
        write('collector-status.json',dict(checked_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),kernel=kernel,
            state=state,label_metrics_computed=False,automatic_submission=False))
        print('C29 WAIT',stamp(),state,flush=True)
        if state=='COMPLETE':break
        if state in ['ERROR','CANCELLED','CANCEL_ACKNOWLEDGED']:raise RuntimeError('Exact C29 run ended '+state)
        time.sleep(60)
    else:raise TimeoutError('Campaign collection deadline; inspect same cloud handle, never infer termination or duplicate')
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name='clarkkitchen';request.kernel_slug=slug
        response=json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    m=response['metadata'];assert m['currentVersionNumber']==1 and m['isPrivate'] and not m['enableInternet']
    assert m['dockerImage']==receipt['docker_image']
    assert scientific_sha(json.loads(response['blob']['source']))==receipt['remote_scientific_sha256']
    pattern=r'(c29_.*\.(json|csv|parquet|log)|c20_engine2_.*\.(json|csv|parquet|log)|neural_(public|ours)\.parquet|outsider_neural_features\.parquet|frozen_candidate_evidence\.parquet|eng_lists\.json|execution_receipt\.json|submission_pipeline\.log|v8_report\.json)$'
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/fetch_output.py',slug.removeprefix('casmi26-'),str(DEST),pattern],cwd=ROOT,check=True,timeout=1800)
    required=['c29_report.json','c29_public_report.json','c29_p01.csv','c29_p14.csv','c29_queries.parquet',
        'c29_all_reference.csv','c29_input_alias_excluded.csv','c29_all_reference_audit.json','c29_input_alias_excluded_audit.json',
        'c29_t4_allocation.json','c29_gpu1_preflight.json','c20_engine2_recovery.json','execution_receipt.json','eng_lists.json']
    assert all((DEST/n).exists() for n in required),'Incomplete C29 collection'
    report=json.loads((DEST/'c29_public_report.json').read_text());control=json.loads((DEST/'c29_report.json').read_text())
    assert report['queries']==control['queries']==240 and control['baseline_replay_exact']
    assert not report['query_labels_mounted'] and not report['metrics_computed'] and not report['scoring_submission']
    assert not control['query_labels_mounted'] and not control['metrics_computed']
    assert sha(DEST/'c29_p01.csv')==control['baseline_sha256']
    recovery=json.loads((DEST/'c20_engine2_recovery.json').read_text())
    assert recovery['final_covered']==240 and recovery['old_entries_exact'] and recovery['coverage_guard_retained']
    assert recovery['query_rows_dropped']==recovery['placeholders_in_engine_lists']==0
    for view in ['all_reference','input_alias_excluded']:
        assert sha(DEST/('c29_'+view+'.csv'))==report['results'][view]['predictions_sha256']
        audit=report['results'][view]['audit'];assert audit['sourceA_valid_rows']>0 and audit['sourceB_valid_rows']>0
    audit=dict(completed_utc=stamp(),pid=os.getpid(),kernel=kernel,state='collected',queries=240,
        artifact_sha256={n:sha(DEST/n) for n in required},label_metrics_computed=False,automatic_submission=False,
        next_action='Evaluate fixed C29used240 only; C30must wait both separate development cycles. Reserved C13metrics remain unread.')
    write('collector-complete.json',audit);write('collector-status.json',dict(audit,deadline_utc=deadline.isoformat()))
    print('DONE C29 UNLABELLED COLLECTION',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('collector-attention.json',dict(checked_utc=stamp(),error_type=type(exc).__name__,reason=str(exc),automatic_retry=False))
        raise
