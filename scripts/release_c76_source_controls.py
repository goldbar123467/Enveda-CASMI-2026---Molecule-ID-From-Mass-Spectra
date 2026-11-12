"""Guarded one-time actual-T4 legacy/C76 source launches; never submit scores."""
import argparse
import json
from datetime import datetime, timezone
from urllib.parse import urlsplit
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
SLOTS = {'legacy': 'c73-full-masked-validation', 'c76': 'c76-full-masked-validation'}


def verified():
    binding = BASE / 'complete-pipeline-binding-r1.json'
    for name, pin in read(binding)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    assert read(BASE / 'cpu-head-parity-r1/independent-comparison.json')['passed']
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0
    assert auth['free_local_and_kaggle_compute_authorized'] and not auth['new_paid_compute_authorized']
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()
    old = read(ROOT / 'outputs/local/c73-clean-target-validation-20261112/kernel-push-result.json')
    assert old['version'] == 0 and 'Maximum weekly GPU quota' in old['error'] and not old['url']
    return old


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['status', 'push-legacy', 'push-c76'])
    args = parser.parse_args(); old = verified()
    out = BASE / 'actual-T4-source-release'; out.mkdir(exist_ok=True)
    import msvcrt
    lock = (out / 'release.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    if args.phase == 'status':
        q = api.quota_view(); quota = json.loads(q.to_json()) if hasattr(q, 'to_json') else q
        jobs = {}
        for arm, slot in SLOTS.items():
            result_path = out / (arm + '-push-result.json')
            if result_path.exists():
                result = read(result_path)
                jobs[arm] = dict(result=result)
                if result['version'] == 1 and not result['error']:
                    jobs[arm]['actual_status'] = json.loads(str(api.kernels_status(result['kernel'])))
        snapshot = dict(checked_utc=stamp(), quota=quota, new_source_jobs=jobs,
            original_legacy_rejection=old['error'], known_rejected_quota_refresh='2026-11-14T00:00:00+00:00',
            new_scoring_entries=0)
        write(out / 'actual-status.json', snapshot); print(json.dumps(snapshot), flush=True); return
    arm = args.phase.split('-')[1]
    # This guards the already demonstrated weekly rejection. After the recorded
    # refresh, Kaggle remains authoritative about whether free quota is available.
    reset = datetime.fromisoformat('2026-11-14T00:00:00+00:00')
    if datetime.now(timezone.utc) < reset:
        held = dict(checked_utc=stamp(), arm=arm, reason=old['error'], earliest_retry_utc=reset.isoformat(),
            actual_push_attempted=False, scoring_entries=0)
        write(out / (arm + '-held-before-refresh.json'), held)
        print('SOURCE LAUNCH HELD AT DEMONSTRATED WEEKLY QUOTA GATE', json.dumps(held), flush=True); return
    if arm == 'c76':
        cloud = ROOT / 'outputs/forks/c73-full-masked-validation-v1'
        legacy = read(cloud / 'collection-audit.json')
        assert legacy['actual_two_T4s'] and legacy['remote_source_exact'] and legacy['queries'] == 512
        assert read(cloud / 'independent-core-health-audit.json')['passed']
        assert read(cloud / 'acquisition-fidelity-audit.json')['all_original_metadata_and_peaks_exact']
    query_dataset = 'clarkkitchen/casmi26-c73-full-masked-validation-20261111'
    assert str(api.dataset_status(query_dataset)) == 'ready'
    folder = ROOT / 'kaggle/forks' / SLOTS[arm]
    metadata = read(folder / 'kernel-metadata.json')
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4' and query_dataset in metadata['dataset_sources']
    intent = out / (arm + '-push-intent.json'); assert not intent.exists(), 'Reconcile prior actual handle; never requeue blindly'
    write(intent, dict(created_utc=stamp(), source_sha256=sha(__file__), kernel=metadata['id'],
        expected_version=1, notebook_sha256=sha(folder / 'notebook.ipynb'),
        complete_pipeline_binding_sha256=sha(BASE / 'complete-pipeline-binding-r1.json'),
        original_version0_rejection_sha256=sha(ROOT / 'outputs/local/c73-clean-target-validation-20261112/kernel-push-result.json'),
        private_inference_only=True, free_T4_only=True, scoring_entries=0))
    response = api.kernels_push(str(folder.resolve()), timeout='32400', acc='NvidiaTeslaT4')
    write(out / (arm + '-push-response.json'), dict(received_utc=stamp(), response=json.loads(response.to_json())))
    result = dict(submitted_utc=stamp(), version=getattr(response, 'version_number', None),
        error=getattr(response, 'error', None), url=getattr(response, 'url', None), kernel=metadata['id'], scoring_entries=0)
    if result['url']:
        url = urlsplit(result['url']); parts = url.path.strip('/').split('/')
        assert url.scheme == 'https' and url.netloc in ['www.kaggle.com', 'kaggle.com']
        assert parts == ['code', *metadata['id'].split('/')], 'Persist/reconcile unexpected canonical handle; do not start a duplicate'
    write(out / (arm + '-push-result.json'), result)
    assert result['version'] == 1 and not result['error'], result
    print('C76 ACTUAL PRIVATE T4 SOURCE RUN ACCEPTED', arm, json.dumps(result), flush=True)


if __name__ == '__main__': main()
