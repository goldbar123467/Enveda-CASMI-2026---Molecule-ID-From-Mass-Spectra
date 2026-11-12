"""Release exact private T4 previews only after source admission; never score."""
import argparse
import json
from urllib.parse import urlsplit
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'actual-T4-production-release'


def verified():
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0
    assert auth['free_local_and_kaggle_compute_authorized'] and not auth['new_paid_compute_authorized']
    binding_path = BASE / 'complete-pipeline-binding-r1.json'
    for name, pin in read(binding_path)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    result = read(BASE / 'source-results.json'); audit = read(BASE / 'independent-source-quality-audit.json')
    assert result['both_source_gates_passed'] and audit['passed'] and audit['both_source_gates_passed']
    assert audit['source_results_sha256'] == sha(BASE / 'source-results.json')
    assert result['complete_pipeline_binding_sha256'] == sha(binding_path)
    assert audit['source_sha256'] == sha(ROOT / 'scripts/audit_c76_source_quality.py')
    consumed = read(ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json')
    assert consumed['state'] == 'complete' and consumed['source_results_sha256'] == sha(BASE / 'source-results.json')
    assert read(BASE / 'cpu-head-parity-r1/independent-comparison.json')['passed']
    local = read(BASE / 'interface-repair-r1/independent-run-audit.json')
    assert local['passed'] and local['inactive_rows_preserved_exact'] == 388
    assert local['real_physical_fixture_promotions_checked'] == 7
    assert local['source_sha256'] == sha(ROOT / 'scripts/audit_c76_head_recovery_r1.py')
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    plan_path = BASE / 'production-evidence-plan-r1.json'; plan = read(plan_path)
    for name, pin in plan['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    return read(binding_path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['status', 'push-cpu', 'push-gpu']); args = parser.parse_args()
    binding = verified(); OUT.mkdir(exist_ok=True)
    import msvcrt
    lock = (OUT / 'release.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    if args.phase == 'status':
        states = {}
        for trial in ['cpu', 'gpu']:
            path = OUT / (trial + '-push-result.json')
            if path.exists():
                result = read(path); states[trial] = dict(result=result)
                if result['version'] == 1 and not result['error']: states[trial]['actual_status'] = json.loads(str(api.kernels_status(result['actual_kernel'])))
        snapshot = dict(checked_utc=stamp(), jobs=states, scoring_entries=0)
        write(OUT / 'actual-status.json', snapshot); print(json.dumps(snapshot)); return
    trial = args.phase.split('-')[1]
    if trial == 'gpu':
        prior = ROOT / 'outputs/forks/c76-cpu-production-v1'
        audit = read(prior / 'independent-production-audit.json')
        assert audit['execution_qualified'] and audit['trial'] == 'cpu'
        assert audit['same_cloud_input_local_head_replay_exact']
        cpu = read(OUT / 'cpu-push-result.json')
        assert json.loads(str(api.kernels_status(cpu['actual_kernel'])))['status'] == 'COMPLETE'
    folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
    meta = read(folder / 'kernel-metadata.json'); entry = binding['candidates'][trial]
    assert sha(folder / 'notebook.ipynb') == entry['notebook_sha256']
    assert sha(folder / 'production-pin.json') == entry['production_pin_sha256']
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape'] == 'NvidiaTeslaT4'
    assert not any('validation' in name for name in meta['dataset_sources'])
    assert str(api.dataset_status('clarkkitchen/casmi26-c76-complete-direct-pair-20261112')) == 'ready'
    intent = OUT / (trial + '-push-intent.json'); assert not intent.exists(), 'Reconcile prior handle, never start a duplicate'
    write(intent, dict(created_utc=stamp(), source_sha256=sha(__file__), trial=trial,
        requested_kernel=meta['id'], expected_version=1, notebook_sha256=entry['notebook_sha256'],
        complete_binding_sha256=sha(BASE / 'complete-pipeline-binding-r1.json'),
        source_quality_audit_sha256=sha(BASE / 'independent-source-quality-audit.json'),
        production_evidence_plan_sha256=sha(BASE / 'production-evidence-plan-r1.json'), private=True,
        free_T4_only=True, C13_read=False, scoring_entries=0))
    response = api.kernels_push(str(folder.resolve()), timeout='32400', acc='NvidiaTeslaT4')
    write(OUT / (trial + '-push-response.json'), dict(received_utc=stamp(), response=json.loads(response.to_json())))
    result = dict(received_utc=stamp(), trial=trial, requested_kernel=meta['id'],
        version=getattr(response, 'version_number', None), error=getattr(response, 'error', None),
        url=getattr(response, 'url', None), scoring_entries=0)
    if result['url']:
        url = urlsplit(result['url']); parts = url.path.strip('/').split('/')
        assert url.scheme == 'https' and url.netloc in ['www.kaggle.com', 'kaggle.com']
        assert len(parts) == 3 and parts[:2] == ['code', 'clarkkitchen']
        result['actual_kernel'] = '/'.join(parts[1:])
    write(OUT / (trial + '-push-result.json'), result)
    assert result['version'] == 1 and not result['error'], result
    print('C76 PRIVATE ACTUAL T4 PRODUCTION PREVIEW ACCEPTED', trial, json.dumps(result), flush=True)


if __name__ == '__main__': main()
