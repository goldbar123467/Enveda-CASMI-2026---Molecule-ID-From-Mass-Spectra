"""Publish private frozen payloads and launch one CPU-only USED engineering run."""
import argparse
import json
from urllib.parse import urlsplit
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'cpu-head-parity'
PACKAGES = {'bundle': ROOT / 'outputs/publish/c76-complete-direct-pair-20261112',
            'fixture': ROOT / 'outputs/publish/c76-used-head-inputs-20261112'}


def actual_kernel():
    result = read(OUT / 'push-result.json')
    url = urlsplit(result['url'])
    parts = url.path.strip('/').split('/')
    assert url.scheme == 'https' and url.netloc in ['www.kaggle.com', 'kaggle.com']
    assert len(parts) == 3 and parts[:2] == ['code', 'clarkkitchen']
    assert result['version'] == 1 and not result['error']
    return '/'.join(parts[1:])


def verified():
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0 and auth['free_local_and_kaggle_compute_authorized']
    assert not auth['new_paid_compute_authorized']
    binding = BASE / 'complete-pipeline-binding.json'
    for name, pin in read(binding)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    frozen = read(OUT / 'freeze.json')
    folder = ROOT / 'kaggle/forks/c76-cpu-head-parity'
    assert sha(folder / 'head_probe.py') == frozen['probe_source_sha256']
    assert sha(folder / 'kernel-metadata.json') == frozen['metadata_sha256']
    metadata = read(folder / 'kernel-metadata.json')
    assert metadata['is_private'] and not metadata['enable_gpu'] and not metadata['enable_internet']
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()
    for kind, package in PACKAGES.items():
        filename = 'pair-manifest.json' if kind == 'bundle' else 'used-input-contract.json'
        manifest = read(package / filename)
        for name, pin in manifest['files_sha256'].items(): assert sha(package / name) == pin, name
        assert set(p.name for p in package.iterdir()) == set(manifest['files_sha256']) | {filename, 'dataset-metadata.json'}
    return folder, frozen


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['publish-bundle', 'publish-fixture', 'status', 'push'])
    args = parser.parse_args()
    import msvcrt
    lock = (OUT / 'release.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    folder, frozen = verified()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    if args.phase == 'status':
        states = {kind: str(api.dataset_status(read(package / 'dataset-metadata.json')['id'])) for kind, package in PACKAGES.items()}
        q = api.quota_view(); quota = json.loads(q.to_json()) if hasattr(q, 'to_json') else q
        result = dict(checked_utc=stamp(), datasets=states, quota=quota, scoring_entries=0)
        if (OUT / 'push-result.json').exists():
            result['actual_kernel'] = actual_kernel()
            result['kernel_status'] = json.loads(str(api.kernels_status(result['actual_kernel'])))
        write(OUT / 'actual-status.json', result); print(json.dumps(result)); return
    if args.phase.startswith('publish-'):
        kind = args.phase.split('-')[1]; package = PACKAGES[kind]
        intent = OUT / (kind + '-upload-intent.json'); assert not intent.exists(), 'Reconcile the prior remote create; never publish a duplicate'
        dataset = read(package / 'dataset-metadata.json')['id']
        write(intent, dict(created_utc=stamp(), dataset=dataset, private=True,
            files_sha256={p.name: sha(p) for p in package.iterdir()}, source_sha256=sha(__file__), scoring_entries=0))
        response = api.dataset_create_new(str(package.resolve()), public=False, quiet=True, convert_to_csv=False, dir_mode='skip')
        payload = json.loads(response.to_json()) if hasattr(response, 'to_json') else str(response)
        write(OUT / (kind + '-upload-response.json'), dict(received_utc=stamp(), response=payload))
        print('PRIVATE FROZEN C76 PAYLOAD UPLOAD RETURNED', kind, dataset, flush=True); return
    for package in PACKAGES.values(): assert str(api.dataset_status(read(package / 'dataset-metadata.json')['id'])) == 'ready'
    intent = OUT / 'push-intent.json'; assert not intent.exists(), 'Reconcile existing exact CPU job instead of requeueing'
    write(intent, dict(created_utc=stamp(), source_sha256=sha(__file__),
        kernel=read(folder / 'kernel-metadata.json')['id'], expected_version=1,
        probe_source_sha256=frozen['probe_source_sha256'], metadata_sha256=frozen['metadata_sha256'],
        actual_GPU_disabled=True, quality_read=False, scoring_entries=0))
    response = api.kernels_push(str(folder.resolve()), timeout='1200')
    write(OUT / 'push-response.json', dict(received_utc=stamp(), response=json.loads(response.to_json())))
    result = dict(version=getattr(response, 'version_number', None), error=getattr(response, 'error', None),
        url=getattr(response, 'url', None), submitted_utc=stamp(), scoring_entries=0)
    write(OUT / 'push-result.json', result)
    assert result['version'] == 1 and not result['error'], result
    print('C76 PRIVATE CPU HEAD PROBE ACCEPTED', json.dumps(result), flush=True)


if __name__ == '__main__': main()
