"""Publish the masked query-only package and release exact private T4 cores."""
import argparse
import json
from urllib.parse import urlsplit
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_confirmation_preparation import ARMS, DATASET, PACKAGE, SLOTS
from c76_confirmation_inference import PLAN, RELEASE, collected, verify_plan
from submit_c76_pair import durable_write


def main(phase):
    verify_plan()  # Prior source/production admission before API or any intent.
    import msvcrt
    RELEASE.mkdir(exist_ok=True)
    with (RELEASE / 'release.lock').open('a+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        if phase == 'status':
            jobs = {}
            for arm in ARMS:
                path = RELEASE / (arm + '-push-result.json')
                if path.exists():
                    result = read(path); jobs[arm] = dict(result=result)
                    if result['version'] == 1 and not result['error']: jobs[arm]['actual'] = json.loads(str(api.kernels_status(result['kernel'])))
            value = dict(checked_utc=stamp(), dataset=str(api.dataset_status(DATASET)), jobs=jobs, scoring_entries=0)
            write(RELEASE / 'actual-status.json', value); print(json.dumps(value)); return
        if phase == 'publish':
            intent = RELEASE / 'dataset-publish-intent.json'; assert not intent.exists(), 'Reconcile the previous private upload before retry'
            metadata = read(PACKAGE / 'dataset-metadata.json'); assert metadata['id'] == DATASET
            allowed = set(read(PACKAGE / 'validation-contract.json')['files']) | {'validation-contract.json', 'dataset-metadata.json'}
            assert {path.name for path in PACKAGE.iterdir()} == allowed
            assert not any(path.is_dir() for path in PACKAGE.iterdir())
            durable_write(intent, dict(created_utc=stamp(), source_sha256=sha(__file__), dataset=DATASET,
                inference_plan_sha256=sha(PLAN), private=True, query_only=True, scoring_entries=0))
            response = api.dataset_create_new(str(PACKAGE.resolve()), public=False, quiet=True, convert_to_csv=False, dir_mode='skip')
            write(RELEASE / 'dataset-publish-response.json', dict(received_utc=stamp(), response=json.loads(response.to_json())))
            assert not getattr(response, 'error', None), str(response)
            print('C76 MASKED RESERVED QUERY DATASET PRIVATE UPLOAD REQUESTED; CHECK ACTUAL READY', DATASET); return
        arm = phase.removeprefix('push-'); assert arm in ARMS
        if arm == 'c76':
            _, audit = collected('legacy'); assert audit['actual_two_T4s']
        assert str(api.dataset_status(DATASET)) == 'ready'
        folder = ROOT / 'kaggle/forks' / SLOTS[arm]
        meta = read(folder / 'kernel-metadata.json'); pin = read(folder / 'validation-pin.json')
        assert pin['notebook_sha256'] == sha(folder / 'notebook.ipynb') and pin['metadata_sha256'] == sha(folder / 'kernel-metadata.json')
        assert meta['id'] == 'clarkkitchen/casmi26-' + SLOTS[arm]
        assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape'] == 'NvidiaTeslaT4'
        assert DATASET in meta['dataset_sources']
        intent = RELEASE / (arm + '-push-intent.json'); assert not intent.exists(), 'Reconcile the exact existing handle; never requeue blindly'
        durable_write(intent, dict(created_utc=stamp(), source_sha256=sha(__file__), arm=arm, kernel=meta['id'],
            expected_version=1, inference_plan_sha256=sha(PLAN), notebook_sha256=sha(folder / 'notebook.ipynb'),
            private=True, free_T4_only=True, scoring_entries=0))
        response = api.kernels_push(str(folder.resolve()), timeout='32400', acc='NvidiaTeslaT4')
        write(RELEASE / (arm + '-push-response.json'), dict(received_utc=stamp(), response=json.loads(response.to_json())))
        result = dict(received_utc=stamp(), kernel=meta['id'], arm=arm, version=getattr(response, 'version_number', None),
            url=getattr(response, 'url', None), error=getattr(response, 'error', None), scoring_entries=0)
        write(RELEASE / (arm + '-push-result.json'), result)
        assert result['version'] == 1 and not result['error']
        url = urlsplit(result['url']); assert url.scheme == 'https' and url.netloc in ['www.kaggle.com', 'kaggle.com']
        assert url.path.strip('/').split('/') == ['code', *meta['id'].split('/')]
        print('C76 RESERVED PRIVATE ACTUAL T4 CORE ACCEPTED', arm, json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['publish', 'status', 'push-legacy', 'push-c76'])
    main(parser.parse_args().phase)
