"""Private CPU proof of the repaired400row production head interface."""
import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'cpu-head-parity-r1'
BUNDLE = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'
PACKAGE = ROOT / 'outputs/publish/c76-r1-production-head-inputs-20261112'
FOLDER = ROOT / 'kaggle/forks/c76-r1-cpu-head-parity'


def verified():
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0 and auth['free_local_and_kaggle_compute_authorized']
    assert not auth['new_paid_compute_authorized']
    for name, pin in read(BASE / 'complete-pipeline-binding-r1.json')['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    assert read(BASE / 'interface-repair-r1/independent-run-audit.json')['composed_r1_local_production_passed']
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()


def prepare():
    verified(); assert not OUT.exists() and not PACKAGE.exists() and not FOLDER.exists()
    prefix = read(BASE / 'interface-repair-r1/completed-prefix-audit.json'); work = Path(prefix['working'])
    names = ['c09e_p01_submission.csv', 'c76_p16.csv', 'c09e_external_receipt.json', 'c09e_matches.parquet',
        'effective_query_spectra.parquet', 'execution_receipt.json', 'neural_candidate_features.parquet',
        'neural_public.parquet', 'neural_ours.parquet']
    PACKAGE.mkdir()
    for name in names:
        assert sha(work / name) == prefix['reusable_unchanged_files_sha256'][name]
        shutil.copyfile(work / name, PACKAGE / name)
    write(PACKAGE / 'head-input-contract.json', dict(frozen_utc=stamp(), files_sha256={name: sha(PACKAGE / name) for name in names},
        actual_public_unlabeled_production_preview=True, active_queries=12, acquisitions=41, output_rows=400,
        intentionally_unprocessed_query_ids=388, no_query_truth_or_expected_head_predictions_mounted=True,
        completed_prefix_audit_sha256=sha(BASE / 'interface-repair-r1/completed-prefix-audit.json')))
    write(PACKAGE / 'dataset-metadata.json', dict(title='CASMI26 C76 r1 production head inputs',
        id='clarkkitchen/casmi26-c76-r1-production-head-inputs-20261112', licenses=[{'name': 'CC0-1.0'}]))
    nb = read(ROOT / 'kaggle/forks/c76-full-masked-validation/notebook.ipynb')
    inner = json.loads(base64.b64decode(value(unwrap(''.join(nb['cells'][12]['source'])), 'FROZEN_B64')))
    bootstrap = ''.join(inner['cells'][2]['source'])
    pair_pin = sha(BUNDLE / 'pair-manifest.json'); data_pin = sha(PACKAGE / 'head-input-contract.json')
    source = '''import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='2'
os.environ['PYTHONHASHSEED']='0'
import torch
torch.set_num_threads(2)
import hashlib,json,shutil,subprocess,sys,platform
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=Path('/kaggle/working');inputs=Path('/kaggle/input')
(root/'official-bootstrap.py').write_text(BOOTSTRAP,encoding='utf8')
subprocess.run([sys.executable,str(root/'official-bootstrap.py')],check=True)
hits=[p for p in inputs.rglob('pair-manifest.json') if sha(p)==PAIR_PIN];assert len(hits)==1
bundle=hits[0].parent
hits=[p for p in inputs.rglob('head-input-contract.json') if sha(p)==INPUT_PIN];assert len(hits)==1
fixture=hits[0].parent;contract=json.loads(hits[0].read_text())
assert contract['active_queries']==12 and contract['acquisitions']==41 and contract['output_rows']==400
assert contract['actual_public_unlabeled_production_preview'] and contract['no_query_truth_or_expected_head_predictions_mounted']
work=root/'head-work';work.mkdir()
for name,pin in contract['files_sha256'].items():
    assert sha(fixture/name)==pin,name
    shutil.copyfile(fixture/name,work/name)
references=list(inputs.rglob('ext_meta.parquet'));assert len(references)==1
subprocess.run([sys.executable,str(bundle/'score_c76_complete_pair.py'),'--work',str(work),'--bundle',str(bundle),
    '--reference',str(references[0].parent),'--output',str(root/'results')],check=True)
assert not torch.cuda.is_initialized()
receipt=json.loads((root/'results/score-receipt.json').read_text())
assert receipt['queries']==12 and receipt['acquisitions']==41 and receipt['valid_rows']==400
assert receipt['exact_unprocessed_fallback_rows']==388 and set(receipt['trials'])=={'cpu','gpu'}
result=dict(source_sha256=sha(__file__),complete_pipeline_binding_sha256=BINDING_PIN,
    pair_manifest_sha256=PAIR_PIN,input_contract_sha256=INPUT_PIN,actual_GPU_disabled=True,GPU_initialized=False,
    queries=12,acquisitions=41,output_rows=400,inactive_rows=388,
    python=platform.python_version(),numpy=__import__('numpy').__version__,torch=torch.__version__,
    official_rdkit=__import__('rdkit').__version__,actual_public_unlabeled_production_preview=True,
    no_expected_head_predictions_mounted=True,quality_labels_read=False,C13_read=False,scoring_entries=0,
    score_receipt_sha256=sha(root/'results/score-receipt.json'))
(root/'head-probe.json').write_text(json.dumps(result,indent=2))
print('C76 R1 ACTUAL CPU400ROW HEAD PROBE COMPLETE',json.dumps(result),flush=True)
'''.replace('BOOTSTRAP', repr(bootstrap)).replace('PAIR_PIN', repr(pair_pin)).replace('INPUT_PIN', repr(data_pin)).replace(
        'BINDING_PIN', repr(sha(BASE / 'complete-pipeline-binding-r1.json')))
    compile(source, 'c76_r1_cpu_head_probe', 'exec'); FOLDER.mkdir(); OUT.mkdir()
    (FOLDER / 'head_probe.py').write_text(source, encoding='utf8', newline='\n')
    metadata = dict(id='clarkkitchen/casmi26-c76-r1-cpu-head-parity', title='casmi26 c76 r1 cpu head parity',
        code_file='head_probe.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=False, enable_tpu=False, enable_internet=False,
        dataset_sources=[read(BUNDLE / 'dataset-metadata.json')['id'], read(PACKAGE / 'dataset-metadata.json')['id'],
            'takumuhata/casmi26-extlib', 'metric/rdkit-2026-3-3-wheel'], competition_sources=[], kernel_sources=[],
        docker_image=read(ROOT / 'kaggle/forks/c76-cpu-production/kernel-metadata.json')['docker_image'], docker_image_pinning_type='original')
    write(FOLDER / 'kernel-metadata.json', metadata)
    write(OUT / 'freeze.json', dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        probe_source_sha256=sha(FOLDER / 'head_probe.py'), metadata_sha256=sha(FOLDER / 'kernel-metadata.json'),
        complete_pipeline_binding_sha256=sha(BASE / 'complete-pipeline-binding-r1.json'),
        pair_manifest_sha256=pair_pin, input_contract_sha256=data_pin,
        comparisons=dict(features='bitexact', lists='exact', scores=dict(atol=1e-10, rtol=1e-10)),
        original_production_context_including388placeholder_rows=True, expected_head_predictions_mounted=False,
        quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 R1 FULL400ROW CPU PROBE PREPARED; NOT RELEASED', flush=True)


def api_client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate(); return api


def kernel():
    pushed = read(OUT / 'push-result.json'); assert pushed['version'] == 1 and not pushed['error']
    url = urlsplit(pushed['url']); parts = url.path.strip('/').split('/')
    assert url.scheme == 'https' and url.netloc in ['www.kaggle.com', 'kaggle.com']
    assert len(parts) == 3 and parts[:2] == ['code', 'clarkkitchen']
    return '/'.join(parts[1:])


def release(phase):
    verified(); pin = read(OUT / 'freeze.json')
    assert sha(FOLDER / 'head_probe.py') == pin['probe_source_sha256']
    assert sha(FOLDER / 'kernel-metadata.json') == pin['metadata_sha256']
    import msvcrt
    lock = (OUT / 'release.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    api = api_client(); packages = {'bundle': BUNDLE, 'fixture': PACKAGE}
    if phase == 'status':
        result = dict(checked_utc=stamp(), datasets={kind: str(api.dataset_status(read(path / 'dataset-metadata.json')['id'])) for kind, path in packages.items()}, scoring_entries=0)
        if (OUT / 'push-result.json').exists(): result.update(actual_kernel=kernel(), kernel_status=json.loads(str(api.kernels_status(kernel()))))
        write(OUT / 'actual-status.json', result); print(json.dumps(result)); return
    if phase.startswith('publish-'):
        kind = phase.split('-')[1]; package = packages[kind]
        filename = 'pair-manifest.json' if kind == 'bundle' else 'head-input-contract.json'
        spec = read(package / filename)
        for name, digest in spec['files_sha256'].items(): assert sha(package / name) == digest, name
        assert set(p.name for p in package.iterdir()) == set(spec['files_sha256']) | {filename, 'dataset-metadata.json'}
        intent = OUT / (kind + '-upload-intent.json'); assert not intent.exists(), 'Reconcile previous upload; no duplicate'
        write(intent, dict(created_utc=stamp(), source_sha256=sha(__file__), private=True,
            files_sha256={p.name: sha(p) for p in package.iterdir()}, scoring_entries=0))
        response = api.dataset_create_new(str(package.resolve()), public=False, quiet=True, convert_to_csv=False, dir_mode='skip')
        payload = json.loads(response.to_json()) if hasattr(response, 'to_json') else str(response)
        write(OUT / (kind + '-upload-response.json'), dict(received_utc=stamp(), response=payload))
        print('C76 R1 PRIVATE PAYLOAD UPLOAD RETURNED', kind); return
    assert phase == 'push' and not (OUT / 'push-intent.json').exists()
    for package in packages.values(): assert str(api.dataset_status(read(package / 'dataset-metadata.json')['id'])) == 'ready'
    write(OUT / 'push-intent.json', dict(created_utc=stamp(), source_sha256=sha(__file__),
        requested_kernel=read(FOLDER / 'kernel-metadata.json')['id'], expected_version=1,
        probe_source_sha256=pin['probe_source_sha256'], metadata_sha256=pin['metadata_sha256'],
        actual_GPU_disabled=True, quality_read=False, C13_read=False, scoring_entries=0))
    response = api.kernels_push(str(FOLDER.resolve()), timeout='1200')
    write(OUT / 'push-response.json', dict(received_utc=stamp(), response=json.loads(response.to_json())))
    result = dict(received_utc=stamp(), version=getattr(response, 'version_number', None), error=getattr(response, 'error', None), url=getattr(response, 'url', None), scoring_entries=0)
    write(OUT / 'push-result.json', result); assert result['version'] == 1 and not result['error'], result
    print('C76 R1 PRIVATE CPU HEAD PROBE ACCEPTED', json.dumps(result), flush=True)


def fetch():
    import requests
    from kaggle.api.kaggle_api_extended import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    verified(); api = api_client(); handle = kernel(); pin = read(OUT / 'freeze.json'); cloud = OUT / 'cloud'
    assert not cloud.exists() and json.loads(str(api.kernels_status(handle)))['status'] == 'COMPLETE'
    wanted = {'head-probe.json', 'results/score-receipt.json', 'results/features.npy', 'results/candidates.parquet',
        'results/physical-parent.csv', 'results/cpu.csv', 'results/gpu.csv', 'results/cpu-scores.npy', 'results/gpu-scores.npy'}
    with api.build_kaggle_client() as client:
        req = ApiGetKernelRequest(); req.user_name = 'clarkkitchen'; req.kernel_slug = handle.split('/')[1]
        remote = client.kernels.kernels_api_client.get_kernel(req); meta = remote.metadata
        assert meta.current_version_number == 1 and meta.is_private and not meta.enable_gpu and not meta.enable_internet
        assert meta.docker_image == read(FOLDER / 'kernel-metadata.json')['docker_image']
        assert hashlib.sha256(remote.blob.source.encode()).hexdigest() == pin['probe_source_sha256']
        found = {}; token = ''; pages = 0; log = None
        while True:
            pages += 1; assert pages <= 100
            req = ApiListKernelSessionOutputRequest(); req.user_name = 'clarkkitchen'; req.kernel_slug = handle.split('/')[1]
            req.page_size = 100; req.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            if pages == 1: log = response.log
            for item in response.files or []:
                if item.file_name in wanted: assert item.file_name not in found; found[item.file_name] = item
            token = response.next_page_token
            if not token: break
        assert set(found) == wanted, sorted(wanted-set(found))
    cloud.mkdir()
    for name, item in found.items():
        response = requests.get(item.url, timeout=60); response.raise_for_status()
        path = cloud / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(response.content)
    if log: (cloud / 'run.log').write_text(log, encoding='utf8')
    actual = read(cloud / 'head-probe.json')
    assert actual['source_sha256'] == pin['probe_source_sha256'] and actual['complete_pipeline_binding_sha256'] == pin['complete_pipeline_binding_sha256']
    assert actual['pair_manifest_sha256'] == pin['pair_manifest_sha256'] and actual['input_contract_sha256'] == pin['input_contract_sha256']
    assert actual['actual_GPU_disabled'] and not actual['GPU_initialized'] and actual['official_rdkit'] == '2026.03.3'
    assert actual['queries'] == 12 and actual['acquisitions'] == 41 and actual['output_rows'] == 400 and actual['inactive_rows'] == 388
    assert actual['no_expected_head_predictions_mounted'] and not actual['quality_labels_read']
    receipt = read(cloud / 'results/score-receipt.json'); assert sha(cloud / 'results/score-receipt.json') == actual['score_receipt_sha256']
    for name in wanted - {'head-probe.json', 'results/score-receipt.json'}: assert sha(cloud / name) == receipt['files_sha256'][name.split('/')[-1]]
    write(OUT / 'cloud-collection-audit.json', dict(completed_utc=stamp(), source_sha256=sha(__file__), actual_kernel=handle,
        remote_source_exact=True, actual_original_image_exact=True, actual_GPU_disabled=True, private=True,
        listing_pages=pages, files_sha256={name: sha(cloud / name) for name in wanted}, quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 R1 ACTUAL CLOUD CPU SOURCE/IMAGE/400ROW OUTPUT VERIFIED', flush=True)


def audit():
    import torch
    import numpy as np
    import pandas as pd
    pin = read(OUT / 'freeze.json'); cloud = OUT / 'cloud'; target = OUT / 'independent-comparison.json'
    assert not target.exists()
    collection = read(OUT / 'cloud-collection-audit.json'); assert collection['source_sha256'] == sha(__file__)
    for name, digest in collection['files_sha256'].items(): assert sha(cloud / name) == digest
    working = Path(read(BASE / 'interface-repair-r1/head-recovery-results.json')['working'])
    trials = {}
    for trial in ['cpu', 'gpu']:
        local = working / ('c76_' + trial + '_scores_r1'); remote = cloud / 'results'
        np.testing.assert_array_equal(np.load(local / 'features.npy'), np.load(remote / 'features.npy'))
        pd.testing.assert_frame_equal(pd.read_parquet(local / 'candidates.parquet'), pd.read_parquet(remote / 'candidates.parquet'), check_exact=True)
        a, b = np.load(local / (trial + '-scores.npy')), np.load(remote / (trial + '-scores.npy'))
        np.testing.assert_allclose(a, b, **pin['comparisons']['scores'])
        pd.testing.assert_frame_equal(pd.read_csv(local / (trial + '.csv'), dtype=str), pd.read_csv(remote / (trial + '.csv'), dtype=str), check_exact=True)
        trials[trial] = dict(scores=len(a), maximum_absolute_score_difference=float(np.max(np.abs(a-b))), all400_final_lists_exact=True)
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        actual_source_image_and_CPU_only_execution_verified=True,
        preview_queries=12, acquisitions=41, output_rows=400, inactive_rows_preserved=388,
        all101features_and_candidate_records_bitexact=True, trials=trials,
        complete_pipeline_binding_sha256=pin['complete_pipeline_binding_sha256'],
        actual_T4_source_or_production_evidence_replaced=False, quality_read=False, C13_read=False,
        scoring_entries=0, qualified_for_scoring=False))
    print('C76 R1 ACTUAL400ROW CPU CROSS-HOST HEAD PARITY PASSED', json.dumps(trials), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'publish-bundle', 'publish-fixture', 'status', 'push', 'fetch', 'audit'])
    phase = parser.parse_args().phase
    if phase == 'prepare': prepare()
    elif phase == 'fetch': fetch()
    elif phase == 'audit': audit()
    else: release(phase)
