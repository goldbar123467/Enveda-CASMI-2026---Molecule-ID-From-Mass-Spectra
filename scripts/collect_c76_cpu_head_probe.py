"""Verify actual CPU execution and compare the frozen deployment arithmetic."""
import argparse
import hashlib
import json
from c75_port_utils import ROOT, read, write, sha, stamp
from release_c76_cpu_head_probe import actual_kernel

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'cpu-head-parity'


def fetch():
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    folder = ROOT / 'kaggle/forks/c76-cpu-head-parity'; metadata = read(folder / 'kernel-metadata.json')
    pin = read(OUT / 'freeze.json'); cloud = OUT / 'cloud'
    assert not cloud.exists()
    assert sha(folder / 'head_probe.py') == pin['probe_source_sha256']
    assert sha(folder / 'kernel-metadata.json') == pin['metadata_sha256']
    api = KaggleApi(); api.authenticate()
    kernel = actual_kernel()
    assert json.loads(str(api.kernels_status(kernel)))['status'] == 'COMPLETE'
    wanted = {'head-probe.json', 'results/score-receipt.json', 'results/features.npy',
        'results/cpu-scores.npy', 'results/gpu-scores.npy', 'results/cpu.csv', 'results/gpu.csv',
        'results/physical-parent.csv', 'results/candidates.parquet'}
    with api.build_kaggle_client() as client:
        req = ApiGetKernelRequest(); req.user_name = 'clarkkitchen'; req.kernel_slug = kernel.split('/')[1]
        remote = client.kernels.kernels_api_client.get_kernel(req); m = remote.metadata
        assert m.current_version_number == 1 and m.is_private and not m.enable_gpu and not m.enable_internet
        assert m.docker_image == metadata['docker_image']
        assert hashlib.sha256(remote.blob.source.encode()).hexdigest() == pin['probe_source_sha256']
        found = {}; token = ''; pages = 0; log = None
        while True:
            pages += 1; assert pages <= 100
            req = ApiListKernelSessionOutputRequest(); req.user_name = 'clarkkitchen'
            req.kernel_slug = kernel.split('/')[1]; req.page_size = 100; req.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            if pages == 1: log = response.log
            for item in response.files or []:
                if item.file_name in wanted:
                    assert item.file_name not in found; found[item.file_name] = item
            token = response.next_page_token
            if not token: break
        assert set(found) == wanted, sorted(wanted - set(found))
    cloud.mkdir()
    for name, item in found.items():
        response = requests.get(item.url, timeout=60); response.raise_for_status()
        target = cloud / name; assert target.resolve().is_relative_to(cloud.resolve())
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(response.content)
    if log: (cloud / 'run.log').write_text(log, encoding='utf8')
    actual = read(cloud / 'head-probe.json')
    assert actual['source_sha256'] == pin['probe_source_sha256']
    assert actual['complete_pipeline_binding_sha256'] == pin['complete_pipeline_binding_sha256']
    assert actual['pair_manifest_sha256'] == pin['pair_manifest_sha256']
    assert actual['used_input_contract_sha256'] == pin['used_input_contract_sha256']
    assert actual['actual_GPU_disabled'] and not actual['GPU_initialized']
    assert actual['USED_only'] and actual['no_expected_head_predictions_mounted']
    assert actual['official_rdkit'] == '2026.03.3' and not actual['quality_labels_read']
    assert sha(cloud / 'results/score-receipt.json') == actual['score_receipt_sha256']
    receipt = read(cloud / 'results/score-receipt.json')
    for name in wanted - {'head-probe.json', 'results/score-receipt.json'}:
        assert sha(cloud / name) == receipt['files_sha256'][name.split('/')[-1]], name
    write(OUT / 'cloud-collection-audit.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        kernel=kernel, requested_kernel=metadata['id'], version=1, remote_source_exact=True, actual_original_image_exact=True,
        actual_GPU_disabled=True, private=True, listing_pages=pages,
        files_sha256={name: sha(cloud / name) for name in wanted},
        engineering_USED_inputs_only=True, quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 CPU HEAD ACTUAL SOURCE/IMAGE/OUTPUT VERIFIED', flush=True)


def audit():
    import torch
    import numpy as np
    import pandas as pd
    pin = read(OUT / 'freeze.json'); collection = read(OUT / 'cloud-collection-audit.json')
    assert collection['source_sha256'] == sha(__file__)
    cloud = OUT / 'cloud'; target = OUT / 'independent-comparison.json'; assert not target.exists()
    for name, expected in collection['files_sha256'].items(): assert sha(cloud / name) == expected, name
    local = __import__('pathlib').Path(read(BASE / 'complete-pair-used-probe/results.json')['working']) / 'c76_scores'
    assert sha(local / 'score-receipt.json') == pin['local_used_score_receipt_sha256']
    a, b = np.load(local / 'features.npy'), np.load(cloud / 'results/features.npy')
    np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(pd.read_parquet(local / 'candidates.parquet'),
        pd.read_parquet(cloud / 'results/candidates.parquet'), check_exact=True)
    compared = {}
    tolerance = pin['comparisons']['scores']
    for trial in ['cpu', 'gpu']:
        a = np.load(local / (trial + '-scores.npy'))
        b = np.load(cloud / 'results' / (trial + '-scores.npy'))
        np.testing.assert_allclose(a, b, **tolerance)
        pd.testing.assert_frame_equal(pd.read_csv(local / (trial + '.csv'), dtype=str),
            pd.read_csv(cloud / 'results' / (trial + '.csv'), dtype=str), check_exact=True)
        compared[trial] = dict(scores=len(a), maximum_absolute_score_difference=float(np.max(np.abs(a-b))), final_lists_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(local / 'physical-parent.csv', dtype=str),
        pd.read_csv(cloud / 'results/physical-parent.csv', dtype=str), check_exact=True)
    result = dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        queries=12, acquisitions=51, actual_source_and_image_verified=True,
        all_101feature_values_and_candidate_records_bitexact=True,
        unchanged_physical_parent_exact=True, trials=compared, original_score_tolerance=tolerance,
        complete_pipeline_binding_sha256=pin['complete_pipeline_binding_sha256'],
        cloud_collection_audit_sha256=sha(OUT / 'cloud-collection-audit.json'),
        actual_CPU_only=True, actual_T4_production_smoke=False,
        engineering_USED_inputs_only=True, quality_read=False, C13_read=False,
        scoring_entries=0, qualified_for_scoring=False)
    write(target, result); print('C76 CPU CROSS-HOST FULL HEAD PARITY PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['fetch', 'audit']); args = parser.parse_args()
    (fetch if args.phase == 'fetch' else audit)()
