"""Collect exact actual-T4 legacy/C76 runs, then audit every acquisition.

Fetch uses the Windows Kaggle SDK; audit uses the established WSL ML runtime.
Neither phase accepts query truth or computes ranking quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
SLOTS = {'legacy': 'c73-full-masked-validation', 'c76': 'c76-full-masked-validation'}
REQUIRED = ['c41_report.json', 'c41_t4_allocation.json', 'c41_gpu1_preflight.json',
    'c41_p01.csv', 'c41_p14.csv', 'c41_queries.parquet', 'c41_neural_p01.parquet',
    'neural_public.parquet', 'neural_ours.parquet', 'frozen_candidate_evidence.parquet',
    'outsider_neural_features.parquet', 'effective_query_spectra.parquet',
    'c41_main_reference_mask_audit.json', 'c41_engine2_reference_mask_receipt.json',
    'c41_legacy_compatibility.json', 'execution_receipt.json', 'submission_pipeline.log',
    'gl_work/gl_out.json.meta.json', 'ice_work/ice_out.json.meta.json']


def verified():
    binding_path = BASE / 'complete-pipeline-binding.json'
    binding = read(binding_path)
    for name, pin in binding['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()


def scientific_sha(nb):
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    return hashlib.sha256(json.dumps(code, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def fetch(arm):
    verified()
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    slot = SLOTS[arm]; folder = ROOT / 'kaggle/forks' / slot
    dest = ROOT / 'outputs/forks' / (slot + '-v1')
    assert not (dest / 'fetch-complete.json').exists()
    api = KaggleApi(); api.authenticate(); kernel = 'clarkkitchen/casmi26-' + slot
    assert json.loads(str(api.kernels_status(kernel)))['status'] == 'COMPLETE'
    wanted = set(REQUIRED)
    if arm == 'c76': wanted.add('gl_work/gl_out.json.c76_adapter.json')
    dest.mkdir(parents=True, exist_ok=True)
    with api.build_kaggle_client() as client:
        req = ApiGetKernelRequest(); req.user_name = 'clarkkitchen'; req.kernel_slug = kernel.split('/')[1]
        remote = client.kernels.kernels_api_client.get_kernel(req)
        meta = remote.metadata; expected = read(folder / 'kernel-metadata.json')
        assert meta.current_version_number == 1 and meta.is_private and meta.enable_gpu and not meta.enable_internet
        assert meta.docker_image == expected['docker_image']
        remote_source = json.loads(remote.blob.source)
        remote_sha = scientific_sha(remote_source)
        assert remote_sha == scientific_sha(read(folder / 'notebook.ipynb'))
        found = {}; pages = 0; token = ''; log = None
        while True:
            pages += 1; assert pages <= 1000, 'Reconcile listing volume; never assume completeness'
            req = ApiListKernelSessionOutputRequest(); req.user_name = 'clarkkitchen'
            req.kernel_slug = kernel.split('/')[1]; req.page_size = 100; req.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            if pages == 1: log = response.log
            for item in response.files or []:
                if item.file_name not in wanted: continue
                assert item.file_name not in found, 'Duplicate output path'
                target = dest / item.file_name
                assert target.resolve().is_relative_to(dest.resolve())
                download = requests.get(item.url, timeout=60); download.raise_for_status()
                content = download.content
                if target.exists(): assert target.read_bytes() == content, 'Preserve and inspect conflicting partial output'
                else: target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
                found[item.file_name] = sha(target)
            token = response.next_page_token
            if not token: break
        assert set(found) == wanted, sorted(wanted - set(found))
    assert log
    log_path = dest / (kernel.split('/')[1] + '.log')
    if log_path.exists(): assert log_path.read_text(encoding='utf8') == log
    else: log_path.write_text(log, encoding='utf8')
    write(dest / 'actual-remote-notebook.json', remote_source)
    write(dest / 'fetch-complete.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        kernel=kernel, version=1, arm=arm, remote_source_exact=True, remote_scientific_sha256=remote_sha,
        actual_original_image_exact=True, private=True, actual_GPU_enabled=True,
        notebook_sha256=sha(folder / 'notebook.ipynb'), listing_pages=pages,
        files_sha256={**found, log_path.name: sha(log_path)}, quality_read=False, C13_read=False))
    print('C76 ACTUAL SOURCE FILES COLLECTED', arm, pages, 'pages; independent audit still required', flush=True)


def audit(arm):
    verified()
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from audit_complete_core_health import check_events
    from c74_complete_target import module
    slot = SLOTS[arm]; folder = ROOT / 'kaggle/forks' / slot
    dest = ROOT / 'outputs/forks' / (slot + '-v1')
    assert not (dest / 'collection-audit.json').exists()
    fetched = read(dest / 'fetch-complete.json')
    assert fetched['source_sha256'] == sha(__file__) and fetched['arm'] == arm
    assert fetched['notebook_sha256'] == sha(folder / 'notebook.ipynb')
    assert fetched['remote_source_exact'] and fetched['actual_original_image_exact']
    for name, pin in fetched['files_sha256'].items(): assert sha(dest / name) == pin, name
    report = read(dest / 'c41_report.json')
    query_pin = read(ROOT / 'kaggle/forks/c73-full-masked-validation/validation-pin.json')
    assert report['queries'] == 512 and report['baseline_replay_exact']
    assert not report['metrics_computed'] and not report['query_labels_mounted'] and not report['scoring_submission']
    assert report['validation_contract_sha256'] == query_pin['query_contract_sha256']
    assert report['baseline_sha256'] == sha(dest / 'c41_p01.csv') and report['p14_receipt']['input_errors'] == 0
    devices = read(dest / 'c41_t4_allocation.json')['devices']
    assert len(devices) == 2 and all('T4' in row['name'] for row in devices)
    preflight = read(dest / 'c41_gpu1_preflight.json')
    assert preflight['dreams_finite'] and preflight['molformer_finite'] and 'T4' in preflight['name']
    execution = read(dest / 'execution_receipt.json')
    assert execution['is_full_run'] and execution['processed_molecules'] == 512
    for name in ['c41_main_reference_mask_audit.json', 'c41_engine2_reference_mask_receipt.json']:
        receipt = read(dest / name)
        assert receipt['excluded_reference_rows'] > 0 and receipt['forbidden_rows_retained'] == 0
        if 'main' in name: assert receipt['window_calls'] > 0 and receipt['engine_representative_builds'] > 0 and receipt['engine_representatives_removed'] > 0
    compat = read(dest / 'c41_legacy_compatibility.json'); assert set(compat) == {'v3', 'v4'}
    for item in compat.values():
        assert set(item['before_sha256']) == set(item['after_sha256']) == {'chem.py', 'frag.py'}
        assert all(item['before_sha256'][n] != item['after_sha256'][n] for n in item['before_sha256'])
    query_path = ROOT / 'outputs/publish/c72-clean-target-holdout-20261112/validation-queries.parquet'
    panel = read(ROOT / 'outputs/local/c72-clean-target-holdout-20261112/validation-freeze.json')
    files = {name.replace('\\', '/'): pin for name, pin in panel['files_sha256'].items()}
    assert files[query_path.relative_to(ROOT).as_posix()] == sha(query_path)
    original, actual = pd.read_parquet(query_path), pd.read_parquet(dest / 'c41_queries.parquet')
    assert len(original) == len(actual) == 2215 and actual.molecule_id.nunique() == 512 and actual.spectrum_id.is_unique
    assert set(original.columns) <= set(actual.columns)
    assert not {'smiles', 'normalized_smiles', 'key', 'inchikey14', 'formula'} & set(actual.columns)
    actual = actual.set_index('spectrum_id').loc[original.spectrum_id].reset_index()
    for col in original.columns:
        for a, b in zip(original[col], actual[col]):
            if isinstance(a, (np.ndarray, list, tuple)): np.testing.assert_array_equal(a, b, err_msg=col)
            elif pd.isna(a): assert pd.isna(b), col
            else: assert a == b, col
    modes = actual.groupby('molecule_id').ionization_mode.agg(lambda v: 'mixed' if v.nunique() > 1 else v.iloc[0]).value_counts().to_dict()
    assert modes == {'positive': 353, 'mixed': 124, 'negative': 35}
    identity = lru_cache(maxsize=50000)(module('c76_source_official_identity', 'c09e_features.py').structure)
    for name in ['c41_p01.csv', 'c41_p14.csv']:
        frame = pd.read_csv(dest / name, dtype=str, keep_default_na=False)
        assert frame.molecule_id.is_unique and frame.molecule_id.tolist() == list(dict.fromkeys(original.molecule_id))
        for text in frame.smiles:
            keys = [identity(s)[0] for s in text.split(';')]
            assert 1 <= len(keys) <= 25 and len(set(keys)) == len(keys)
    for name in ['c41_neural_p01.parquet', 'neural_public.parquet', 'neural_ours.parquet']:
        frame = pd.read_parquet(dest / name)
        assert set(frame.molecule_id) == set(original.molecule_id) and frame.neural_ok.astype(bool).all()
        assert not frame.duplicated(['molecule_id', 'metric_key']).any() and np.isfinite(frame.neural_score).all()
    for name in ['gl_work/gl_out.json.meta.json', 'ice_work/ice_out.json.meta.json']:
        meta = read(dest / name)
        assert meta['status'] == 'ok' and meta['n_mols_scored'] == meta['n_mols_covered'] > 0
        assert meta['n_pred_fail'] == 0 and meta.get('n_feat_fail', 0) == 0
    if arm == 'c76':
        adapter = read(dest / 'gl_work/gl_out.json.c76_adapter.json')
        assert adapter['weights_exact'] and adapter['decision_device'] == 'cpu' and adapter['cpu_threads'] == 2
        assert adapter['intensity_device'].startswith('cuda')
    events = json.loads((dest / ('casmi26-' + slot + '.log')).read_text(encoding='utf8'))
    events.append(dict(data=(dest / 'submission_pipeline.log').read_text(encoding='utf8', errors='replace')))
    health = check_events(events, 512)
    write(dest / 'acquisition-fidelity-audit.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        queries=512, acquisitions=2215, original_query_sha256=sha(query_path),
        actual_query_modes=modes, all_original_acquisitions_retained=True,
        all_original_metadata_and_peaks_exact=True, quality_labels_read=False, C13_read=False))
    write(dest / 'independent-core-health-audit.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        passed=True, queries=512, **health, quality_labels_read=False))
    write(dest / 'collection-audit.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        kernel=fetched['kernel'], version=1, arm=arm, queries=512, acquisitions=2215,
        actual_two_T4s=True, remote_source_exact=True, actual_original_image_exact=True,
        baseline_replay_exact=True, forbidden_reference_rows=0, formate_source_copy_verified=True,
        complete_pipeline_binding_sha256=sha(BASE / 'complete-pipeline-binding.json'),
        files_sha256=fetched['files_sha256'], quality_metrics_read=False, C13_read=False, scoring_entries=0))
    print('C76 COMPLETE ACTUAL-T4 SOURCE/ACQUISITION/CORE AUDIT PASSED', arm, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['fetch', 'audit'])
    parser.add_argument('arm', choices=['legacy', 'c76']); args = parser.parse_args()
    (fetch if args.phase == 'fetch' else audit)(args.arm)
