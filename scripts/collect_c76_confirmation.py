"""Collect/audit actual C13 core execution without opening query truth."""
import argparse
import hashlib
import json
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import BINDING
from c76_confirmation_preparation import ARMS, CORE_PINS, ORIGINAL, SLOTS, read_queries
from c76_confirmation_inference import CLOUDS, PLAN, RELEASE, verify_plan
from submit_c09_external_once import scientific_sha

REQUIRED = ['c41_report.json', 'c41_t4_allocation.json', 'c41_gpu1_preflight.json',
    'c41_p01.csv', 'c41_p14.csv', 'c41_queries.parquet', 'c41_neural_p01.parquet',
    'neural_public.parquet', 'neural_ours.parquet', 'frozen_candidate_evidence.parquet',
    'outsider_neural_features.parquet', 'effective_query_spectra.parquet', 'eng_lists.json', 'pc_lists.json',
    'c41_main_reference_mask_audit.json', 'c41_engine2_reference_mask_receipt.json',
    'c41_legacy_compatibility.json', 'c76_c13_channel_ids.json', 'execution_receipt.json', 'submission_pipeline.log',
    'gl_work/gl_out.json.meta.json', 'ice_work/ice_out.json.meta.json']


def fetch(arm):
    verify_plan()
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    push = read(RELEASE / (arm + '-push-result.json')); folder = ROOT / 'kaggle/forks' / SLOTS[arm]
    meta = read(folder / 'kernel-metadata.json'); kernel = meta['id']; dest = CLOUDS[arm]
    assert push['kernel'] == kernel and push['version'] == 1 and not push['error']
    assert not (dest / 'fetch-complete.json').exists()
    api = KaggleApi(); api.authenticate(); assert json.loads(str(api.kernels_status(kernel)))['status'] == 'COMPLETE'
    wanted = set(REQUIRED)
    if arm == 'c76': wanted.add('gl_work/gl_out.json.c76_adapter.json')
    with api.build_kaggle_client() as client:
        req = ApiGetKernelRequest(); req.user_name, req.kernel_slug = kernel.split('/')
        remote = client.kernels.kernels_api_client.get_kernel(req); actual = remote.metadata
        assert actual.current_version_number == 1 and actual.is_private and actual.enable_gpu and not actual.enable_internet
        assert actual.docker_image == meta['docker_image']
        notebook = json.loads(remote.blob.source); remote_sha = scientific_sha(notebook)
        assert remote_sha == scientific_sha(read(folder / 'notebook.ipynb'))
        found = {}; token = ''; tokens = {''}; log = None; pages = 0
        while True:
            pages += 1; assert pages <= 1000
            req = ApiListKernelSessionOutputRequest(); req.user_name, req.kernel_slug = kernel.split('/')
            req.page_size = 100; req.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            if pages == 1: log = response.log
            for item in response.files or []:
                if item.file_name not in wanted: continue
                assert item.file_name not in found
                path = dest / item.file_name; assert path.resolve().is_relative_to(dest.resolve())
                download = requests.get(item.url, timeout=60); download.raise_for_status()
                if path.exists(): assert path.read_bytes() == download.content
                else: path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(download.content)
                found[item.file_name] = sha(path)
            token = response.next_page_token
            if not token: break
            assert token not in tokens; tokens.add(token)
        assert set(found) == wanted, sorted(wanted-set(found))
    assert log
    log_path = dest / (kernel.split('/')[1] + '.log')
    if log_path.exists(): assert log_path.read_text(encoding='utf8') == log
    else: log_path.write_text(log, encoding='utf8')
    write(dest / 'actual-remote-notebook.json', notebook)
    write(dest / 'fetch-complete.json', dict(completed_utc=stamp(), source_sha256=sha(__file__), arm=arm,
        kernel=kernel, version=1, inference_plan_sha256=sha(PLAN), notebook_sha256=sha(folder / 'notebook.ipynb'),
        remote_source_exact=True, remote_scientific_sha256=remote_sha, actual_original_image_exact=True,
        private=True, actual_GPU_enabled=True, listing_pages=pages,
        files_sha256={**found, log_path.name: sha(log_path)}, query_labels_read=False, scoring_entries=0))
    print('C76 RESERVED ACTUAL CORE OUTPUTS FETCHED', arm, flush=True)


def audit(arm):
    verify_plan()
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from audit_c59_core_health import check_execution, check_channel_keys
    from c74_complete_target import module
    dest = CLOUDS[arm]; target = dest / 'collection-audit.json'; assert not target.exists()
    fetched = read(dest / 'fetch-complete.json'); folder = ROOT / 'kaggle/forks' / SLOTS[arm]
    assert fetched['source_sha256'] == sha(__file__) and fetched['arm'] == arm and fetched['inference_plan_sha256'] == sha(PLAN)
    assert fetched['notebook_sha256'] == sha(folder / 'notebook.ipynb')
    assert fetched['remote_source_exact'] and fetched['actual_original_image_exact'] and fetched['actual_GPU_enabled'] and fetched['private']
    for name, pin in fetched['files_sha256'].items(): assert sha(dest / name) == pin
    report = read(dest / 'c41_report.json'); pin = read(folder / 'validation-pin.json')
    assert report['queries'] == 120 and report['baseline_replay_exact'] and not report['metrics_computed']
    assert not report['query_labels_mounted'] and not report['recipe_selection'] and not report['scoring_submission']
    assert report['validation_contract_sha256'] == pin['query_contract_sha256']
    assert report['baseline_sha256'] == sha(dest / 'c41_p01.csv') and report['p14_receipt']['input_errors'] == 0
    execution = read(dest / 'execution_receipt.json')
    assert execution['is_full_run'] and execution['processed_molecules'] == 120 and execution['source_sha256'] == CORE_PINS[arm]
    devices = read(dest / 'c41_t4_allocation.json')['devices']
    assert len(devices) == 2 and all('T4' in row['name'] for row in devices)
    preflight = read(dest / 'c41_gpu1_preflight.json')
    assert 'T4' in preflight['name'] and preflight['dreams_finite'] and preflight['molformer_finite']
    original = read_queries(); actual = pd.read_parquet(dest / 'c41_queries.parquet')
    assert len(actual) == actual.spectrum_id.nunique() == actual.molecule_id.nunique() == 120
    assert set(original.columns) <= set(actual.columns)
    assert not {'smiles', 'normalized_smiles', 'key', 'inchikey14', 'formula', 'mass', 'source_number'} & set(actual.columns)
    actual = actual.set_index('spectrum_id').loc[original.spectrum_id].reset_index()
    for column in original.columns:
        for a, b in zip(original[column], actual[column]):
            if isinstance(a, (np.ndarray, list, tuple)): np.testing.assert_array_equal(a, b, err_msg=column)
            elif pd.isna(a): assert pd.isna(b), column
            else: assert a == b, column
    assert actual.ionization_mode.value_counts().to_dict() == {'positive': 60, 'negative': 60}
    identity = lru_cache(maxsize=50000)(module('c76_reserved_core_identity', 'c09e_features.py').structure)
    ids = original.molecule_id.tolist()
    for name in ['c41_p01.csv', 'c41_p14.csv']:
        frame = pd.read_csv(dest / name, dtype=str, keep_default_na=False)
        assert frame.molecule_id.tolist() == ids and frame.molecule_id.is_unique
        for text in frame.smiles:
            keys = [identity(s)[0] for s in text.split(';')]; assert 1 <= len(keys) <= 25 and len(keys) == len(set(keys))
    for name in ['c41_neural_p01.parquet', 'neural_public.parquet', 'neural_ours.parquet']:
        frame = pd.read_parquet(dest / name)
        assert set(frame.molecule_id) == set(ids) and frame.neural_ok.astype(bool).all() and np.isfinite(frame.neural_score).all()
        assert not frame.duplicated(['molecule_id', 'metric_key']).any()
    support = read(ROOT / 'outputs/local/c59-c13-compatibility-20261112/channel-support-freeze.json')
    channel = read(dest / 'c76_c13_channel_ids.json')
    assert set(channel['input_IDs']) == set(ids) and not channel['query_selection_or_token_rewriting'] and not channel['quality_labels_read']
    engine, pc = read(dest / 'eng_lists.json'), read(dest / 'pc_lists.json')
    assert set(channel['Engine2_IDs']) == set(engine) and set(channel['PubChem_IDs']) == set(pc)
    check_channel_keys(support['all_query_ids'], support['pubchem_supported_ids'], engine, pc, ids)
    events = read(dest / (fetched['kernel'].split('/')[1] + '.log'))
    events.append(dict(data=(dest / 'submission_pipeline.log').read_text(encoding='utf8', errors='replace')))
    health = check_execution(events, 120, 119)
    for name in ['c41_main_reference_mask_audit.json', 'c41_engine2_reference_mask_receipt.json']:
        value = read(dest / name); assert value['excluded_reference_rows'] > 0 and value['forbidden_rows_retained'] == 0
        if 'main' in name: assert value['window_calls'] > 0 and value['engine_representative_builds'] > 0 and value['engine_representatives_removed'] > 0
    from legacy_formate_compat import patch_chem_source, patch_frag_source
    source = ROOT / 'outputs/local/c37-selection-trace-20261111/pinned-public-source'
    before = {name: sha(source / name) for name in ['chem.py', 'frag.py']}
    after = {name: hashlib.sha256(fn((source / name).read_text(encoding='utf8')).encode()).hexdigest()
        for name, fn in [('chem.py', patch_chem_source), ('frag.py', patch_frag_source)]}
    compatibility = read(dest / 'c41_legacy_compatibility.json'); assert set(compatibility) == {'v3', 'v4'}
    for value in compatibility.values(): assert value['before_sha256'] == before and value['after_sha256'] == after
    for name in ['gl_work/gl_out.json.meta.json', 'ice_work/ice_out.json.meta.json']:
        value = read(dest / name)
        assert value['status'] == 'ok' and value['n_mols_scored'] == value['n_mols_covered'] > 0
        assert value['n_pred_fail'] == 0 and value.get('n_feat_fail', 0) == 0
    if arm == 'c76':
        adapter = read(dest / 'gl_work/gl_out.json.c76_adapter.json')
        assert adapter['weights_exact'] and adapter['decision_device'] == 'cpu' and adapter['cpu_threads'] == 2
        assert adapter['intensity_device'].startswith('cuda')
    write(dest / 'independent-core-health-audit.json', dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        queries=120, pubchem_supported=119, **health, query_labels_read=False))
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), arm=arm, kernel=fetched['kernel'], version=1,
        complete_pipeline_binding_sha256=sha(BINDING), inference_plan_sha256=sha(PLAN), queries=120, acquisitions=120,
        actual_two_T4s=True, remote_source_exact=True, actual_original_image_exact=True, baseline_replay_exact=True,
        all_original_metadata_peaks_and_unknown_CE_exact=True, original_trailing_space_token_preserved=True,
        exact_PC_supported_IDs=119, exact_Engine2_IDs=120, forbidden_reference_rows=0,
        files_sha256={**fetched['files_sha256'], 'independent-core-health-audit.json': sha(dest / 'independent-core-health-audit.json')},
        quality_labels_read=False, scoring_entries=0))
    print('C76 RESERVED ALL120 ACQUISITIONS / EXACT119 PC / CORE AUDIT PASSED', arm, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['fetch', 'audit']); parser.add_argument('arm', choices=ARMS)
    args = parser.parse_args(); {'fetch': fetch, 'audit': audit}[args.phase](args.arm)
