"""Verify actual T4 previews and replay their identical head inputs locally."""
import argparse
import json
import subprocess
import sys
from c75_port_utils import ROOT, read, write, sha, stamp
from collect_c76_source_controls import scientific_sha

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'


def fetch(trial):
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    from release_c76_production_smokes import verified
    verified()
    pushed = read(BASE / 'actual-T4-production-release' / (trial + '-push-result.json'))
    assert pushed['version'] == 1 and not pushed['error']
    kernel = pushed['actual_kernel']; folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
    dest = ROOT / 'outputs/forks' / ('c76-' + trial + '-production-v1')
    assert not (dest / 'fetch-complete.json').exists()
    meta = read(folder / 'kernel-metadata.json'); api = KaggleApi(); api.authenticate()
    assert json.loads(str(api.kernels_status(kernel)))['status'] == 'COMPLETE'
    wanted = {'submission.csv', 'c76_execution.json', 'c76_p16.csv', 'c09e_p01_submission.csv',
        'p16_execution_receipt.json', 'p16_fixture_receipt.json', 'execution_receipt.json',
        'effective_query_spectra.parquet', 'neural_candidate_features.parquet', 'neural_public.parquet', 'neural_ours.parquet',
        'c09e_external_receipt.json', 'c09e_matches.parquet', 'submission_pipeline.log',
        'gl_work/gl_out.json.meta.json', 'gl_work/gl_out.json.c76_adapter.json', 'ice_work/ice_out.json.meta.json'}
    wanted |= {'c76_scores/' + name for name in ['score-receipt.json', 'features.npy', 'feature-names.json',
        'candidates.parquet', 'physical-parent.csv', 'physical-parent-events.json', trial + '.csv', trial + '-scores.npy']}
    with api.build_kaggle_client() as client:
        req = ApiGetKernelRequest(); req.user_name = 'clarkkitchen'; req.kernel_slug = kernel.split('/')[1]
        remote = client.kernels.kernels_api_client.get_kernel(req); actual = remote.metadata
        assert actual.current_version_number == 1 and actual.is_private and actual.enable_gpu and not actual.enable_internet
        assert actual.docker_image == meta['docker_image']
        remote_hash = scientific_sha(json.loads(remote.blob.source))
        assert remote_hash == scientific_sha(read(folder / 'notebook.ipynb'))
        found = {}; token = ''; pages = 0; log = None
        while True:
            pages += 1; assert pages <= 1000
            req = ApiListKernelSessionOutputRequest(); req.user_name = 'clarkkitchen'
            req.kernel_slug = kernel.split('/')[1]; req.page_size = 100; req.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            if pages == 1: log = response.log
            for item in response.files or []:
                if item.file_name not in wanted: continue
                assert item.file_name not in found
                target = dest / item.file_name; assert target.resolve().is_relative_to(dest.resolve())
                response_file = requests.get(item.url, timeout=60); response_file.raise_for_status()
                if target.exists(): assert target.read_bytes() == response_file.content
                else: target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(response_file.content)
                found[item.file_name] = sha(target)
            token = response.next_page_token
            if not token: break
        assert set(found) == wanted, sorted(wanted-set(found))
    assert log
    log_path = dest / (kernel.split('/')[1] + '.log')
    log_path.write_text(log, encoding='utf8')
    write(dest / 'actual-remote-notebook.json', json.loads(remote.blob.source))
    write(dest / 'fetch-complete.json', dict(completed_utc=stamp(), source_sha256=sha(__file__), trial=trial,
        kernel=kernel, version=1, requested_kernel=meta['id'], remote_source_exact=True,
        remote_scientific_sha256=remote_hash, actual_original_image_exact=True,
        actual_GPU_enabled=True, private=True, notebook_sha256=sha(folder / 'notebook.ipynb'),
        listing_pages=pages, files_sha256={**found, log_path.name: sha(log_path)},
        quality_labels_read=False, C13_read=False, scoring_entries=0))
    print('C76 ACTUAL T4 PRODUCTION FILES COLLECTED', trial, flush=True)


def audit(trial):
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from c74_complete_target import module
    from audit_complete_core_health import check_events
    dest = ROOT / 'outputs/forks' / ('c76-' + trial + '-production-v1')
    target = dest / 'independent-production-audit.json'; assert not target.exists()
    fetched = read(dest / 'fetch-complete.json')
    assert fetched['source_sha256'] == sha(__file__) and fetched['trial'] == trial
    assert fetched['remote_source_exact'] and fetched['actual_original_image_exact']
    folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
    assert fetched['notebook_sha256'] == sha(folder / 'notebook.ipynb')
    for name, pin in fetched['files_sha256'].items(): assert sha(dest / name) == pin, name
    pin = read(folder / 'production-pin.json')
    receipt = read(dest / 'c76_execution.json')
    devices = receipt['actual_devices']
    assert len(devices) == 2 and all('T4' in item['name'] for item in devices)
    assert receipt['scientific_payload_sha256'] == pin['production_payload_sha256']
    assert receipt['pair_manifest_sha256'] == pin['pair_manifest_sha256']
    assert not receipt['is_hidden_rerun'] and not receipt['quality_labels_read'] and not receipt['training_performed']
    assert receipt['queries'] == 12 and receipt['valid_rows'] == 400 and receipt['exact_unprocessed_fallback_rows'] == 388
    assert receipt['trials'][trial]['changed_lists'] > 0
    execution = read(dest / 'execution_receipt.json')
    assert not execution['is_full_run'] and execution['processed_molecules'] == 12
    assert execution['source_sha256'] == pin['production_payload_sha256']
    expected = pd.read_parquet(BASE / 'local-production-preview/expected-preview-queries.parquet')
    actual = pd.read_parquet(dest / 'effective_query_spectra.parquet')
    assert len(actual) == len(expected) == 41 and actual.spectrum_id.is_unique
    aligned = actual.set_index('spectrum_id').loc[expected.spectrum_id]; original = expected.set_index('spectrum_id')
    for name in original.columns:
        for a, b in zip(aligned[name], original[name]):
            if isinstance(b, (np.ndarray, list, tuple)): np.testing.assert_array_equal(a, b, err_msg=name)
            elif pd.isna(b): assert pd.isna(a), name
            else: assert a == b, name
    events = json.loads((dest / (fetched['kernel'].split('/')[1] + '.log')).read_text(encoding='utf8'))
    events.append(dict(data=(dest / 'submission_pipeline.log').read_text(encoding='utf8', errors='replace')))
    health = check_events(events, 12)
    for name in ['gl_work/gl_out.json.meta.json', 'ice_work/ice_out.json.meta.json']:
        value = read(dest / name)
        assert value['status'] == 'ok' and value['n_mols_scored'] == value['n_mols_covered'] > 0
        assert value['n_pred_fail'] == 0 and value.get('n_feat_fail', 0) == 0
    adapter = read(dest / 'gl_work/gl_out.json.c76_adapter.json')
    assert adapter['weights_exact'] and adapter['decision_device'] == 'cpu' and adapter['cpu_threads'] == 2
    assert adapter['intensity_device'].startswith('cuda')
    p16 = read(dest / 'p16_execution_receipt.json'); fixture = read(dest / 'p16_fixture_receipt.json')
    assert not p16['is_hidden_rerun'] and p16['valid_rows'] == 400 and p16['changed_path_fixture'] == fixture
    assert fixture['real_queries'] == 30 and fixture['changed_rows'] == 7 and fixture['inserted_identities'] == 2
    assert fixture['promoted_existing_identities'] == 5 and fixture['invalid_structure_fallback_passed']
    assert fixture['exact_expected_lists'] and fixture['local_cloud_peak_score_max_abs_difference'] < 1e-12
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'
    replay = dest / 'independent-local-head-replay'; assert not replay.exists()
    reference = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    with (dest / 'independent-local-head-replay.log').open('w') as stream:
        subprocess.run([sys.executable, str(bundle / 'score_c76_complete_pair.py'), '--work', str(dest),
            '--bundle', str(bundle), '--reference', str(reference), '--output', str(replay), '--trial', trial],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    source = dest / 'c76_scores'
    np.testing.assert_array_equal(np.load(source / 'features.npy'), np.load(replay / 'features.npy'))
    pd.testing.assert_frame_equal(pd.read_parquet(source / 'candidates.parquet'), pd.read_parquet(replay / 'candidates.parquet'), check_exact=True)
    policy = read(ROOT / 'configs/c76-production-evidence-v1.json')
    np.testing.assert_allclose(np.load(source / (trial + '-scores.npy')), np.load(replay / (trial + '-scores.npy')),
        **policy['numeric_head_replay_tolerance'])
    frame = pd.read_csv(dest / 'submission.csv', dtype=str, keep_default_na=False)
    replayed = pd.read_csv(replay / (trial + '.csv'), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(frame, replayed, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(source / 'physical-parent.csv', dtype=str),
        pd.read_csv(replay / 'physical-parent.csv', dtype=str), check_exact=True)
    sample = pd.read_csv(BASE / 'production-inputs/sample_submission.csv', dtype=str, keep_default_na=False)
    parent = pd.read_csv(source / 'physical-parent.csv', dtype=str, keep_default_na=False)
    p01 = pd.read_csv(dest / 'c09e_p01_submission.csv', dtype=str, keep_default_na=False)
    assert frame.molecule_id.tolist() == sample.molecule_id.tolist() and frame.molecule_id.is_unique
    inactive = ~frame.molecule_id.isin(set(actual.molecule_id)); assert int(inactive.sum()) == 388
    pd.testing.assert_frame_equal(frame.loc[inactive], p01.loc[inactive], check_exact=True)
    identify = lru_cache(maxsize=50000)(module('c76_cloud_preview_identity', 'c09e_features.py').structure)
    for before, after in zip(parent.smiles, frame.smiles):
        a, b = before.split(';'), after.split(';')
        assert 1 <= len(a) == len(b) <= 25 and set(a) == set(b)
        assert [identify(s)[2] for s in a] == [identify(s)[2] for s in b]
        keys = [identify(s)[0] for s in b]; assert len(keys) == len(set(keys))
    files = [dest / name for name in fetched['files_sha256']] + [replay / 'score-receipt.json']
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), execution_qualified=True,
        trial=trial, actual_two_T4s=True, remote_source_exact=True, actual_original_image_exact=True,
        notebook_sha256=pin['notebook_sha256'], complete_pipeline_binding_sha256=sha(BASE / 'complete-pipeline-binding-r1.json'),
        pair_manifest_sha256=pin['pair_manifest_sha256'], actual_query_count=12, acquisitions=41, valid_rows=400,
        exact_inactive_fallback_rows=388, original_P16_changed_real_fixture_rows=7,
        same_cloud_input_local_head_replay_exact=True, all_features_bitexact=True, final_lists_exact=True,
        independent_head_arithmetic_checked=True, membership_and_formula_slots_preserved=True,
        changed_real_preview_lists=receipt['trials'][trial]['changed_lists'], core_health=health,
        local_2070_complete_core_equivalence_claimed=False, old_C75_gate_passed=False,
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        C13_still_required=True, quality_labels_read=False, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 ACTUAL T4 PRODUCTION AND SAME-CLOUD-INPUT LOCAL HEAD REPLAY PASSED', trial, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['fetch', 'audit'])
    parser.add_argument('trial', choices=['cpu', 'gpu']); args = parser.parse_args()
    (fetch if args.phase == 'fetch' else audit)(args.trial)
