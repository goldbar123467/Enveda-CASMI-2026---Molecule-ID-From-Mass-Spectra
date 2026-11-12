"""Verify the completed core/P16 prefix before repairing the failed head ABI."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp
from audit_complete_core_health import check_events


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    out = base / 'interface-repair-r1'; prior = base / 'local-production-preview'
    target = out / 'completed-prefix-audit.json'; assert not target.exists()
    assert read(prior / 'supervisor-failed.json')['status'] == 'failed'
    assert 'head stage failed' in read(prior / 'run-failed.json')['error']
    assert not (prior / 'run-complete.json').exists()
    pin = read(prior / 'preview-freeze.json'); working = Path(pin['working'])
    assert not (working / 'submission.csv').exists(), 'Failed required head must leave no scoring file'
    assert read(prior / 'run-progress.json')['outer_cell'] == 28
    assert sha(prior / 'notebook.ipynb') == pin['local_notebook_sha256']
    health = check_events([dict(data=p.read_text(encoding='utf8', errors='replace'))
        for p in [prior / 'pipeline.log', working / 'submission_pipeline.log']], 12)
    execution = read(working / 'execution_receipt.json')
    assert not execution['is_full_run'] and execution['processed_molecules'] == 12
    assert execution['source_sha256'] == pin['local_payload_sha256']
    expected = pd.read_parquet(prior / 'expected-preview-queries.parquet')
    q = pd.read_parquet(working / 'effective_query_spectra.parquet')
    assert len(q) == len(expected) == 41 and q.molecule_id.nunique() == 12 and q.spectrum_id.is_unique
    a = q.set_index('spectrum_id').loc[expected.spectrum_id]; b = expected.set_index('spectrum_id')
    for name in b:
        for x, y in zip(a[name], b[name]):
            if isinstance(y, (list, tuple, np.ndarray)): np.testing.assert_array_equal(x, y)
            elif pd.isna(y): assert pd.isna(x)
            else: assert x == y
    receipt = read(working / 'c09e_external_receipt.json')
    assert receipt['required_stage_completed'] and receipt['valid_rows'] == 400
    assert sha(working / 'c09e_p01_submission.csv') == receipt['baseline_sha256']
    assert sha(working / 'c76_p16.csv') == receipt['csv_sha256']
    assert sha(working / 'effective_query_spectra.parquet') == receipt['query_sha256']
    p16 = read(working / 'p16_execution_receipt.json'); fixture = read(working / 'p16_fixture_receipt.json')
    assert not p16['is_hidden_rerun'] and p16['merge_status'] == 'applied'
    assert p16['changed_path_fixture'] == fixture and fixture['real_queries'] == 30 and fixture['changed_rows'] == 7
    assert fixture['inserted_identities'] == 2 and fixture['promoted_existing_identities'] == 5
    assert fixture['exact_expected_lists'] and fixture['invalid_structure_fallback_passed']
    neural = {}; active = set(q.molecule_id)
    for name in ['neural_candidate_features.parquet', 'neural_public.parquet', 'neural_ours.parquet']:
        frame = pd.read_parquet(working / name); included = frame[frame.molecule_id.isin(active)]
        assert set(included.molecule_id) == active and included.neural_ok.astype(bool).all()
        assert np.isfinite(included.neural_score).all() and frame.molecule_id.nunique() == 400
        assert not frame.duplicated(['molecule_id', 'metric_key']).any()
        neural[name] = dict(all_query_ids=400, active_query_ids=12, inactive_query_ids=388,
            all_rows=len(frame), active_rows=len(included), every_active_neural_row_healthy=True)
    for name in ['gl_work/gl_out.json.meta.json', 'ice_work/ice_out.json.meta.json']:
        value = read(working / name)
        assert value['status'] == 'ok' and value['n_mols_scored'] == value['n_mols_covered'] > 0
        assert value['n_pred_fail'] == 0 and value.get('n_feat_fail', 0) == 0
    names = ['execution_receipt.json', 'c09e_external_receipt.json', 'p16_execution_receipt.json',
        'p16_fixture_receipt.json', 'c09e_p01_submission.csv', 'c76_p16.csv', 'effective_query_spectra.parquet',
        'c09e_matches.parquet', 'neural_candidate_features.parquet', 'neural_public.parquet', 'neural_ours.parquet',
        'c09e_fixture_baseline.csv', 'c09e_fixture_final.csv', 'c09e_fixture_queries.parquet', 'c09e_fixture_matches.parquet',
        'gl_work/gl_out.json.meta.json', 'gl_work/gl_out.json.c76_adapter.json', 'ice_work/ice_out.json.meta.json']
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), completed_prefix_passed=True,
        original_full_run_passed=False, failure_confined_to_new_head_active_ID_guard=True,
        old_full_run_failure_sha256=sha(prior / 'run-failed.json'),
        preview_freeze_sha256=sha(prior / 'preview-freeze.json'), working=str(working),
        preview_queries=12, acquisitions=41, output_rows=400, inactive_rows=388,
        all_original_preview_metadata_and_peaks_exact=True, core_health=health, neural_evidence=neural,
        original_P16_real_changed_fixture_rows=7,
        reusable_unchanged_files_sha256={name: sha(working / name) for name in names},
        no_GPU_core_restart_needed=True, no_head_output_or_scoring_file_created=True,
        quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 COMPLETED CORE/P16 PREFIX VERIFIED; HEAD-ONLY RECOVERY IS CONCRETE', flush=True)


if __name__ == '__main__': main()
