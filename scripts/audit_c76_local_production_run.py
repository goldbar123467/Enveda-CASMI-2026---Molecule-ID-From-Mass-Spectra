"""Audit the full local preview, inactive fallback and changed physical path."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
import torch
import importlib.util
import types
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp
from audit_complete_core_health import check_events


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'; out = base / 'local-production-preview'
    target = out / 'independent-run-audit.json'; assert not target.exists()
    pin = read(out / 'preview-freeze.json'); plan = read(out / 'launch-plan.json')
    for name, digest in plan['files_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert read(out / 'supervisor-complete.json')['status'] == 'complete'
    done = read(out / 'run-complete.json'); working = Path(done['working'])
    assert done['source_sha256'] == sha(ROOT / 'scripts/run_c76_local_production.py')
    assert done['preview_queries'] == 12 and done['preview_acquisitions'] == pin['preview_acquisitions']
    assert done['actual_single_2070'] and not done['actual_T4_evidence']
    allocation = read(working / 'c76_local_allocation.json')
    assert allocation['actual_single_2070'] and not allocation['actual_two_T4s']
    assert len(allocation['devices']) == 1 and '2070' in allocation['devices'][0]['name']
    execution = read(working / 'execution_receipt.json')
    assert not execution['is_full_run'] and execution['processed_molecules'] == 12
    assert execution['source_sha256'] == pin['local_payload_sha256']
    health = check_events([dict(data=p.read_text(encoding='utf8', errors='replace'))
        for p in [out / 'pipeline.log', working / 'submission_pipeline.log']], 12)
    expected = pd.read_parquet(out / 'expected-preview-queries.parquet')
    q = pd.read_parquet(working / 'effective_query_spectra.parquet')
    assert len(q) == len(expected) == pin['preview_acquisitions'] and q.molecule_id.nunique() == 12
    assert q.spectrum_id.is_unique and set(q.spectrum_id) == set(expected.spectrum_id)
    aligned = q.set_index('spectrum_id').loc[expected.spectrum_id]
    original = expected.set_index('spectrum_id')
    for col in original.columns:
        for a, b in zip(aligned[col], original[col]):
            if isinstance(b, (np.ndarray, list, tuple)): np.testing.assert_array_equal(a, b, err_msg=col)
            elif pd.isna(b): assert pd.isna(a), col
            else: assert a == b, col
    assert set(q.molecule_id) == set(pin['preview_ids'])
    sample = pd.read_csv(base / 'production-inputs/sample_submission.csv', dtype=str, keep_default_na=False)
    p01 = pd.read_csv(working / 'c09e_p01_submission.csv', dtype=str, keep_default_na=False)
    p16 = pd.read_csv(working / 'c76_p16.csv', dtype=str, keep_default_na=False)
    assert sample.molecule_id.tolist() == p01.molecule_id.tolist() == p16.molecule_id.tolist()
    assert len(sample) == 400
    inactive = ~p01.molecule_id.isin(set(q.molecule_id))
    assert int(inactive.sum()) == 388
    pd.testing.assert_frame_equal(p01.loc[inactive], p16.loc[inactive], check_exact=True)
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-20261112'
    manifest = read(bundle / 'pair-manifest.json')
    for name, digest in manifest['files_sha256'].items(): assert sha(bundle / name) == digest, name
    ranker = module('c76_preview_original_ranker', bundle / 'c67_ranker.py')
    features = module('c76_preview_official_identity', bundle / 'identity_features.py')
    compatibility = module('c76_preview_compatibility', bundle / 'legacy_formate_compat.py')
    identity = lru_cache(maxsize=50000)(features.structure)
    for name in ['neural_candidate_features.parquet', 'neural_public.parquet', 'neural_ours.parquet']:
        frame = pd.read_parquet(working / name)
        assert set(frame.molecule_id) == set(q.molecule_id) and frame.neural_ok.astype(bool).all()
        assert not frame.duplicated(['molecule_id', 'metric_key']).any() and np.isfinite(frame.neural_score).all()
    neural = {}
    for name in ['gl_work/gl_out.json.meta.json', 'ice_work/ice_out.json.meta.json']:
        receipt = read(working / name)
        assert receipt['status'] == 'ok' and receipt['n_mols_scored'] == receipt['n_mols_covered'] > 0
        assert receipt['n_pred_fail'] == 0 and receipt.get('n_feat_fail', 0) == 0
        neural[name] = dict(supported_queries=receipt['n_mols_scored'], prediction_failures=0)
    adapter = read(working / 'gl_work/gl_out.json.c76_adapter.json')
    assert adapter['weights_exact'] and adapter['decision_device'] == 'cpu' and adapter['cpu_threads'] == 2
    assert adapter['intensity_device'].startswith('cuda')
    p16_receipt = read(working / 'p16_execution_receipt.json')
    fixture_receipt = read(working / 'p16_fixture_receipt.json')
    assert not p16_receipt['is_hidden_rerun'] and p16_receipt['valid_rows'] == 400
    assert p16_receipt['changed_path_fixture'] == fixture_receipt
    assert fixture_receipt['real_queries'] == 30 and fixture_receipt['changed_rows'] == 7
    assert fixture_receipt['inserted_identities'] == 2 and fixture_receipt['promoted_existing_identities'] == 5
    assert fixture_receipt['exact_expected_lists'] and fixture_receipt['local_cloud_peak_score_max_abs_difference'] < 1e-12
    assert fixture_receipt['invalid_structure_fallback_passed']
    # Exercise the original physical adapter on seven REAL reference promotions,
    # even if the unmodified public preview itself has no eligible promotions.
    legacy = bundle / 'legacy_chem_source.py'
    patched = compatibility.patch_chem_source(legacy.read_text(encoding='utf8'))
    chemistry = types.ModuleType('c76_preview_mass'); exec(compile(patched, str(legacy), 'exec'), chemistry.__dict__)
    old = pd.read_csv(working / 'c09e_fixture_baseline.csv', dtype=str, keep_default_na=False)
    new = pd.read_csv(working / 'c09e_fixture_final.csv', dtype=str, keep_default_na=False)
    fq = pd.read_parquet(working / 'c09e_fixture_queries.parquet')
    fm = pd.read_parquet(working / 'c09e_fixture_matches.parquet')
    reference = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    refs = pd.read_parquet(reference / 'ext_meta.parquet')
    chosen, events = ranker.physical_parent(old, new, fq, fm, refs, chemistry.neutral_mass)
    assert len(events) == 7 and len(fq) == 113
    rebuilt = new.copy(); accepted = reverted = 0
    aliases = {'[M+H-H2O]+': '[M-H2O+H]+', '[M+H-2H2O]+': '[M-2H2O+H]+'}
    def inferred(mz, token):
        data = chemistry.ADDUCTS.get(compatibility.normalize_legacy_adduct(token))
        if data is None: return float('nan')
        number, charge, delta, _ = data
        return (float(mz)*charge-delta)/number
    for index, row in old.iterrows():
        if row.smiles == new.smiles.iloc[index]: continue
        group = fq[fq.molecule_id.eq(row.molecule_id)]
        values = np.array([inferred(r.precursor_mz, r.adduct) for r in group.itertuples(index=False)])
        values = values[np.isfinite(values)]
        query_mass = float(np.median(values)) if len(values) else float('nan')
        hit = fm[fm.molecule_id.eq(row.molecule_id) & fm.view.eq('all_reference')].iloc[0]
        ref = refs.iloc[int(hit.reference_row)]
        ref_mass = inferred(ref.precursor_mz, aliases.get(ref.adduct, ref.adduct))
        keep = bool(np.isfinite(query_mass) and np.isfinite(ref_mass) and abs(query_mass-ref_mass) <= max(.002, query_mass*1e-5))
        if keep: accepted += 1
        else: reverted += 1; rebuilt.at[index, 'smiles'] = row.smiles
    pd.testing.assert_frame_equal(chosen, rebuilt, check_exact=True)
    chosen.to_csv(out / 'physical-fixture-checked.csv', index=False)
    write(out / 'physical-fixture-events.json', events)
    trials = {}; parent = pd.read_csv(working / 'c76_scores/physical-parent.csv', dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(parent, pd.read_csv(working / 'c76_gpu_scores/physical-parent.csv', dtype=str), check_exact=True)
    pd.testing.assert_frame_equal(parent.loc[inactive], p01.loc[inactive], check_exact=True)
    np.testing.assert_array_equal(np.load(working / 'c76_scores/features.npy'), np.load(working / 'c76_gpu_scores/features.npy'))
    pd.testing.assert_frame_equal(pd.read_parquet(working / 'c76_scores/candidates.parquet'),
        pd.read_parquet(working / 'c76_gpu_scores/candidates.parquet'), check_exact=True)
    for trial, directory, receipt_name in [('cpu', 'c76_scores', 'c76_execution.json'), ('gpu', 'c76_gpu_scores', 'c76_gpu_execution.json')]:
        receipt = read(working / receipt_name)
        assert receipt['source_sha256'] == sha(bundle / 'score_c76_complete_pair.py')
        assert receipt['pair_manifest_sha256'] == sha(bundle / 'pair-manifest.json')
        assert receipt['queries'] == 12 and receipt['valid_rows'] == 400 and receipt['exact_unprocessed_fallback_rows'] == 388
        assert not receipt['training_performed'] and not receipt['quality_labels_read']
        assert receipt['scientific_payload_sha256'] == pin['local_payload_sha256']
        assert receipt['trials'][trial]['independent_scores_and_lists_exact']
        frame = pd.read_csv(working / ('c76_' + trial + '_submission.csv'), dtype=str, keep_default_na=False)
        assert frame.molecule_id.tolist() == sample.molecule_id.tolist()
        pd.testing.assert_frame_equal(frame.loc[inactive], p01.loc[inactive], check_exact=True)
        assert sha(working / ('c76_' + trial + '_submission.csv')) == receipt['files_sha256'][trial + '.csv']
        changes = 0
        for a, b in zip(parent.smiles, frame.smiles):
            before, after = a.split(';'), b.split(';')
            assert 1 <= len(before) == len(after) <= 25 and set(before) == set(after)
            assert [identity(s)[2] for s in before] == [identity(s)[2] for s in after]
            keys = [identity(s)[0] for s in after]; assert len(keys) == len(set(keys))
            changes += before != after
        assert changes == receipt['trials'][trial]['changed_lists'] > 0
        trials[trial] = dict(changed_lists=changes, exact_inactive_fallback_rows=388,
            all_membership_formula_slots_and_official_identities_valid=True, csv_sha256=sha(working / ('c76_' + trial + '_submission.csv')))
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        run_complete_sha256=sha(out / 'run-complete.json'), launch_plan_sha256=sha(out / 'launch-plan.json'),
        preview_queries=12, preview_acquisitions=len(q), all_original_preview_metadata_and_peaks_exact=True,
        output_rows=400, inactive_rows_preserved_exact=388, core_health=health, neural=neural,
        original_real_P16_regression_queries=30, original_real_P16_changed_lists=7,
        real_physical_fixture_promotions_checked=7, mass_consistent_promotions_kept=accepted,
        inconsistent_or_unknown_mass_promotions_reverted=reverted, physical_mass_rule_independently_rebuilt=True,
        same_parent_and_features_for_both_separate_heads=True, trials=trials,
        actual_single_2070=True, actual_T4_source_or_production_evidence_replaced=False,
        quality_read=False, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 COMPLETE LOCAL PRODUCTION/INACTIVE-FALLBACK/CHANGED-PHYSICAL-PATH AUDIT PASSED', flush=True)


if __name__ == '__main__': main()
