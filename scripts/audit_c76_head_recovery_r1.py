"""Audit the composed R1 preview without falsely passing the failed R0 run."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
import torch
import importlib.util
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp
from audit_c76_physical_fixture import audit as audit_physical


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'; out = base / 'interface-repair-r1'
    target = out / 'independent-run-audit.json'; assert not target.exists()
    prefix_path = out / 'completed-prefix-audit.json'; prefix = read(prefix_path)
    recovery_path = out / 'head-recovery-results.json'; recovery = read(recovery_path)
    intent = read(out / 'head-recovery-intent.json'); prep = read(out / 'candidate-preparation.json')
    assert recovery['passed'] and recovery['source_sha256'] == sha(ROOT / 'scripts/recover_c76_production_heads_r1.py')
    assert recovery['recovery_intent_sha256'] == sha(out / 'head-recovery-intent.json')
    assert intent['completed_prefix_audit_sha256'] == sha(prefix_path)
    assert intent['candidate_preparation_sha256'] == sha(out / 'candidate-preparation.json')
    assert prefix['completed_prefix_passed'] and not prefix['original_full_run_passed']
    assert not recovery['GPU_core_restarted'] and recovery['actual_new_head_stages_executed']
    assert recovery['original_all_active_12query_51acquisition_vectors_scores_lists_bitexact']
    assert recovery['corrupted_active_row_rejected']
    working = Path(recovery['working']); assert str(working) == prefix['working']
    for name, digest in prefix['reusable_unchanged_files_sha256'].items(): assert sha(working / name) == digest, name
    bundle = ROOT / prep['bundle']; manifest = read(bundle / 'pair-manifest.json')
    assert sha(bundle / 'pair-manifest.json') == prep['pair_manifest_sha256'] == recovery['pair_manifest_sha256']
    for name, digest in manifest['files_sha256'].items(): assert sha(bundle / name) == digest, name
    before = (out / 'previous-sources/scripts/score_c76_complete_pair.py').read_text(encoding='utf8')
    after = (ROOT / 'scripts/score_c76_complete_pair.py').read_text(encoding='utf8')
    addition = "    # Production previews also retain placeholder evidence for unprocessed IDs.\n    # Infer from every ID with spectra; preserve all other answer rows below.\n    raw = raw[raw.molecule_id.isin(active)].reset_index(drop=True)\n    public = public[public.molecule_id.isin(active)].reset_index(drop=True)\n    ours = ours[ours.molecule_id.isin(active)].reset_index(drop=True)\n"
    assert after.count(addition) == 1 and after.replace(addition, '') == before
    p01 = pd.read_csv(working / 'c09e_p01_submission.csv', dtype=str, keep_default_na=False)
    sample = pd.read_csv(base / 'production-inputs/sample_submission.csv', dtype=str, keep_default_na=False)
    queries = pd.read_parquet(working / 'effective_query_spectra.parquet')
    assert len(queries) == 41 and queries.molecule_id.nunique() == 12
    assert p01.molecule_id.tolist() == sample.molecule_id.tolist() and len(p01) == 400
    inactive = ~p01.molecule_id.isin(set(queries.molecule_id)); assert inactive.sum() == 388
    spec = importlib.util.spec_from_file_location('c76_r1_audit_identity', bundle / 'identity_features.py')
    features = importlib.util.module_from_spec(spec); spec.loader.exec_module(features)
    identity = lru_cache(maxsize=50000)(features.structure)
    parents = []; candidate_tables = []; vectors = []; trials = {}
    for trial in ['cpu', 'gpu']:
        folder = working / ('c76_' + trial + '_scores_r1')
        receipt = read(working / ('c76_' + trial + '_execution_r1.json'))
        assert receipt['source_sha256'] == sha(bundle / 'score_c76_complete_pair.py')
        assert receipt['pair_manifest_sha256'] == sha(bundle / 'pair-manifest.json')
        assert receipt['queries'] == 12 and receipt['acquisitions'] == 41 and receipt['valid_rows'] == 400
        assert receipt['exact_unprocessed_fallback_rows'] == 388 and not receipt['is_hidden_rerun']
        assert not receipt['training_performed'] and not receipt['quality_labels_read']
        for name, digest in receipt['files_sha256'].items(): assert sha(folder / name) == digest, name
        parent = pd.read_csv(folder / 'physical-parent.csv', dtype=str, keep_default_na=False)
        final = pd.read_csv(working / ('c76_' + trial + '_submission_r1.csv'), dtype=str, keep_default_na=False)
        assert final.molecule_id.tolist() == sample.molecule_id.tolist() and final.molecule_id.is_unique
        pd.testing.assert_frame_equal(parent.loc[inactive], p01.loc[inactive], check_exact=True)
        pd.testing.assert_frame_equal(final.loc[inactive], p01.loc[inactive], check_exact=True)
        changed = leaders = 0
        for a, b in zip(parent.smiles, final.smiles):
            old, new = a.split(';'), b.split(';')
            assert 1 <= len(old) == len(new) <= 25 and set(old) == set(new)
            assert [identity(s)[2] for s in old] == [identity(s)[2] for s in new]
            keys = [identity(s)[0] for s in new]; assert len(keys) == len(set(keys))
            changed += old != new; leaders += identity(old[0])[0] != identity(new[0])[0]
        assert changed == receipt['trials'][trial]['changed_lists'] > 0
        assert leaders == receipt['trials'][trial]['rank1_changes']
        assert receipt['trials'][trial]['independent_scores_and_lists_exact']
        parents.append(parent); candidate_tables.append(pd.read_parquet(folder / 'candidates.parquet')); vectors.append(np.load(folder / 'features.npy'))
        trials[trial] = dict(changed_lists=changed, rank1_changes=leaders, exact_inactive_rows=388,
            independent_scores_and_lists_exact=True, candidate_membership_and_formula_slots_preserved=True,
            CSV_sha256=sha(working / ('c76_' + trial + '_submission_r1.csv')))
    pd.testing.assert_frame_equal(parents[0], parents[1], check_exact=True)
    pd.testing.assert_frame_equal(candidate_tables[0], candidate_tables[1], check_exact=True)
    np.testing.assert_array_equal(vectors[0], vectors[1])
    assert set(candidate_tables[0].molecule_id) == set(queries.molecule_id)
    physical = audit_physical(working, bundle, ROOT / 'outputs/research/2026-11-09-silver/extlib', out)
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        original_r0_full_run_passed=False, composed_r1_local_production_passed=True,
        completed_prefix_audit_sha256=sha(prefix_path), head_recovery_results_sha256=sha(recovery_path),
        pair_manifest_sha256=sha(bundle / 'pair-manifest.json'), original_GPU_core_restarted=False,
        preview_queries=12, preview_acquisitions=41, output_rows=400, inactive_rows_preserved_exact=388,
        every_actual_processed_query_and_acquisition_retained=True,
        unchanged_all_active_12query_51acquisition_math_bitexact=True, corrupted_active_row_still_rejected=True,
        same_parent_and_features_for_both_heads=True, trials=trials,
        real_physical_fixture_promotions_checked=7, physical_fixture=physical,
        actual_T4_source_or_production_evidence_replaced=False,
        quality_read=False, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 R1 INDEPENDENT COMPLETE PREVIEW/388FALLBACK/CHANGEDPHYSICAL AUDIT PASSED', flush=True)


if __name__ == '__main__': main()
