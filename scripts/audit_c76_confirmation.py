"""Independent all-query ranks, exact gates, groups and accounting audit.

This audits the same completed joint consumption, without any new selection,
fitting, resampling decision or confirmation evaluation.
"""
import os
from fractions import Fraction
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import BINDING, CONFIRM, CONSUMPTION, FAMILY, POLICY, rooted
from c76_confirmation_inputs import LABELS, QUERY, verify_predictions


def component_names(labels):
    # An adjacency traversal independently checks the evaluator's union-find.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    molecules = [Chem.MolFromSmiles(s) for s in labels.smiles]; assert all(m is not None for m in molecules)
    fps = [gen.GetFingerprint(m) for m in molecules]
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in molecules]
    graph = [set() for _ in range(len(labels))]
    for i in range(len(labels)):
        for j in range(i):
            if labels.key.iloc[i] == labels.key.iloc[j] or labels.formula.iloc[i] == labels.formula.iloc[j] or \
                (scaffolds[i] and scaffolds[i] == scaffolds[j]) or DataStructs.TanimotoSimilarity(fps[i], fps[j]) >= .7:
                graph[i].add(j); graph[j].add(i)
    names = [None] * len(labels)
    for start in range(len(labels)):
        if names[start] is not None: continue
        todo = [start]; members = set()
        while todo:
            node = todo.pop()
            if node in members: continue
            members.add(node); todo.extend(graph[node] - members)
        name = min(labels.molecule_id.iloc[i] for i in members)
        for node in members: names[node] = name
    return names


def main():
    plan, frozen, policy, paths, contract = verify_predictions(unconsumed=False)
    target = CONFIRM / 'independent-confirmation-audit.json'; assert not target.exists()
    result_path = CONFIRM / 'confirmation-results.json'; result = read(result_path); consumed = read(CONSUMPTION)
    assert result['family'] == consumed['family'] == FAMILY and consumed['state'] == 'complete'
    assert consumed['results_sha256'] == sha(result_path)
    assert consumed['source_sha256'] == result['source_sha256'] == plan['evaluator_source_sha256']
    assert result['source_sha256'] == sha(ROOT / 'scripts/evaluate_c76_confirmation.py')
    assert plan['auditor_source_sha256'] == sha(__file__)
    assert consumed['policy_sha256'] == result['policy_sha256'] == sha(POLICY)
    assert consumed['joint_prediction_freeze_sha256'] == result['joint_prediction_freeze_sha256'] == sha(CONFIRM / 'joint-prediction-freeze.json')
    assert consumed['inference_plan_sha256'] == result['inference_plan_sha256'] == sha(CONFIRM / 'inference-plan.json')
    assert result['complete_pipeline_binding_sha256'] == sha(BINDING)
    for name, pin in result['files_sha256'].items(): assert sha(CONFIRM / name) == pin, name
    assert consumed['labels_sha256'] == result['labels_sha256'] == contract['labels_sha256'] == sha(LABELS)
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from c74_complete_target import module
    identify = lru_cache(maxsize=50000)(module('c76_c13_audit_identity', 'c09e_features.py').structure)
    q = pd.read_parquet(QUERY); ids = q.molecule_id.tolist()
    assert len(q) == q.molecule_id.nunique() == q.spectrum_id.nunique() == 120
    labels = pd.read_csv(LABELS, dtype=str).set_index('molecule_id').loc[ids].reset_index()
    assert len(labels) == labels.key.nunique() == 120
    assert [identify(s)[0] for s in labels.smiles] == labels.key.tolist()
    assert [identify(s)[2] for s in labels.smiles] == labels.formula.tolist()
    modes = q.ionization_mode.tolist(); assert labels['mode'].tolist() == modes
    assert q.ionization_mode.value_counts().to_dict() == policy['actual_mode_counts'] == result['actual_mode_counts']
    groups = component_names(labels)
    stored_groups = pd.read_csv(CONFIRM / 'chemistry-groups.csv', dtype=str, keep_default_na=False)
    assert stored_groups.molecule_id.tolist() == ids and stored_groups.query_mode.tolist() == modes
    assert stored_groups.chemistry_group.tolist() == groups and result['chemistry_groups'] == len(set(groups))
    rebuilt = {}; ordered = {}; formulas = {}; identity_map = {}
    with np.load(CONFIRM / 'confirmation-ranks.npz', allow_pickle=False) as stored:
        assert set(stored.files) == set(paths)
        for name, path in paths.items():
            frame = pd.read_csv(rooted(path), dtype=str, keep_default_na=False)
            assert frame.molecule_id.is_unique and set(frame.molecule_id) == set(ids)
            ordered[name] = []; formulas[name] = []; ranks = []
            for mid, truth in zip(ids, labels.key):
                text = frame.set_index('molecule_id').loc[mid, 'smiles']; strings = text.split(';')
                descriptors = [identify(s) for s in strings]; keys = [d[0] for d in descriptors]
                assert 1 <= len(keys) <= 25 and len(keys) == len(set(keys))
                ordered[name].append(keys); formulas[name].append([d[2] for d in descriptors])
                ranks.append(keys.index(truth) + 1 if truth in keys else 0)
                identity_map.update(zip(strings, keys))
            rebuilt[name] = np.asarray(ranks, np.int16)
            assert stored[name].shape == (120,); np.testing.assert_array_equal(rebuilt[name], stored[name])
    saved_map = pd.read_parquet(CONFIRM / 'quality-identity-map.parquet')
    assert saved_map.smiles.is_unique and dict(zip(saved_map.smiles, saved_map.key)) == identity_map
    accounting = pd.read_csv(CONFIRM / 'recovered-displaced-accounting.csv', dtype=str, keep_default_na=False)
    assert len(accounting) == 2 * 6 * 120 and not accounting.duplicated(['candidate', 'control', 'molecule_id']).any()
    codes, unique = pd.factorize(groups, sort=True); counts = np.bincount(codes)
    draws = np.random.default_rng(policy['bootstrap_seed']).integers(0, len(unique),
        size=(policy['bootstrap_repetitions'], len(unique)))
    eligibility = {}
    for trial in ['cpu', 'gpu']:
        passed = []
        assert all(set(a) == set(b) for a, b in zip(ordered[trial], ordered['c76_physical']))
        assert formulas[trial] == formulas['c76_physical']
        assert result['head_rank1_changes'][trial] == sum(a[0] != b[0] for a, b in zip(ordered[trial], ordered['c76_physical']))
        for control in policy['required_controls'] + policy['additional_reported_controls']:
            a, b = rebuilt[control], rebuilt[trial]; entry = result['comparisons'][trial][control]
            records = {'full': entry['full'], **entry['actual_query_mode']}; checks = []
            masks = {'full': np.ones(120, bool)}
            masks.update({mode: q.ionization_mode.eq(mode).to_numpy() for mode in policy['actual_mode_counts']})
            for mode, mask in masks.items():
                old = sum((Fraction(1, int(rank)) for rank in a[mask] if rank), Fraction()) / int(mask.sum())
                new = sum((Fraction(1, int(rank)) for rank in b[mask] if rank), Fraction()) / int(mask.sum())
                delta = new-old; r = records[mode]
                assert r['n'] == int(mask.sum()) and (r['numerator'], r['denominator']) == (delta.numerator, delta.denominator)
                assert r['nonnegative'] == (delta >= 0); checks.append(delta >= 0)
                for field, expected in [('baseline_MRR', old), ('candidate_MRR', new), ('delta', delta)]:
                    assert abs(r[field] - float(expected)) < 1e-14
                assert r['baseline_absent'] == int((a[mask] == 0).sum()) and r['candidate_absent'] == int((b[mask] == 0).sum())
                assert r['recovered'] == int(((a == 0) & (b > 0) & mask).sum())
                assert r['displaced'] == int(((a > 0) & (b == 0) & mask).sum())
            assert entry['full_and_all_mode_guards'] == all(checks)
            if control in policy['required_controls']: passed.append(all(checks))
            rr_delta = np.divide(1., b, out=np.zeros(120), where=b > 0) - np.divide(1., a, out=np.zeros(120), where=a > 0)
            sums = np.bincount(codes, weights=rr_delta)
            interval = np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [.025, .975])
            np.testing.assert_allclose(entry['paired_chemistry_cluster_bootstrap_95percent_interval'], interval, atol=1e-14, rtol=0)
            rows = accounting[accounting.candidate.eq(trial) & accounting.control.eq(control)].set_index('molecule_id').loc[ids]
            np.testing.assert_array_equal(rows.before_rank.astype(int), a); np.testing.assert_array_equal(rows.after_rank.astype(int), b)
            np.testing.assert_allclose(rows.reciprocal_rank_delta.astype(float), rr_delta, atol=1e-14, rtol=0)
            assert rows.query_mode.tolist() == modes and rows.chemistry_group.tolist() == groups
            for i, mid in enumerate(ids):
                expected = dict(truth_recovered=bool(a[i] == 0 and b[i] > 0), truth_displaced=bool(a[i] > 0 and b[i] == 0),
                    rank_improved=bool(b[i] > 0 and (a[i] == 0 or b[i] < a[i])),
                    rank_regressed=bool(a[i] > 0 and (b[i] == 0 or b[i] > a[i])),
                    rank1_changed=ordered[control][i][0] != ordered[trial][i][0],
                    candidate_membership_changed=set(ordered[control][i]) != set(ordered[trial][i]))
                for field, value in expected.items(): assert rows.loc[mid, field] == str(value), (trial, control, mid, field)
        eligibility[trial] = all(passed)
    assert eligibility == result['individual_confirmation_gates']
    assert all(eligibility.values()) == result['both_confirmation_gates_passed']
    reserved = pd.read_parquet(ROOT / 'outputs/local/c68-reserved-fit-boundary-20261112/global-exclusion-chemistry.parquet')
    assert len(reserved) == reserved.key.nunique() == 231 and set(labels.key) <= set(reserved.key)
    assert read(ROOT / 'outputs/local/c71-final-clean-direct-pair-20261112/complete-candidate-freeze.json')['global_C13_reserved_chemistry_fit_overlap'] == 0
    assert result['queries'] == result['acquisitions'] == 120
    assert result['all_original_queries_retained'] and result['all_absent_truths_zero'] and result['no_recipe_or_cohort_selection']
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        confirmation_results_sha256=sha(result_path), inference_plan_sha256=sha(CONFIRM / 'inference-plan.json'),
        queries=120, acquisitions=120, individual_confirmation_gates=eligibility,
        both_confirmation_gates_passed=all(eligibility.values()),
        every_rank_and_exact_gate_independently_rebuilt=True, all_query_accounting_independently_rebuilt=True,
        chemistry_components_and_descriptive_bootstrap_rebuilt=True,
        fitted_pair_entire_reserved231_quarantine_preserved=True,
        universal_pretraining_separation_claimed=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 INDEPENDENT RESERVED CONFIRMATION AUDIT COMPLETE', eligibility, flush=True)


if __name__ == '__main__': main()
