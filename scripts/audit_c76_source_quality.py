"""Independently reconstruct all source ranks and the predeclared exact gates."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
import torch
import numpy as np
import pandas as pd
from fractions import Fraction
from functools import lru_cache
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    panel = ROOT / 'outputs/local/c72-clean-target-holdout-20261112'
    target = base / 'independent-source-quality-audit.json'; assert not target.exists()
    plan = read(base / 'source-quality-plan.json')
    result = read(base / 'source-results.json')
    consumed = read(panel / 'evaluation-consumption.json')
    assert consumed['state'] == 'complete' and consumed['source_results_sha256'] == sha(base / 'source-results.json')
    assert result['source_sha256'] == plan['evaluator_source_sha256'] == sha(ROOT / 'scripts/evaluate_c76_source.py')
    assert plan['auditor_source_sha256'] == sha(__file__)
    assert result['quality_plan_sha256'] == sha(base / 'source-quality-plan.json')
    policy_path = ROOT / 'configs/c76-source-evaluation-v1.json'; policy = read(policy_path)
    assert result['policy_sha256'] == plan['policy_sha256'] == sha(policy_path)
    frozen_path = base / 'joint-source-prediction-freeze.json'; frozen = read(frozen_path)
    assert result['joint_prediction_freeze_sha256'] == consumed['joint_prediction_freeze_sha256'] == sha(frozen_path)
    binding_path = base / 'complete-pipeline-binding.json'
    assert result['complete_pipeline_binding_sha256'] == plan['complete_pipeline_binding_sha256'] == sha(binding_path)
    for name, pin in read(binding_path)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    for name, pin in frozen['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    for name, pin in result['files_sha256'].items(): assert sha(base / name) == pin, name
    labels_path = panel / 'labels.csv'
    label_files = {name.replace('\\', '/'): pin for name, pin in read(panel / 'validation-freeze.json')['files_sha256'].items()}
    assert sha(labels_path) == label_files[labels_path.relative_to(ROOT).as_posix()]
    labels = pd.read_csv(labels_path, dtype=str).set_index('molecule_id')
    assert len(labels) == 512 and labels.index.is_unique and labels.group.nunique() == 471
    q = pd.read_parquet(ROOT / 'outputs/publish/c72-clean-target-holdout-20261112/validation-queries.parquet')
    actual_modes = q.groupby('molecule_id').ionization_mode.agg(lambda x: 'mixed' if len(set(x)) > 1 else x.iloc[0]).loc[labels.index]
    assert len(q) == 2215 and actual_modes.tolist() == labels.query_mode.tolist()
    assert actual_modes.value_counts().to_dict() == policy['actual_mode_counts'] == result['actual_mode_counts']
    from c74_complete_target import module
    identify = lru_cache(maxsize=50000)(module('c76_quality_audit_identity', 'c09e_features.py').structure)
    paths = {}
    for arm, entries in frozen['controls'].items():
        for filename, path in entries.items():
            name = {'P01.csv': 'P01', 'P16.csv': 'P16', 'physical-parent.csv': 'physical'}[filename]
            paths[arm + '_' + name] = path
    paths.update(frozen['candidates'])
    assert set(paths) == set(policy['required_controls'] + policy['additional_reported_controls'] + ['cpu', 'gpu'])
    rebuilt = {}; keys_by_name = {}
    with np.load(base / 'source-ranks.npz', allow_pickle=False) as stored:
        assert set(stored.files) == set(paths)
        for name, path in paths.items():
            table = pd.read_csv(ROOT / path, dtype=str, keep_default_na=False).set_index('molecule_id')
            assert table.index.is_unique and set(table.index) == set(labels.index)
            ranks = []; lists = []
            for mid in labels.index:
                keys = [identify(smiles)[0] for smiles in table.loc[mid, 'smiles'].split(';')]
                assert 1 <= len(keys) <= 25 and len(keys) == len(set(keys))
                rank = 0
                for position, key in enumerate(keys, 1):
                    if key == labels.loc[mid, 'key']:
                        rank = position; break
                ranks.append(rank); lists.append(keys)
            rebuilt[name] = np.array(ranks, np.int16); keys_by_name[name] = lists
            np.testing.assert_array_equal(rebuilt[name], stored[name])
    accounting = pd.read_csv(base / 'source-recovered-displaced-accounting.csv', dtype=str)
    assert len(accounting) == 2 * 6 * 512
    assert not accounting.duplicated(['candidate', 'control', 'molecule_id']).any()
    masks = {'full': np.ones(512, bool)}
    masks.update({mode: actual_modes.eq(mode).to_numpy() for mode in policy['actual_mode_counts']})
    def exact_mean(ranks, mask):
        return sum((Fraction(1, int(rank)) if rank else Fraction() for rank in ranks[mask]), Fraction()) / int(mask.sum())
    eligibility = {}
    for trial in ['cpu', 'gpu']:
        required = []
        for control in policy['required_controls'] + policy['additional_reported_controls']:
            entry = result['comparisons'][trial][control]
            records = {'full': entry['full'], **entry['actual_query_mode']}
            guards = []
            a, b = rebuilt[control], rebuilt[trial]
            for mode, mask in masks.items():
                old, new = exact_mean(a, mask), exact_mean(b, mask); delta = new-old
                record = records[mode]
                assert record['n'] == int(mask.sum())
                assert (record['numerator'], record['denominator']) == (delta.numerator, delta.denominator)
                assert record['strictly_positive'] == (delta > 0) and record['nonnegative'] == (delta >= 0)
                for name, expected in [('baseline_MRR', old), ('candidate_MRR', new), ('delta', delta)]:
                    assert abs(record[name] - float(expected)) < 1e-14
                assert record['baseline_absent'] == int((a[mask] == 0).sum())
                assert record['candidate_absent'] == int((b[mask] == 0).sum())
                assert record['recovered'] == int(((a == 0) & (b > 0) & mask).sum())
                assert record['displaced'] == int(((a > 0) & (b == 0) & mask).sum())
                guards.append(delta > 0 if mode == 'full' else delta >= 0)
            assert entry['strict_full_and_all_mode_guards'] == all(guards)
            if control in policy['required_controls']: required.append(all(guards))
            rows = accounting[accounting.candidate.eq(trial) & accounting.control.eq(control)].set_index('molecule_id').loc[labels.index]
            np.testing.assert_array_equal(rows.before_rank.astype(int), a)
            np.testing.assert_array_equal(rows.after_rank.astype(int), b)
            assert rows.query_mode.tolist() == actual_modes.tolist()
            assert rows.chemistry_group.tolist() == labels.group.tolist()
            for i, mid in enumerate(labels.index):
                assert (rows.loc[mid, 'truth_recovered'] == 'True') == bool(a[i] == 0 and b[i] > 0)
                assert (rows.loc[mid, 'truth_displaced'] == 'True') == bool(a[i] > 0 and b[i] == 0)
                assert (rows.loc[mid, 'candidate_membership_changed'] == 'True') == (set(keys_by_name[control][i]) != set(keys_by_name[trial][i]))
        eligibility[trial] = all(required)
    assert eligibility == result['individual_source_gates']
    assert all(eligibility.values()) == result['both_source_gates_passed']
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        source_results_sha256=sha(base / 'source-results.json'), queries=512, acquisitions=2215,
        every_official_identity_and_absent_truth_zero_independently_rebuilt=True,
        every_rank_and_exact_rational_gate_independently_rebuilt=True,
        all_actual_modes_and_recovered_displaced_accounting_exact=True,
        individual_source_gates=eligibility, both_source_gates_passed=all(eligibility.values()),
        actual_production_and_once_C13_still_required=True, C13_read=False,
        scoring_entries=0, qualified_for_scoring=False))
    print('C76 INDEPENDENT SOURCE QUALITY AUDIT COMPLETE', eligibility, flush=True)


if __name__ == '__main__': main()
