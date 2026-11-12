"""Prepare, then perform exactly one frozen joint source-quality evaluation."""
import argparse
from fractions import Fraction
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
PANEL = ROOT / 'outputs/local/c72-clean-target-holdout-20261112'
POLICY = ROOT / 'configs/c76-source-evaluation-v1.json'
PLAN = BASE / 'source-quality-plan-r1.json'


def prepare():
    assert not PLAN.exists() and not (PANEL / 'evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert not (BASE / 'joint-source-prediction-freeze.json').exists()
    binding = BASE / 'complete-pipeline-binding-r1.json'
    for name, pin in read(binding)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    write(PLAN, dict(frozen_utc=stamp(), evaluator_source_sha256=sha(__file__),
        auditor_source_sha256=sha(ROOT / 'scripts/audit_c76_source_quality.py'),
        policy_sha256=sha(POLICY), complete_pipeline_binding_sha256=sha(binding),
        original_C72_preparation_protocol_sha256=sha(PANEL / 'preparation-protocol.json'),
        controls_addendum_sha256=sha(ROOT / 'configs/c76-source-controls-addendum-v1.json'),
        quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 JOINT QUALITY CODE AND ORIGINAL GATES FROZEN BEFORE SOURCE PREDICTIONS', flush=True)


def point(before, after, mask, np):
    n = int(mask.sum()); assert n
    def exact(ranks):
        counts = np.bincount(ranks[mask], minlength=26)
        return sum((Fraction(int(counts[r]), r) for r in range(1, 26)), Fraction()) / n
    a, b = exact(before), exact(after); delta = b-a
    return dict(n=n, baseline_MRR=float(a), candidate_MRR=float(b), delta=float(delta),
        numerator=delta.numerator, denominator=delta.denominator,
        strictly_positive=delta > 0, nonnegative=delta >= 0,
        baseline_absent=int((before[mask] == 0).sum()), candidate_absent=int((after[mask] == 0).sum()),
        recovered=int(((before == 0) & (after > 0) & mask).sum()),
        displaced=int(((before > 0) & (after == 0) & mask).sum()))


def run():
    import os
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from c74_complete_target import module
    from complete_c76_target import verify
    verify()
    plan = read(PLAN); policy = read(POLICY)
    assert plan['evaluator_source_sha256'] == sha(__file__)
    assert plan['auditor_source_sha256'] == sha(ROOT / 'scripts/audit_c76_source_quality.py')
    assert plan['policy_sha256'] == sha(POLICY)
    frozen_path = BASE / 'joint-source-prediction-freeze.json'; frozen = read(frozen_path)
    assert frozen['complete_pipeline_binding_sha256'] == plan['complete_pipeline_binding_sha256']
    assert frozen['all_six_controls_and_both_heads_frozen_before_quality']
    assert frozen['quality_protocol_sha256'] == plan['original_C72_preparation_protocol_sha256'] == sha(PANEL / 'preparation-protocol.json')
    assert frozen['controls_addendum_sha256'] == plan['controls_addendum_sha256']
    for name, pin in frozen['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    labels_path = PANEL / 'labels.csv'
    panel_files = {name.replace('\\', '/'): pin for name, pin in read(PANEL / 'validation-freeze.json')['files_sha256'].items()}
    assert sha(labels_path) == panel_files[labels_path.relative_to(ROOT).as_posix()]
    consumption = PANEL / 'evaluation-consumption.json'
    assert not (BASE / 'source-results.json').exists()
    # Exclusive, durable intent is written BEFORE any prediction is matched to
    # query truth. On failure retain it and reconcile; never silently reread.
    with consumption.open('x', encoding='utf8') as stream:
        import json
        json.dump(dict(started_utc=stamp(), state='started', family='C76 fixed complete CPU/GPU pair',
            source_sha256=sha(__file__), quality_plan_sha256=sha(PLAN),
            joint_prediction_freeze_sha256=sha(frozen_path), C13_read=False, scoring_entries=0), stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    labels = pd.read_csv(labels_path, dtype=str)
    assert len(labels) == labels.molecule_id.nunique() == 512 and labels.group.nunique() == 471
    q = pd.read_parquet(ROOT / 'outputs/publish/c72-clean-target-holdout-20261112/validation-queries.parquet')
    assert len(q) == 2215
    modes = q.groupby('molecule_id').ionization_mode.agg(lambda a: 'mixed' if a.nunique() > 1 else a.iloc[0])
    assert labels.molecule_id.map(modes).tolist() == labels.query_mode.tolist()
    assert labels.query_mode.value_counts().to_dict() == policy['actual_mode_counts']
    identity = lru_cache(maxsize=50000)(module('c76_quality_official_identity', 'c09e_features.py').structure)
    paths = {arm + '_' + {'P01.csv': 'P01', 'P16.csv': 'P16', 'physical-parent.csv': 'physical'}[name]: path
        for arm, mapping in frozen['controls'].items() for name, path in mapping.items()}
    paths.update(frozen['candidates'])
    assert set(paths) == set(policy['required_controls'] + policy['additional_reported_controls'] + ['cpu', 'gpu'])
    ranks = {}; identity_records = {}; ordered_lists = {}
    for name, path in paths.items():
        frame = pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)
        assert frame.molecule_id.is_unique and set(frame.molecule_id) == set(labels.molecule_id)
        frame = frame.set_index('molecule_id').loc[labels.molecule_id]
        ranks[name] = np.zeros(512, np.int16); ordered_lists[name] = []
        for i, text in enumerate(frame.smiles):
            strings = text.split(';'); keys = [identity(s)[0] for s in strings]
            assert 1 <= len(keys) <= 25 and len(set(keys)) == len(keys)
            identity_records.update(zip(strings, keys)); ordered_lists[name].append(keys)
            ranks[name][i] = {key: rank for rank, key in enumerate(keys, 1)}.get(labels.key.iloc[i], 0)
    np.savez_compressed(BASE / 'source-ranks.npz', **ranks)
    pd.DataFrame(identity_records.items(), columns=['smiles', 'key']).to_parquet(BASE / 'source-quality-identity-map.parquet', index=False)
    comparisons = {}; eligibility = {}; accounting = []
    all_mask = np.ones(512, bool)
    group_codes, unique_groups = pd.factorize(labels.group, sort=True)
    counts = np.bincount(group_codes)
    generator = np.random.default_rng(policy['bootstrap_seed'])
    draws = generator.integers(0, len(unique_groups), size=(policy['bootstrap_repetitions'], len(unique_groups)))
    for trial in ['cpu', 'gpu']:
        comparisons[trial] = {}; passes = []
        for control in policy['required_controls'] + policy['additional_reported_controls']:
            before, after = ranks[control], ranks[trial]
            full = point(before, after, all_mask, np)
            mode = {name: point(before, after, labels.query_mode.eq(name).to_numpy(), np) for name in policy['actual_mode_counts']}
            a = np.divide(1., before, out=np.zeros(512), where=before > 0)
            b = np.divide(1., after, out=np.zeros(512), where=after > 0)
            sums = np.bincount(group_codes, weights=b-a)
            boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            interval = np.quantile(boot, [.025, .975]).tolist()
            comparisons[trial][control] = dict(full=full, actual_query_mode=mode,
                paired_chemistry_cluster_bootstrap_95percent_interval=interval,
                strict_full_and_all_mode_guards=full['strictly_positive'] and all(x['nonnegative'] for x in mode.values()))
            if control in policy['required_controls']:
                passes.append(comparisons[trial][control]['strict_full_and_all_mode_guards'])
            for i in range(512):
                accounting.append(dict(candidate=trial, control=control, molecule_id=labels.molecule_id.iloc[i],
                    source=labels.source.iloc[i], query_mode=labels.query_mode.iloc[i], chemistry_group=labels.group.iloc[i],
                    before_rank=int(before[i]), after_rank=int(after[i]), reciprocal_rank_delta=float(b[i]-a[i]),
                    truth_recovered=before[i] == 0 and after[i] > 0,
                    truth_displaced=before[i] > 0 and after[i] == 0,
                    candidate_membership_changed=set(ordered_lists[control][i]) != set(ordered_lists[trial][i])))
        eligibility[trial] = all(passes)
    pd.DataFrame(accounting).to_csv(BASE / 'source-recovered-displaced-accounting.csv', index=False)
    result = dict(completed_utc=stamp(), source_sha256=sha(__file__), quality_plan_sha256=sha(PLAN),
        policy_sha256=sha(POLICY), complete_pipeline_binding_sha256=plan['complete_pipeline_binding_sha256'],
        joint_prediction_freeze_sha256=sha(frozen_path), queries=512, acquisitions=2215, chemistry_groups=471,
        actual_mode_counts=policy['actual_mode_counts'], required_controls=policy['required_controls'],
        comparisons=comparisons, individual_source_gates=eligibility, both_source_gates_passed=all(eligibility.values()),
        all_absent_truths_zero=True, all_original_queries_and_actual_modes_retained=True,
        files_sha256={name: sha(BASE / name) for name in ['source-ranks.npz', 'source-quality-identity-map.parquet', 'source-recovered-displaced-accounting.csv']},
        independent_quality_audit_still_required=True, actual_production_and_once_C13_still_required=True,
        C72_now_USED=True, no_cohort_or_recipe_rescue=True, C13_read=False, scoring_entries=0,
        qualified_for_scoring=False,
        limitation='Controlled in-corpus target panel; historical public parent/encoder exposure persists. No independent-laboratory or public-score calibration claim.')
    write(BASE / 'source-results.json', result)
    consumed = read(consumption); consumed.update(completed_utc=stamp(), state='complete',
        source_results_sha256=sha(BASE / 'source-results.json'), both_source_gates_passed=result['both_source_gates_passed'])
    write(consumption, consumed)
    print('C76 ONE-TIME JOINT SOURCE QUALITY COMPLETE', eligibility, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['prepare', 'run']); args = parser.parse_args()
    (prepare if args.phase == 'prepare' else run)()
