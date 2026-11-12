"""One joint C13 read for the unchanged R1 pair, after real prerequisite runs.

Inference preparation and its independent execution audit remain mandatory.
The durable global intent is created before the first label file is opened.
"""
import argparse
import json
import os
from fractions import Fraction
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import BINDING, CONFIRM, CONSUMPTION, FAMILY, POLICY, rooted
from c76_confirmation_inputs import LABELS, QUERY, verify_predictions
from submit_c76_pair import durable_write


def exact_point(before, after, indices):
    selected = list(indices); assert selected
    a = sum((Fraction(1, int(before[i])) if before[i] else Fraction() for i in selected), Fraction()) / len(selected)
    b = sum((Fraction(1, int(after[i])) if after[i] else Fraction() for i in selected), Fraction()) / len(selected)
    delta = b-a
    return dict(n=len(selected), baseline_MRR=float(a), candidate_MRR=float(b), delta=float(delta),
        numerator=delta.numerator, denominator=delta.denominator, nonnegative=delta >= 0,
        baseline_absent=sum(int(before[i] == 0) for i in selected),
        candidate_absent=sum(int(after[i] == 0) for i in selected),
        recovered=sum(int(before[i] == 0 and after[i] > 0) for i in selected),
        displaced=sum(int(before[i] > 0 and after[i] == 0) for i in selected))


def chemistry_groups(labels):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    molecules = [Chem.MolFromSmiles(s) for s in labels.smiles]; assert all(m is not None for m in molecules)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in molecules]
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in molecules]
    roots = list(range(len(labels)))
    def find(i):
        while roots[i] != i: roots[i] = roots[roots[i]]; i = roots[i]
        return i
    for i in range(len(labels)):
        for j in range(i):
            if labels.key.iloc[i] == labels.key.iloc[j] or labels.formula.iloc[i] == labels.formula.iloc[j] or \
                (scaffolds[i] and scaffolds[i] == scaffolds[j]) or DataStructs.TanimotoSimilarity(fps[i], fps[j]) >= .7:
                a, b = find(i), find(j); roots[max(a, b)] = min(a, b)
    names = {}
    for i in range(len(labels)): names.setdefault(find(i), []).append(labels.molecule_id.iloc[i])
    return [min(names[find(i)]) for i in range(len(labels))]


def run(consume):
    assert consume, 'Use --consume for the already-authorized single joint evaluation'
    plan, frozen, policy, paths, contract = verify_predictions()
    assert not (CONFIRM / 'confirmation-results.json').exists()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from c74_complete_target import module
    identify = lru_cache(maxsize=50000)(module('c76_c13_official_identity', 'c09e_features.py').structure)
    q = pd.read_parquet(QUERY)
    assert len(q) == q.molecule_id.nunique() == 120 and q.spectrum_id.is_unique
    assert q.ionization_mode.value_counts().to_dict() == policy['actual_mode_counts']
    ids = q.molecule_id.astype(str).tolist(); modes = q.ionization_mode.astype(str).tolist()
    lists = {}; forms = {}; smiles_map = {}; leader_changes = {}
    for name, path in paths.items():
        frame = pd.read_csv(rooted(path), dtype=str, keep_default_na=False)
        assert frame.molecule_id.is_unique and set(frame.molecule_id) == set(ids)
        frame = frame.set_index('molecule_id').loc[ids]
        lists[name] = []; forms[name] = []
        for text in frame.smiles:
            strings = text.split(';'); descriptors = [identify(s) for s in strings]
            keys = [x[0] for x in descriptors]
            assert 1 <= len(keys) <= 25 and len(keys) == len(set(keys))
            lists[name].append(keys); forms[name].append([x[2] for x in descriptors])
            smiles_map.update(zip(strings, keys))
    for trial in ['cpu', 'gpu']:
        leader_changes[trial] = 0
        for i in range(120):
            assert set(lists[trial][i]) == set(lists['c76_physical'][i])
            assert forms[trial][i] == forms['c76_physical'][i]
            leader_changes[trial] += int(lists[trial][i][0] != lists['c76_physical'][i][0])
    commitment = dict(started_utc=stamp(), state='started', family=FAMILY,
        source_sha256=sha(__file__), policy_sha256=sha(POLICY),
        inference_plan_sha256=sha(CONFIRM / 'inference-plan.json'),
        joint_prediction_freeze_sha256=sha(CONFIRM / 'joint-prediction-freeze.json'),
        labels_sha256=contract['labels_sha256'], scoring_entries=0)
    with CONSUMPTION.open('x', encoding='utf8') as stream:
        json.dump(commitment, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    # First access to query truth is AFTER the durable exclusive global intent.
    assert sha(LABELS) == contract['labels_sha256']
    labels = pd.read_csv(LABELS, dtype=str)
    assert len(labels) == labels.molecule_id.nunique() == 120 and labels.key.is_unique
    labels = labels.set_index('molecule_id').loc[ids].reset_index()
    assert labels['mode'].tolist() == modes
    assert [identify(s)[0] for s in labels.smiles] == labels.key.tolist()
    assert [identify(s)[2] for s in labels.smiles] == labels.formula.tolist()
    fit = read(ROOT / 'outputs/local/c71-final-clean-direct-pair-20261112/complete-candidate-freeze.json')
    assert fit['global_C13_reserved_chemistry_fit_overlap'] == 0
    reserved = pd.read_parquet(ROOT / 'outputs/local/c68-reserved-fit-boundary-20261112/global-exclusion-chemistry.parquet')
    assert len(reserved) == reserved.key.nunique() == 231 and set(labels.key) <= set(reserved.key)
    groups = chemistry_groups(labels)
    pd.DataFrame(dict(molecule_id=ids, query_mode=modes, chemistry_group=groups)).to_csv(CONFIRM / 'chemistry-groups.csv', index=False)
    ranks = {name: np.array([next((r for r, key in enumerate(keys, 1) if key == labels.key.iloc[i]), 0)
        for i, keys in enumerate(rows)], np.int16) for name, rows in lists.items()}
    np.savez_compressed(CONFIRM / 'confirmation-ranks.npz', **ranks)
    pd.DataFrame(smiles_map.items(), columns=['smiles', 'key']).to_parquet(CONFIRM / 'quality-identity-map.parquet', index=False)
    codes, unique = pd.factorize(groups, sort=True); counts = np.bincount(codes)
    draws = np.random.default_rng(policy['bootstrap_seed']).integers(0, len(unique),
        size=(policy['bootstrap_repetitions'], len(unique)))
    comparisons = {}; eligibility = {}; accounting = []
    for trial in ['cpu', 'gpu']:
        comparisons[trial] = {}; passed = []
        for control in policy['required_controls'] + policy['additional_reported_controls']:
            a, b = ranks[control], ranks[trial]
            full = exact_point(a, b, range(120))
            mode_results = {name: exact_point(a, b, [i for i, value in enumerate(modes) if value == name]) for name in policy['actual_mode_counts']}
            delta = np.divide(1., b, out=np.zeros(120), where=b > 0) - np.divide(1., a, out=np.zeros(120), where=a > 0)
            sums = np.bincount(codes, weights=delta)
            boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            passed_here = full['nonnegative'] and all(r['nonnegative'] for r in mode_results.values())
            comparisons[trial][control] = dict(full=full, actual_query_mode=mode_results,
                full_and_all_mode_guards=passed_here,
                paired_chemistry_cluster_bootstrap_95percent_interval=np.quantile(boot, [.025, .975]).tolist())
            if control in policy['required_controls']: passed.append(passed_here)
            for i, mid in enumerate(ids):
                accounting.append(dict(candidate=trial, control=control, molecule_id=mid, query_mode=modes[i], chemistry_group=groups[i],
                    before_rank=int(a[i]), after_rank=int(b[i]), reciprocal_rank_delta=float(delta[i]),
                    truth_recovered=bool(a[i] == 0 and b[i] > 0), truth_displaced=bool(a[i] > 0 and b[i] == 0),
                    rank_improved=bool(b[i] > 0 and (a[i] == 0 or b[i] < a[i])),
                    rank_regressed=bool(a[i] > 0 and (b[i] == 0 or b[i] > a[i])),
                    rank1_changed=lists[control][i][0] != lists[trial][i][0],
                    candidate_membership_changed=set(lists[control][i]) != set(lists[trial][i])))
        eligibility[trial] = all(passed)
    pd.DataFrame(accounting).to_csv(CONFIRM / 'recovered-displaced-accounting.csv', index=False)
    files = ['confirmation-ranks.npz', 'quality-identity-map.parquet', 'chemistry-groups.csv', 'recovered-displaced-accounting.csv']
    result = dict(completed_utc=stamp(), family=FAMILY, source_sha256=sha(__file__), policy_sha256=sha(POLICY),
        complete_pipeline_binding_sha256=sha(BINDING), inference_plan_sha256=sha(CONFIRM / 'inference-plan.json'),
        joint_prediction_freeze_sha256=sha(CONFIRM / 'joint-prediction-freeze.json'), labels_sha256=sha(LABELS),
        queries=120, acquisitions=120, actual_mode_counts=policy['actual_mode_counts'], chemistry_groups=len(unique),
        comparisons=comparisons, individual_confirmation_gates=eligibility, both_confirmation_gates_passed=all(eligibility.values()),
        head_rank1_changes=leader_changes, head_membership_and_formula_slots_preserved=True,
        all_absent_truths_zero=True, all_original_queries_retained=True, no_recipe_or_cohort_selection=True,
        fitting_global231_chemistry_overlap=0, entire_reserved_inventory_checked=231,
        public_parent_pretraining_exposure_not_excluded=True, universal_pretraining_separation_claimed=False,
        files_sha256={name: sha(CONFIRM / name) for name in files}, limitation=policy['limitation'],
        scoring_entries=0, qualified_for_scoring=False)
    write(CONFIRM / 'confirmation-results.json', result)
    commitment.update(state='complete', completed_utc=stamp(), results_sha256=sha(CONFIRM / 'confirmation-results.json'))
    durable_write(CONSUMPTION, commitment)
    print('C76 ONCE-ONLY JOINT RESERVED CONFIRMATION COMPLETE', eligibility, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--consume', action='store_true')
    args = parser.parse_args(); run(args.consume)
