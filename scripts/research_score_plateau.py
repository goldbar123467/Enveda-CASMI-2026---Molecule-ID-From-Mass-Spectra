"""CPU-only postmortem on already-consumed development artifacts, never train/submit.

Oracle ranks use known development answers to describe a mathematical limit.
They are not a deployable recipe or an estimate of hidden leaderboard gains.
No fresh/reserved label file is opened.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/local/c21-negative-tail-20261110'
TRIALS = ROOT / 'outputs/local/two-past-trials-20261110'
OUT = ROOT / 'outputs/local/plateau-research-20261111'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def best_rrf_rank(n, truth_rank):
    """Exact best rank over every possible neural order under fixed .2/offset3.

For a tail truth, giving it neural rank1 dominates every alternative. Giving
the protected first candidate neural rank2 consumes the largest remaining
bonus without harming the tail. Matching other candidates to ranks3..n
maximizes how many can score below the truth; greedy threshold matching is
optimal because each candidate's acceptable ranks form a suffix.
"""
    if truth_rank <= 1:
        return truth_rank
    target = 1 / (3 + truth_rank) + .2 / 4
    minimum_ranks = []
    for baseline_rank in range(2, n + 1):
        if baseline_rank == truth_rank:
            continue
        feasible = []
        for neural_rank in range(3, n + 1):
            value = 1 / (3 + baseline_rank) + .2 / (3 + neural_rank)
            if value < target or (value == target and baseline_rank > truth_rank):
                feasible.append(neural_rank)
        minimum_ranks.append(min(feasible, default=n + 1))
    available = list(range(3, n + 1))
    below = 0
    for minimum in sorted(minimum_ranks):
        compatible = next((r for r in available if r >= minimum), None)
        if compatible is not None:
            available.remove(compatible)
            below += 1
    return 2 + n - 2 - below


def verify_oracle():
    # An independent exhaustive calculation checks the matching derivation.
    cases = 0
    for n in range(2, 8):
        minima = {rank: n for rank in range(2, n + 1)}
        for permutation in itertools.permutations(range(1, n + 1)):
            utility = [1 / (4 + i) + .2 / (3 + permutation[i]) for i in range(n)]
            order = [0] + sorted(range(1, n), key=lambda i: (-utility[i], i))
            for rank in minima:
                minima[rank] = min(minima[rank], order.index(rank - 1) + 1)
        for rank, minimum in minima.items():
            assert best_rrf_rank(n, rank) == minimum, (n, rank, minimum)
            cases += 1
    return cases


def main():
    pa.set_cpu_count(2)
    assert (ROOT / '.local/compute-paused.json').is_file()
    assert not (OUT / 'used240-postmortem.json').exists(), 'Postmortem already sealed'
    # C12 consumed all160 positive labels before the C21 full240 evaluation.
    prior = json.loads((ROOT / 'outputs/local/c12-deep-candidates-20261110/confirmation.json').read_text())
    assert prior['queries'] == 120
    assert json.loads((BASE / 'development.json').read_text())['queries'] == 240
    selection = json.loads((TRIALS / 'selection.json').read_text())
    files = [BASE / n for n in ['parent.csv', 'queries.parquet', 'fingerprint-smiles.parquet']]
    files += [ROOT / n for n in ['outputs/local/c10-casmi2022-20261110/labels.csv',
        'outputs/local/c20-negative-development-20261110/labels.csv']]
    files += [TRIALS / (name + '.csv') for name in selection['selected']]
    files += [TRIALS / 'selection.json', OUT / 'account-status.json']
    pins = {str(p.relative_to(ROOT)): sha(p) for p in files}
    parent = pd.read_csv(files[0], dtype=str, keep_default_na=False)
    queries = pd.read_parquet(files[1])
    index = pd.read_parquet(files[2], columns=['smiles', 'metric_key'])
    keys = dict(zip(index.smiles, index.metric_key))
    labels = pd.concat([pd.read_csv(p, dtype=str) for p in files[3:5]], ignore_index=True)
    labels = labels.set_index('molecule_id').loc[parent.molecule_id]
    assert len(labels) == labels.key.nunique() == len(parent) == 240
    query_modes = queries.drop_duplicates('molecule_id').set_index('molecule_id').ionization_mode
    positive = parent.molecule_id.map(query_modes).eq('positive').to_numpy()
    assert positive.sum() == 160
    positions = []
    candidate_keys = []
    for row in parent.itertuples(index=False):
        values = [keys[s] for s in row.smiles.split(';')]
        assert 1 <= len(values) <= 25 and len(values) == len(set(values))
        truth = labels.loc[row.molecule_id, 'key']
        positions.append(values.index(truth) + 1 if truth in values else 0)
        candidate_keys.append(values)
    ranks = np.asarray(positions)
    rr = np.asarray([1/r if r else 0. for r in ranks])
    expected_base = selection['results'][selection['selected'][0]]['full240']['baseline_mrr25']
    assert abs(rr.mean() - expected_base) < 1e-12
    brute_cases = verify_oracle()
    oracle = np.asarray([best_rrf_rank(len(k), r) for k, r in zip(candidate_keys, ranks)])
    oracle_rr = np.asarray([1/r if r else 0. for r in oracle])
    assert np.all(oracle_rr >= rr)
    groups = {}
    for name, mask in [('all240', np.ones(240, bool)), ('positive160', positive), ('negative80', ~positive)]:
        r = ranks[mask]
        present, first = int((r > 0).sum()), int((r == 1).sum())
        groups[name] = dict(n=int(mask.sum()), mrr25=float(rr[mask].mean()), truth_present=present,
            truth_absent=int((r == 0).sum()), rank1_correct=first,
            rank2_to5=int(((r >= 2) & (r <= 5)).sum()), rank6_to10=int(((r >= 6) & (r <= 10)).sum()),
            rank11_to25=int((r >= 11).sum()), oracle_same_candidates_free_rank=present/len(r),
            oracle_keep_rank1_free_tail=(first + .5*(present-first))/len(r),
            oracle_fixed_point2_rrf=float(oracle_rr[mask].mean()),
            fixed_rrf_remaining_headroom=float((oracle_rr[mask]-rr[mask]).mean()))
    predictions = {}
    trial_ranks = {}
    for name in selection['selected']:
        frame = pd.read_csv(TRIALS / (name + '.csv'), dtype=str, keep_default_na=False)
        assert frame.molecule_id.tolist() == parent.molecule_id.tolist()
        after = []
        changed = 0
        for before, row in zip(parent.itertuples(index=False), frame.itertuples(index=False)):
            b, a = before.smiles.split(';'), row.smiles.split(';')
            assert a[0] == b[0] and set(a) == set(b)
            truth = labels.loc[row.molecule_id, 'key']
            values = [keys[s] for s in a]
            after.append(values.index(truth) + 1 if truth in values else 0)
            changed += int(a != b)
        ar = np.asarray(after); arr = np.asarray([1/r if r else 0. for r in ar])
        assert abs(arr.mean() - selection['results'][name]['full240']['candidate_mrr25']) < 1e-12
        trial_ranks[name] = ar
        predictions[name] = dict(mrr25=float(arr.mean()), delta=float((arr-rr).mean()),
            changed_lists=changed, changed_truth_ranks=int((ar != ranks).sum()),
            improved=int((arr > rr).sum()), harmed=int((arr < rr).sum()),
            public_score=.478, development_ci95=selection['results'][name]['full240']['ci95'])
    a, b = [pd.read_csv(TRIALS / (n + '.csv'), dtype=str) for n in selection['selected']]
    same = int(a.smiles.eq(b.smiles).sum())
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        inputs_sha256=pins, all240_labels_previously_consumed=True, reserved_labels_read=False,
        training_performed=False, gpu_inference_performed=False, submission_performed=False,
        oracle_exhaustive_small_case_checks=brute_cases, groups=groups, latest_trials=predictions,
        two_trials_same_full_list_rows=same, two_trials_same_truth_ranks=int((trial_ranks[selection['selected'][0]] == trial_ranks[selection['selected'][1]]).sum()),
        rrf25_best_rank_by_original_rank={str(r): best_rrf_rank(25, r) for r in [2,3,5,10,15,20,25]},
        statistical_promotion=False, new_recipe_selected=False,
        limitation='Post-hoc description of used source-shifted development only. Oracles know the answers and are unattainable quality ceilings, not hidden scores. Public aggregate results cannot locate hidden per-molecule errors or prove the pending P18 cleanup failure path executed.')
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'used240-postmortem.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(dict(molecule_id=parent.molecule_id, baseline_rank=ranks, fixed_rrf_oracle_rank=oracle)).to_csv(OUT/'used240-rank-breakdown.csv',index=False)
    print('DONE READ ONLY PLATEAU POSTMORTEM', json.dumps({k:report[k] for k in ['groups','latest_trials','two_trials_same_full_list_rows','two_trials_same_truth_ranks','rrf25_best_rank_by_original_rank']}), flush=True)


if __name__ == '__main__':
    main()
