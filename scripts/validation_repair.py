"""Strict query-level evaluation and CPU calibration on frozen serving features."""
import hashlib
import numpy as np


def reciprocal_rank(order, keys, truth):
    assert len(order) == len(set(order)) and len(order) <= 25
    values = [keys[i] for i in order]
    assert len(values) == len(set(values)), 'Duplicate official identity'
    return 1. / (values.index(truth) + 1) if truth in values else 0.


def evaluate_orders(blocks, orders):
    assert len(blocks) == len(orders), 'Missing query; never drop absent truths'
    return np.asarray([reciprocal_rank(o, b['keys'], b['truth']) for b, o in zip(blocks, orders)])


def folds_for(blocks, indices, count, salt):
    groups = {}
    for i in indices:
        groups.setdefault(blocks[i]['group'], []).append(i)
    assert len(groups) >= count, 'Too few independent chemistry groups'
    def tie(group):
        return hashlib.sha256((salt + str(group)).encode()).hexdigest()
    result = {}; sizes = np.zeros(count, int); modes = np.zeros((count, 2), int)
    for group in sorted(groups, key=lambda g: (-len(groups[g]), tie(g))):
        members = groups[group]
        composition = np.asarray([sum(blocks[i]['mode'] == m for i in members) for m in ['positive','negative']])
        f = min(range(count), key=lambda j: (sizes[j], np.sum((modes[j]+composition)**2), j))
        for i in members: result[i] = f
        sizes[f] += len(members); modes[f] += composition
    assert set(result) == set(indices) and all(sizes)
    return result


def relative(values):
    """Query-only transforms; unavailable channels retain a separate mask."""
    values = np.asarray(values, float); valid = np.isfinite(values)
    z = np.zeros(len(values)); rank = np.zeros(len(values))
    if valid.sum() > 1 and np.ptp(values[valid]) > 1e-8:
        v = values[valid]; z[valid] = np.clip((v-v.mean())/max(v.std(), 1e-8), -3, 3)/3
        rank[valid] = [1./(1+int((v > x).sum())) for x in v]
    return valid.astype(float), z, rank


def zscore(values):
    a = np.asarray(values, float)
    return (a-a.mean())/max(float(a.std()), 1e-8)


def formula_order(block, learned_scores, alpha, protect=.9, eligible=None):
    n = len(block['keys']); original = list(range(n))
    assert np.isfinite(learned_scores).all() and 0 <= alpha <= 1
    if alpha == 0 or block['library'] >= protect:
        return original
    groups = {}
    for i, f in enumerate(block['formulas']): groups.setdefault(f, []).append(i)
    answer = original.copy()
    for positions in groups.values():
        if len(positions) < 2 or (eligible is not None and not all(eligible[i] for i in positions)):
            continue
        base = zscore([1./(4+i) for i in positions])
        new = zscore(np.asarray(learned_scores)[positions])
        utility = (1-alpha)*base + alpha*new
        chosen = sorted(range(len(positions)), key=lambda j: (-utility[j], positions[j]))
        for slot, j in zip(positions, chosen): answer[slot] = positions[j]
    assert set(answer) == set(original)
    assert [block['formulas'][i] for i in answer] == block['formulas']
    return answer


def fit_pairwise(blocks, train, feature_indices, l2=.1):
    """Equal-query-weight ridge on positive same-formula pair differences."""
    rows = []; weights = []; eligible_queries = 0
    for i in train:
        block = blocks[i]
        if block['truth'] not in block['keys']: continue
        t = block['keys'].index(block['truth'])
        decoys = [j for j, f in enumerate(block['formulas']) if f == block['formulas'][t] and j != t]
        if not decoys: continue
        x = block['x'][:, feature_indices]
        rows.extend(x[t]-x[j] for j in decoys)
        weights.extend([1./len(decoys)]*len(decoys)); eligible_queries += 1
    assert rows, 'No same-formula training contrasts'
    d = np.asarray(rows); w = np.asarray(weights); normalizer = w.sum()
    lhs = (d.T*w) @ d / normalizer + l2*np.eye(d.shape[1])
    rhs = d.T @ w / normalizer
    beta = np.linalg.solve(lhs, rhs)
    assert np.isfinite(beta).all()
    return beta, dict(training_queries=len(train), eligible_queries=eligible_queries, pairs=len(rows),
                      train_ids=[blocks[i]['mid'] for i in train], train_groups=sorted({blocks[i]['group'] for i in train}))


def choose_alpha(blocks, train, feature_indices, config):
    folds = folds_for(blocks, train, config['inner_folds'], config['split_salt']+':inner')
    totals = {a: 0. for a in config['alpha_candidates']}; audits = []
    for fold in range(config['inner_folds']):
        fitting = [i for i in train if folds[i] != fold]; testing = [i for i in train if folds[i] == fold]
        assert not ({blocks[i]['group'] for i in fitting} & {blocks[i]['group'] for i in testing})
        beta, audit = fit_pairwise(blocks, fitting, feature_indices, config['ridge_l2'])
        audit.update(test_ids=[blocks[i]['mid'] for i in testing], test_groups=sorted({blocks[i]['group'] for i in testing}))
        audits.append(audit)
        for i in testing:
            b = blocks[i]; scores = b['x'][:, feature_indices] @ beta
            for alpha in totals:
                order = formula_order(b, scores, alpha, config['protect_library_at'])
                totals[alpha] += reciprocal_rank(order, b['keys'], b['truth'])
    best = max(totals.values())
    alpha = min(a for a, value in totals.items() if value >= best-1e-12)
    return alpha, dict(inner_mrr={str(a): v/len(train) for a,v in totals.items()}, audits=audits)


def crossfit(blocks, feature_indices, config, outer=None):
    indices = list(range(len(blocks)))
    folds = outer or folds_for(blocks, indices, config['outer_folds'], config['split_salt'])
    orders = [None]*len(blocks); audits = []
    for fold in range(config['outer_folds']):
        train = [i for i in indices if folds[i] != fold]; test = [i for i in indices if folds[i] == fold]
        assert not ({blocks[i]['group'] for i in train} & {blocks[i]['group'] for i in test})
        alpha, selection = choose_alpha(blocks, train, feature_indices, config)
        beta, audit = fit_pairwise(blocks, train, feature_indices, config['ridge_l2'])
        audit.update(fold=fold, alpha=alpha, beta=beta.tolist(), selection=selection,
                     test_ids=[blocks[i]['mid'] for i in test], test_groups=sorted({blocks[i]['group'] for i in test}))
        audits.append(audit)
        for i in test:
            b = blocks[i]
            orders[i] = formula_order(b, b['x'][:,feature_indices] @ beta, alpha, config['protect_library_at'])
    assert all(o is not None for o in orders)
    return orders, audits


def clustered_summary(before, after, groups, draws=10000, seed=20261136):
    before, after = np.asarray(before), np.asarray(after)
    assert len(before) == len(after) == len(groups) and len(before)
    delta = after-before
    unique = sorted(set(groups)); sums = np.asarray([delta[np.asarray(groups)==g].sum() for g in unique])
    sizes = np.asarray([np.sum(np.asarray(groups)==g) for g in unique])
    rng = np.random.default_rng(seed); picked = rng.integers(len(unique), size=(draws,len(unique)))
    sampled = sums[picked].sum(1)/sizes[picked].sum(1)
    return dict(n=len(before), independent_groups=len(unique), baseline_mrr25=float(before.mean()), candidate_mrr25=float(after.mean()),
                delta=float(delta.mean()), ci95=np.quantile(sampled,[.025,.975]).tolist(), improved=int((delta>0).sum()), harmed=int((delta<0).sum()))


def validation_gate(primary, polarity, controls, config, independent_confirmation=False):
    reasons = []
    if primary['delta'] < config['minimum_primary_delta']: reasons.append('primary_gain_below_predeclared_minimum')
    if primary['ci95'][0] <= config['minimum_primary_ci_lower']: reasons.append('primary_grouped_interval_includes_zero')
    if any(v['delta'] < -1e-12 for v in polarity.values()): reasons.append('polarity_regression')
    if controls['P14']['delta'] >= 0: reasons.append('known_harmful_P14_has_wrong_local_sign')
    if not independent_confirmation: reasons.append('no_untouched_independent_confirmation')
    return dict(passed=not reasons, reasons=reasons, leaderboard_calibrated=controls['P14']['delta']<0,
                automatic_gpu_work=False, automatic_submission=False)
