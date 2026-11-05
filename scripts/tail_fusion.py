"""Candidate ranking and a conservative, identity-disjoint tail-fusion gate.

Uses only fixed candidate lists. It cannot measure candidate generation or hidden Kaggle MRR.
"""
import hashlib

import numpy as np


def partition(key):
    """Stable molecule-level development/confirmation assignment; no label dependence."""
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16) % 2


def standardize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite candidate scores")
    return (scores - scores.mean()) / max(float(scores.std()), 1e-6)


def rrf_order(scores, weight, offset=3.):
    scores = np.asarray(scores)
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    fusion = 1 / (offset + np.arange(1, len(scores) + 1)) + weight / (offset + ranks)
    return np.argsort(-fusion, kind="stable")


def baseline_order(public):
    lead = int(rrf_order(public, .25)[0])
    return np.array([lead] + [int(i) for i in rrf_order(public, .75) if i != lead])


def tail_order(public, ours, alpha):
    """Public rank-1 decision stays fixed even if the new model disagrees."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    lead = int(baseline_order(public)[0])
    blend = (1 - alpha) * standardize(public) + alpha * standardize(ours)
    return np.array([lead] + [int(i) for i in rrf_order(blend, .75) if i != lead])


def reciprocal_rank(labels, order):
    hit = np.flatnonzero(np.asarray(labels)[order])
    return float(1 / (hit[0] + 1)) if len(hit) and hit[0] < 25 else 0.


def paired_summary(before, after, seed=20261105):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 1 or len(before) == 0:
        raise ValueError("Paired nonempty molecule vectors required")
    delta = after - before
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(n=len(delta), baseline_mrr25=float(before.mean()), candidate_mrr25=float(after.mean()),
                delta=float(delta.mean()), ci95=[float(lo), float(hi)],
                improved=int((delta > 1e-12).sum()), harmed=int((delta < -1e-12).sum()))


def passes(summary):
    return summary["n"] >= 40 and summary["delta"] > 0 and summary["ci95"][0] > 0
