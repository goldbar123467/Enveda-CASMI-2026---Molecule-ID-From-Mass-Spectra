"""Fixed, label-free averaging over acquisition views."""
import numpy as np


def aggregate_views(scores, groups=None):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite spectrum-by-candidate score matrix required')
    if groups is None:
        return scores.mean(0)
    groups = np.asarray(groups)
    if groups.shape != (len(scores),):
        raise ValueError('One view identifier per spectrum required')
    return np.mean([scores[groups == group].mean(0) for group in np.unique(groups)], axis=0)
