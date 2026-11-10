"""Identity-unique, checkpointable fixed source-stratum sampling."""
import numpy as np


def balanced_ids(members,others,size,rng,fraction=.5):
    count=int(size*fraction)
    assert count==size*fraction and 0<count<size
    assert len(members)>=count and len(others)>=size-count
    ids=np.r_[rng.choice(members,count,replace=False),rng.choice(others,size-count,replace=False)]
    rng.shuffle(ids)
    assert len(np.unique(ids))==size,'Source strata must be identity-disjoint'
    return ids
