"""A GPU-backed agreement rule: every inversion must improve BOTH RRF orders."""
import numpy as np


def utilities(positions,scores):
    preference=sorted(positions,key=lambda i:(-scores[i],i));rank={i:j for j,i in enumerate(preference,1)}
    return {i:1/(3+i+1)+.2/(3+rank[i]) for i in positions}


def order(formulas,scores,eligible,supported=True,weight=.2,offset=3.,reference_scores=None):
    n=len(formulas);assert 1<=n<=25 and len(scores)==len(eligible)==n and weight==.2 and offset==3.
    result=list(range(n))
    if not supported:return result
    scores=np.asarray(scores,float);eligible=np.asarray(eligible,bool)
    if reference_scores is not None:reference_scores=np.asarray(reference_scores,float);assert reference_scores.shape==scores.shape
    groups={}
    for i,f in enumerate(formulas):
        if i:groups.setdefault(f,[]).append(i)
    for positions in groups.values():
        if len(positions)<2 or not eligible[positions].all():continue
        assert np.isfinite(scores[positions]).all();own=utilities(positions,scores)
        if reference_scores is None:chosen=sorted(positions,key=lambda i:(-own[i],i))
        else:
            assert np.isfinite(reference_scores[positions]).all();reference=utilities(positions,reference_scores);chosen=list(positions)
            # A bounded adjacent sorting process. Each newly inverted pair must
            # be strictly preferred by both heads; ties preserve parent order.
            for _ in range(len(positions)):
                changed=False
                for j in range(len(chosen)-1):
                    a,b=chosen[j:j+2]
                    if own[b]>own[a] and reference[b]>reference[a]:chosen[j],chosen[j+1]=b,a;changed=True
                if not changed:break
            else:raise AssertionError('Consensus sorting exceeded its inversion bound')
        for slot,choice in zip(positions,chosen):result[slot]=choice
    assert result[0]==0 and set(result)==set(range(n)) and [formulas[i] for i in result]==list(formulas)
    return result
