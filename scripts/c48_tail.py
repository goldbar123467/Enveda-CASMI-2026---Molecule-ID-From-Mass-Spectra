"""Fixed CPU prefix1 and GPU prefix2 same-formula tail permutations."""
import numpy as np


def order(formulas,scores,eligible,supported=True,weight=.2,offset=3.,protected_prefix=1):
    n=len(formulas);answer=list(range(n))
    assert 1<=n<=25 and len(scores)==len(eligible)==n
    assert weight==.2 and offset==3. and protected_prefix in [1,2]
    if not supported:return answer
    scores=np.asarray(scores,float);eligible=np.asarray(eligible,bool);groups={}
    for i,formula in enumerate(formulas):
        if i>=protected_prefix:groups.setdefault(formula,[]).append(i)
    for positions in groups.values():
        if len(positions)<2 or not np.all(eligible[positions]):continue
        assert np.isfinite(scores[positions]).all()
        ranked=sorted(positions,key=lambda i:(-scores[i],i));ranks={i:r+1 for r,i in enumerate(ranked)}
        utility={i:1/(offset+i+1)+weight/(offset+ranks[i]) for i in positions}
        chosen=sorted(positions,key=lambda i:(-utility[i],i))
        for slot,i in zip(positions,chosen):answer[slot]=i
    assert answer[:min(protected_prefix,n)]==list(range(min(protected_prefix,n))) and set(answer)==set(range(n))
    assert [formulas[i] for i in answer]==list(formulas)
    return answer
