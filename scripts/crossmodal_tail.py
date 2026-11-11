"""One fixed same-formula tail rule for the two predeclared C40 heads."""
import numpy as np


def order(formulas,scores,eligible,supported=True,weight=.2,offset=3.):
    n=len(formulas);answer=list(range(n))
    assert 1<=n<=25 and len(scores)==len(eligible)==n
    assert weight==.2 and offset==3.,'This recipe is frozen, not a tuning grid'
    if not supported:return answer
    scores=np.asarray(scores,float);eligible=np.asarray(eligible,bool)
    groups={}
    for i,f in enumerate(formulas):
        if i:groups.setdefault(f,[]).append(i)
    for positions in groups.values():
        if len(positions)<2 or not np.all(eligible[positions]):continue
        assert np.isfinite(scores[positions]).all()
        ranked=sorted(positions,key=lambda i:(-scores[i],i))
        ranks={i:r+1 for r,i in enumerate(ranked)}
        utility={i:1./(offset+i+1)+weight/(offset+ranks[i]) for i in positions}
        chosen=sorted(positions,key=lambda i:(-utility[i],i))
        for slot,i in zip(positions,chosen):answer[slot]=i
    assert answer[0]==0 and set(answer)==set(range(n))
    assert [formulas[i] for i in answer]==list(formulas)
    return answer
