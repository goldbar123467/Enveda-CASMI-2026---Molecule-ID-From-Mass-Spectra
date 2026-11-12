"""Restrict one fixed RRF isomer ranking to parent positions3through10."""
import numpy as np


def order(formulas,scores,eligible,supported=True,weight=.2,offset=3.,protected_prefix=2):
    n=len(formulas);assert 1<=n<=25 and len(scores)==len(eligible)==n and weight==.2 and offset==3. and protected_prefix==2
    result=list(range(n))
    if not supported:return result
    scores=np.asarray(scores,float);eligible=np.asarray(eligible,bool);groups={}
    for i in range(2,min(10,n)):groups.setdefault(formulas[i],[]).append(i)
    for positions in groups.values():
        if len(positions)<2 or not eligible[positions].all():continue
        assert np.isfinite(scores[positions]).all()
        preferred=sorted(positions,key=lambda i:(-scores[i],i));ranks={i:j for j,i in enumerate(preferred,1)}
        utilities={i:1/(3+i+1)+.2/(3+ranks[i]) for i in positions}
        chosen=sorted(positions,key=lambda i:(-utilities[i],i))
        for slot,choice in zip(positions,chosen):result[slot]=choice
    assert result[:2]==list(range(min(2,n))) and result[10:]==list(range(10,n))
    assert sorted(result)==list(range(n)) and [formulas[i] for i in result]==list(formulas)
    return result
