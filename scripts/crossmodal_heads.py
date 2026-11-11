"""Trainable alignment over frozen spectra/molecule vectors, not generation."""
import numpy as np


def fit_ridge(x,y,l2,block_size=2048,report=lambda **kwargs:None):
    assert x.ndim==y.ndim==2 and len(x)==len(y) and len(x)>0 and l2>0
    n,d=x.shape;gram=np.zeros((d+1,d+1),np.float64);rhs=np.zeros((d+1,y.shape[1]),np.float64)
    for start in range(0,n,block_size):
        a=np.asarray(x[start:start+block_size],np.float64);b=np.asarray(y[start:start+block_size],np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        a=np.concatenate([a,np.ones((len(a),1))],axis=1)
        gram+=a.T@a;rhs+=a.T@b;report(stage='ridge_sufficient_statistics',completed=min(n,start+len(a)),total=n)
    penalty=np.eye(d+1)*float(l2);penalty[-1,-1]=0.
    weights=np.linalg.solve(gram/n+penalty,rhs/n)
    assert np.isfinite(weights).all()
    return weights


def ridge_predict(x,weights):
    assert x.shape[1]+1==weights.shape[0]
    return np.asarray(x)@weights[:-1]+weights[-1]


def normalize_rows(x):
    x=np.asarray(x);norm=np.linalg.norm(x,axis=1,keepdims=True)
    assert np.isfinite(x).all() and np.all(norm>1e-9)
    return x/norm


def pair_batch(rng,count,formulas,groups,size=256,fraction=.5):
    ids=rng.choice(count,size=size,replace=False)
    occupied=set(ids.tolist())
    for j in range(size//2):
        pool=groups[formulas[ids[j]]]
        if len(pool)<2 or rng.random()>=fraction:continue
        for _ in range(8):
            other=int(pool[rng.integers(len(pool))])
            if other not in occupied:
                occupied.remove(int(ids[size//2+j]));ids[size//2+j]=other;occupied.add(other);break
    assert len(set(ids))==size and np.all((ids>=0)&(ids<count))
    return ids
