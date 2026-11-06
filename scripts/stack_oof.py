"""Validate and combine the two shared-fold candidate-score collections.

The L2 listwise stack is a development diagnostic. A reserved SHA256 cohort is
excluded from fitting the stack. Base models see none of their own query's fold,
but other-fold base models can see a reserved identity; this is ordinary OOF,
not nested validation of the complete ensemble-building procedure.
No Kaggle test scores, test labels or leaderboard scores enter the fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def standardize(x):
    return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-6)


def mrr(scores):
    ranks=1+(scores[:,1:]>=scores[:,:1]).sum(1)
    return np.where(ranks<=25,1/ranks,0.)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    args=ap.parse_args();root=args.root
    families=[]
    for family in ['control','external']:
        packs=[]
        for fold in range(5):
            folder=root/'models'/family/f'fold-{fold}'
            receipt=json.loads((folder/'complete.json').read_text())
            assert not receipt['smoke'] and receipt['epochs']==128
            path=folder/'preds/oof.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['predictions_sha256']
            assert hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()==receipt['model_sha256']
            p=dict(np.load(path,allow_pickle=False))
            assert (p['folds']==fold).all()
            packs.append(p)
        allp={k:np.concatenate([p[k] for p in packs]) for k in packs[0]}
        assert len(set(allp['keys']))==len(allp['keys'])
        families.append(allp)
    a,b=families
    expected=dict(np.load(root/'evaluation.npz',allow_pickle=False))
    for key in ['keys','ids','folds','candidates','natural_product']:
        assert np.array_equal(a[key],b[key]), f'OOF alignment mismatch: {key}'
        assert np.array_equal(a[key],expected[key]), f'Incomplete or changed evaluation cohort: {key}'
    assert a['scores'].shape == b['scores'].shape == (len(expected['ids']),256)
    assert np.isfinite(a['scores']).all() and np.isfinite(b['scores']).all()
    confirmation=np.array([int.from_bytes(hashlib.sha256(('stack-confirm:'+k).encode()).digest()[:8],'little')%5==0 for k in a['keys']])
    development=~confirmation
    x=np.stack([standardize(p[channel]) for p in families for channel in ['cosine','fingerprint']],-1).astype(np.float32)
    torch.set_num_threads(2)
    inputs=torch.from_numpy(x[development])
    weights=torch.nn.Parameter(torch.tensor([.5,.5,.5,.5]))
    opt=torch.optim.LBFGS([weights],max_iter=150,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        scores=torch.einsum('ncf,f->nc',inputs,weights)
        loss=torch.nn.functional.cross_entropy(scores,torch.zeros(len(scores),dtype=torch.long))+.1*weights.square().sum()
        loss.backward();return loss
    opt.step(closure)
    fitted=np.einsum('ncf,f->nc',x,weights.detach().numpy())
    equal=.5*standardize(a['scores'])+.5*standardize(b['scores'])
    report=dict(weights=weights.detach().tolist(),no_automatic_submission=True,
                baseline_public_mrr=.480,limitation='Truth-forced retrieval diagnostic, ordinary OOF (not nested end-to-end validation); full-pipeline independent confirmation still required',
                cohorts={})
    for name,mask in [('development',development),('confirmation',confirmation),('natural_products',a['natural_product'])]:
        baseline=mrr(a['scores'])[mask]
        report['cohorts'][name]=dict(n=int(mask.sum()))
        for label,score in [('control',a['scores']),('external',b['scores']),('equal_blend',equal),('l2_stack',fitted)]:
            rr=mrr(score)[mask]
            delta=rr-baseline
            rng=np.random.default_rng(20261106)
            boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
            report['cohorts'][name][label]=dict(mrr25=float(rr.mean()),delta=float(delta.mean()),
                                                    delta_ci95=np.quantile(boots,[.025,.975]).tolist())
    (root/'stack').mkdir(exist_ok=True)
    (root/'stack/report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(root/'stack/oof.npz',keys=a['keys'],folds=a['folds'],confirmation=confirmation,
                        control=a['scores'],external=b['scores'],equal_blend=equal,l2_stack=fitted)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
