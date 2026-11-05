"""Controlled 32-epoch continuation: general vs a fixed curated-NP mixture.

The parent holdout must exclude every evaluation identity across all sources.
Full-data parents are accepted only for production refits. Smoke output is
explicitly marked and is never a qualifying model.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

import local_seeds as L
from np_sampling import mixture_probabilities,patch_training_loop


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['general','np'],required=True)
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--np-keys',type=Path,required=True)
    ap.add_argument('--stage',choices=['holdout','refit'],default='holdout')
    ap.add_argument('--seed',type=int,default=22)
    ap.add_argument('--mixture',type=float,choices=[.25,.5,.75],default=.5)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); started=time.time()
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    assert parent['stage']==('holdout' if args.stage=='holdout' else 'full_refit'),'Refit-parent leakage is forbidden in holdout evaluation'
    assert parent['epoch']==128
    L.OVERRIDES.update(parent.get('cfg_overrides',{}))
    if args.stage=='holdout':
        actual=set((L.OUT/'np_holdout_keys.txt').read_text().split())
        excluded=set(args.parent.parent.parent.joinpath('np_holdout_keys.txt').read_text().split())
        assert len(actual)==27600 and actual==excluded
    keys=set(args.np_keys.read_text().split())
    original_setup=L.setup

    def setup(split):
        cfg,data,device=original_setup(split)
        if args.smoke:
            if args.stage=='holdout':
                data.train=data.train[:1024]
                data.val=data.val[:16]
            # Exercise real NN/optimizer/sampling, using tiny deterministic pools;
            # scientific metrics and the original pool construction are not tested by this smoke.
            ids=np.arange(len(data.mol)) if args.stage=='refit' else data.train.copy()
            class TinyPools:
                def __init__(self,allowed,count):
                    self.small=np.asarray(allowed)[:count+1]
                    self.count=count
                def __getitem__(self,key):
                    return self.small[self.small!=key][:self.count]
            L.T.hard_pools=lambda mol,allowed,count:TinyPools(allowed,count)
            L.T.candidate_sets=lambda d,c:{int(i):np.r_[i,ids[:31]] for i in d.val}
        training=np.arange(len(data.mol)) if args.stage=='refit' else data.train
        probability,audit=mixture_probabilities(training,data.mol.key.astype(str).to_numpy(),keys,args.mixture)
        assert args.smoke or audit['annotated_np_molecules']>1000
        original=inspect.getsource(L.T.train_stage)
        patched=patch_training_loop(original)
        namespace=dict(L.T.__dict__,_INITIAL_STATE=parent['state_dict'],
                       _SAMPLING_PROBABILITIES=probability if args.mode=='np' else None)
        exec(compile(patched,'fixed_mixture_train_stage','exec'),namespace)
        L.T.train_stage=namespace['train_stage']
        audit.update(mode=args.mode,stage=args.stage,smoke=args.smoke,epochs=1 if args.smoke else 32,lr=2e-5,
                     parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),parent_epoch=parent['epoch'],
                     original_training_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                     patched_training_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                     curated_keys_sha256=hashlib.sha256('\n'.join(sorted(keys)).encode()).hexdigest(),
                     training_indices_sha256=hashlib.sha256(training.astype('<i8').tobytes()).hexdigest(),
                     heldout_identity_sha256=hashlib.sha256('\n'.join(sorted(data.mol.key.iloc[data.val].astype(str))).encode()).hexdigest() if args.stage=='holdout' else None,
                     effective_np_fraction=audit['sampled_np_fraction'] if args.mode=='np' else audit['baseline_np_fraction'],
                     objective='original retrieval + original fingerprint auxiliary; no distillation; original negative pools and FP class weights',
                     fixed_final_epoch=True)
        L.OUT.mkdir(parents=True,exist_ok=True)
        (L.OUT/'finetune_manifest.json').write_text(json.dumps(audit,indent=2))
        print('FINETUNE MANIFEST',json.dumps(audit),flush=True)
        return cfg,data,device

    L.setup=setup
    if args.smoke:
        L.OVERRIDES.update(steps_per_epoch=4)
    L.train(args.stage,[args.seed],1 if args.smoke else 32,'np' if args.stage=='holdout' else 'kit',lr=2e-5)
    folder=L.OUT/('holdout_np' if args.stage=='holdout' else 'refit')
    paths=list(folder.glob('*.pt'));assert len(paths)==1
    checkpoint=torch.load(paths[0],map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==(1 if args.smoke else 32)
    assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
    assert any(not torch.equal(t,parent['state_dict'][key]) for key,t in checkpoint['state_dict'].items()),'No learned parameter changed'
    checkpoint['initialization_parent_sha256']=hashlib.sha256(args.parent.read_bytes()).hexdigest()
    checkpoint['initialization_parent_epoch']=128
    checkpoint['continuation_epochs']=checkpoint['epoch']
    checkpoint['smoke']=args.smoke
    torch.save(checkpoint,paths[0])
    print(f'[{time.time()-started:.0f}s] DONE continuation verified; smoke={args.smoke}',flush=True)


if __name__=='__main__':
    main()
