"""Fresh fixed-schedule nonlinear head on RTX2070; no quality-based selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import json,time,math,random
from datetime import datetime,timezone
from torch.nn import functional as F
from precision_cache import ROOT,sha,write
from crossmodal_heads import pair_batch
from c40_gpu_model import CrossModalHead
from training_monitor import system_memory


def atomic_torch(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--limit-updates',type=int);parser.add_argument('--plan',type=str,default='configs/c40-crossmodal-heads-20261111.json');args=parser.parse_args()
    config=ROOT/args.plan;cfg=json.loads(config.read_text());spec=cfg['gpu'];out=ROOT/cfg['output']
    directory=out/'gpu-nonlinear';directory.mkdir(exist_ok=True)
    assert not (directory/'complete.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(0)
    torch.cuda.set_per_process_memory_fraction(spec['memory_fraction'])
    frozen=json.loads((out/'head-feature-freeze.json').read_text())
    for n,h in frozen['files_sha256'].items():assert sha(out/n)==h,n
    contract=dict(config_sha256=sha(config),feature_freeze_sha256=sha(out/'head-feature-freeze.json'),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in ['train_c40_gpu.py','c40_gpu_model.py','crossmodal_heads.py']},
        random_initialization=True,old_retrieval_head_weights_not_used=True,validation_metrics_read=False)
    pin=directory/'source-contract.json'
    if pin.exists():assert json.loads(pin.read_text())==contract,'Source/config changed'
    else:write(pin,contract)
    random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
    rng=np.random.default_rng(spec['seed']);model=CrossModalHead(spec).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    x=np.load(out/'fit-features.npy',mmap_mode='r');y=np.load(out/'fit-targets.npy',mmap_mode='r')
    identity=np.load(out/'fitting-identities.npy',allow_pickle=False)
    mol=pd.read_parquet(ROOT/cfg['cache']/'molecules.parquet',columns=['formula'])
    formulas=mol.formula.iloc[identity].to_numpy();groups={f:g.index.to_numpy() for f,g in pd.Series(formulas).groupby(formulas,sort=False)}
    step=0;history=[];elapsed_before=0.;resume=directory/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False);assert state['contract']==contract
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];history=state['history']
        rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        elapsed_before=float(state['elapsed_seconds']);del state
    started=time.perf_counter();invoked=0;losses=[]
    def report(**kw):
        assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active'],'Goal stopped'
        assert system_memory()['free_gib']>=spec['minimum_free_ram_gib'],'RAM reserve reached'
        elapsed=elapsed_before+time.perf_counter()-started
        assert elapsed<spec['maximum_hours']*3600,'Frozen training wall budget reached'
        row=dict(updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),step=step,total_steps=spec['steps'],elapsed_seconds=elapsed,
                 maximum_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**kw)
        write(directory/'status.json',row);print('C40 GPU',json.dumps(row),flush=True)
    def save():
        atomic_torch(dict(contract=contract,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,history=history,
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            elapsed_seconds=elapsed_before+time.perf_counter()-started),resume)
    report(state='training',stage='New nonlinear alignment from random initialization')
    try:
        while step<spec['steps']:
            ids=pair_batch(rng,len(x),formulas,groups,spec['batch_size'],spec['same_formula_partner_fraction'])
            a=torch.tensor(np.asarray(x[ids]),device='cuda');b=torch.tensor(np.asarray(y[ids]),device='cuda')
            warm=min(1.,(step+1)/spec['warmup_steps'])
            decay=1. if step<spec['warmup_steps'] else .5*(1+math.cos(math.pi*(step-spec['warmup_steps'])/(spec['steps']-spec['warmup_steps'])))
            for group in optimizer.param_groups:group['lr']=spec['learning_rate']*warm*decay
            logits=model(a,b);target=torch.arange(len(ids),device='cuda')
            loss=.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm);optimizer.step();step+=1;invoked+=1;losses.append(float(loss.detach()))
            if step%100==0:
                history.append(dict(step=step,loss=float(np.mean(losses))));report(state='training',stage='Nonlinear frozen-feature alignment',loss=history[-1]['loss'])
                losses=[]
            if step%spec['checkpoint_interval']==0:save()
            if args.limit_updates and invoked>=args.limit_updates:
                save();report(state='pilot_checkpoint',stage='Resumable prefix of fixed schedule');return
    except BaseException:
        save();raise
    final=dict(cfg=spec,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},steps=step,contract=contract)
    atomic_torch(final,directory/'final_model.pt')
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),model_sha256=sha(directory/'final_model.pt'),
        source_contract_sha256=sha(pin),steps=step,fitting_identities=len(x),history=history,
        final_checkpoint_only=True,old_head_weights_reused=False,validation_metrics_read=False,C13_read=False,
        maximum_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,qualified_for_submission=False)
    write(directory/'complete.json',result);report(state='complete',stage='Fixed30000-update nonlinear head fitted')
    print('DONE C40 GPU TRAINING',step,flush=True)


if __name__=='__main__':main()
