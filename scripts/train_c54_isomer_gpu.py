"""Fit one fixed-schedule residual on clean training identities, never queries."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import torch
import numpy as np
import pandas as pd
import argparse,json,math,random,time
from pathlib import Path
from datetime import datetime,timezone
from c54_gpu_model import BoundedResidualHead
from precision_cache import ROOT,sha,write
from training_monitor import system_memory


def atomic_save(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/c54-isomer-fingerprint-pair-20261112.json');args=parser.parse_args()
    config=ROOT/args.plan;cfg=json.loads(config.read_text());out=ROOT/cfg['output'];spec=cfg['gpu']
    target=out/'gpu-residual';target.mkdir(exist_ok=True);assert not (target/'complete.json').exists()
    assert read_active() and torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(spec['memory_fraction'])
    freeze=json.loads((out/'head-feature-freeze.json').read_text());cpu=json.loads((out/'cpu-ridge/complete.json').read_text())
    for name,pin in freeze['files_sha256'].items():assert sha(out/name)==pin
    assert cpu['feature_freeze_sha256']==sha(out/'head-feature-freeze.json') and cpu['model_sha256']==sha(out/'cpu-ridge/model.npz')
    contract=dict(config_sha256=sha(config),feature_freeze_sha256=sha(out/'head-feature-freeze.json'),CPU_anchor_model_sha256=cpu['model_sha256'],
        source_sha256={name:sha(ROOT/'scripts'/name) for name in ['train_c54_isomer_gpu.py','c54_gpu_model.py']},
        C39_quality_known_before_design=True,C51_quality_known_before_design=True,query_rows_used_for_fitting=False,new_source_quality_read=False,C13_read=False,
        anchor='Frozen same-cycle clean CPU ridge; no old C40/C42 fitted-head weights loaded.')
    pin=target/'source-contract.json'
    if pin.exists():assert json.loads(pin.read_text())==contract
    else:write(pin,contract)
    random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed']);rng=np.random.default_rng(spec['seed'])
    anchor=torch.from_numpy(np.load(out/'cpu-ridge/model.npz')['weights'])
    model=BoundedResidualHead(spec,anchor).cuda().train();optimizer=torch.optim.AdamW(model.residual.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    x=np.load(out/'fit-features.npy',mmap_mode='r');y=np.load(out/'fit-targets.npy',mmap_mode='r')
    assert x.shape==(freeze['fitting_identities'],1039) and y.shape==(len(x),2048)
    fit_ids=np.load(out/'fitting-identities.npy');formulas=pd.read_parquet(ROOT/cfg['cache']/'molecules.parquet',columns=['formula']).formula.iloc[fit_ids].to_numpy();members={f:np.asarray(ix,np.int64) for f,ix in pd.DataFrame(dict(formula=formulas)).groupby('formula').indices.items()}
    step=0;history=[];elapsed_before=0.;resume=target/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False);assert state['contract']==contract
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];history=state['history']
        rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);elapsed_before=state['elapsed_seconds'];del state
    started=time.perf_counter();losses=[]
    def report(state,**kwargs):
        assert read_active() and system_memory()['free_gib']>=spec['minimum_free_ram_gib']
        elapsed=elapsed_before+time.perf_counter()-started;assert elapsed<spec['maximum_hours']*3600
        record=dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),state=state,step=step,total_steps=spec['steps'],
            elapsed_seconds=elapsed,maximum_GPU_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**kwargs)
        write(target/'status.json',record);print('C54 GPU ISOMER',json.dumps(record),flush=True)
    def save():
        atomic_save(dict(contract=contract,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,history=history,
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            elapsed_seconds=elapsed_before+time.perf_counter()-started),resume)
    report('training')
    try:
        while step<spec['steps']:
            ids=rng.choice(len(x),size=spec['batch_size'],replace=False)
            a=torch.tensor(np.asarray(x[ids]),device='cuda');b=torch.tensor(np.asarray(y[ids]),device='cuda')
            warm=min(1.,(step+1)/spec['warmup_steps']);decay=1. if step<spec['warmup_steps'] else .5*(1+math.cos(math.pi*(step-spec['warmup_steps'])/(spec['steps']-spec['warmup_steps'])))
            for group in optimizer.param_groups:group['lr']=spec['learning_rate']*warm*decay
            prediction=model.encode_spectra(a);positive=(prediction*b).sum(-1);loss=(1-positive).mean()
            paired=[];negative=[]
            for j,identity in enumerate(ids):
                pool=members[formulas[identity]]
                if len(pool)<2:continue
                other=int(pool[rng.integers(len(pool))])
                while other==identity:other=int(pool[rng.integers(len(pool))])
                paired.append(j);negative.append(other)
            assert len(paired)>0
            neg=torch.tensor(np.asarray(y[negative]),device='cuda');where=torch.tensor(paired,device='cuda')
            wrong=(prediction[where]*neg).sum(-1)
            pair_loss=torch.nn.functional.softplus((wrong-positive[where]+spec['pair_margin'])/spec['pair_temperature']).mean()
            loss=loss+spec['pair_weight']*pair_loss;assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.residual.parameters(),1.);assert torch.isfinite(norm)
            optimizer.step();step+=1;losses.append(float(loss.detach()))
            if step%100==0:
                history.append(dict(step=step,loss=float(np.mean(losses))));report('training',loss=history[-1]['loss']);losses=[]
            if step%spec['checkpoint_interval']==0:save()
    except BaseException:
        save();raise
    assert all(sha(ROOT/'scripts'/name)==value for name,value in contract['source_sha256'].items()) and sha(config)==contract['config_sha256']
    final=target/'final_model.pt';atomic_save(dict(cfg=spec,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},steps=step,contract=contract),final)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),model_sha256=sha(final),source_contract_sha256=sha(pin),
        feature_freeze_sha256=sha(out/'head-feature-freeze.json'),fitting_identities=len(x),steps=step,history=history,
        fixed_final_checkpoint=True,CPU_anchor_sha256=cpu['model_sha256'],maximum_relative_residual_norm=spec['maximum_relative_residual_norm'],
        anchor_weights_frozen=True,molecular_geometry_unchanged=True,new_source_quality_read=False,C13_read=False,qualified_for_scoring=False)
    write(target/'complete.json',result);report('complete');print('C54 FIXED GPU FIT COMPLETE',sha(final),flush=True)


def read_active():return json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']


if __name__=='__main__':main()
