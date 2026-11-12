"""Fit one complete CPU/GPU pair after both fixed chemistry-OOF audits.

All869development queries are USED; only the780eligible queries enter fitting. New source/confirmation outcomes remain
unavailable. This produces no prospective qualification or submission release.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp
DATA=ROOT/'outputs/local/c69-clean-parent-evidence-oof-20261112'
DEV=ROOT/'outputs/local/c70-clean-gpu-parent-evidence-oof-20261112'
OUT=ROOT/'outputs/local/c71-final-clean-direct-pair-20261112'


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    assert not (OUT/'complete-candidate-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert read(DATA/'independent-oof-audit.json')['CPU_gate_passed']
    assert read(DEV/'independent-oof-audit.json')['GPU_gate_passed']
    assert read(DEV/'runtime-parity-audit.json')['feature_values_bitwise_exact']
    cp=ROOT/'configs/c69-clean-parent-evidence-oof-20261112.json';gp=ROOT/'configs/c70-clean-gpu-parent-evidence-oof-20261112.json'
    cpu=read(cp)['cpu'];gpu=read(gp)['gpu'];freeze=read(DATA/'data-freeze.json')
    for name,pin in freeze['files_sha256'].items():assert sha(ROOT/name)==pin
    assert read(DEV/'runtime-parity-audit.json')['runtime_sha256']==sha(ROOT/'scripts/c67_ranker.py')
    q=pd.read_parquet(DATA/'queries.parquet');rows=pd.read_parquet(DATA/'candidates.parquet');x=np.load(DATA/'features.npy').astype(np.float64)
    assert len(q)==869 and q.acquisitions.sum()==3452 and len(rows)==21719
    write(OUT/'fit-run-freeze.json',dict(started_utc=stamp(),source_sha256=sha(__file__),cpu_policy_sha256=sha(cp),gpu_policy_sha256=sha(gp),
        data_freeze_sha256=sha(DATA/'data-freeze.json'),runtime_sha256=sha(ROOT/'scripts/c67_ranker.py'),
        CPU_audit_sha256=sha(DATA/'independent-oof-audit.json'),GPU_audit_sha256=sha(DEV/'independent-oof-audit.json'),
        selection='Exact audited development recipes; strict C68quarantined refit on780of869USED queries; all869retained in evaluation; no new source or C13 outcomes; fixed final epoch',C13_read=False,scoring_entries=0))
    pairs=[]
    for _,g in rows.groupby(['molecule_id','formula'],sort=False):
        if not g.available.all() or len(g)<=1 or g.target.sum()!=1:continue
        a=int(g.index[g.target.eq(1)][0]);negative=g.index[g.target.eq(0)&g.fitting_eligible]
        if not rows.fitting_eligible.iloc[a] or not len(negative):continue
        pairs.extend((a,int(b),1/len(negative)) for b in negative)
    a=np.array([v[0] for v in pairs]);b=np.array([v[1] for v in pairs]);pw=np.array([v[2] for v in pairs])
    eligible=rows.fitting_eligible.to_numpy();assert eligible.sum()==18833 and q.fitting_eligible.sum()==780
    parent=-np.log(rows.parent_rank.to_numpy(float));mean=x[eligible].mean(0);std=np.maximum(x[eligible].std(0),1e-4);z=np.clip((x-mean)/std,-10,10)
    delta=torch.tensor(z[a]-z[b],dtype=torch.float64);offset=torch.tensor(parent[a]-parent[b],dtype=torch.float64)
    weights=torch.tensor(pw/pw.sum(),dtype=torch.float64);w=torch.nn.Parameter(torch.zeros(x.shape[1],dtype=torch.float64))
    opt=torch.optim.LBFGS([w],lr=1.,max_iter=cpu['max_iter'],history_size=20,tolerance_grad=1e-9,tolerance_change=1e-11,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=(torch.nn.functional.softplus(-(offset+delta@w))*weights).sum()+cpu['ridge']*w.square().sum()
        loss.backward();return loss
    opt.step(closure);fitted=w.detach().numpy();np.savez(OUT/'cpu-model.npz',weights=fitted,mean=mean,std=std)
    grad=float(np.abs(w.grad.detach().numpy()).max());assert grad<2e-5 and not torch.cuda.is_initialized()
    anchor=parent+z@fitted;seed=read(gp)['seed']+5
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(0)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.GELU(approximate='tanh'),torch.nn.Linear(32,1)).to('cuda:0')
    torch.nn.init.zeros_(net[2].weight);torch.nn.init.zeros_(net[2].bias)
    zz=torch.tensor(z,dtype=torch.float32,device='cuda:0');aa=torch.tensor(anchor,dtype=torch.float32,device='cuda:0')
    pa=torch.tensor(a,device='cuda:0');pb=torch.tensor(b,device='cuda:0');weight=torch.tensor(pw,dtype=torch.float32,device='cuda:0');weight/=weight.sum()
    fitix=torch.tensor(np.flatnonzero(eligible),device='cuda:0')
    opt=torch.optim.AdamW(net.parameters(),lr=gpu['learning_rate'],weight_decay=gpu['weight_decay'])
    for epoch in range(gpu['epochs']):
        opt.zero_grad();correction=.25*torch.tanh(net(zz).flatten());margin=aa[pa]-aa[pb]+correction[pa]-correction[pb]
        loss=(torch.nn.functional.softplus(-margin)*weight).sum()+gpu['correction_MSE_penalty']*correction[fitix].square().mean()
        loss.backward();opt.step()
    np.savez(OUT/'gpu-model.npz',weights=fitted,mean=mean,std=std,hidden_weight=net[0].weight.detach().cpu().numpy(),
        hidden_bias=net[0].bias.detach().cpu().numpy(),output_weight=net[2].weight.detach().cpu().numpy(),output_bias=net[2].bias.detach().cpu().numpy())
    files=[Path(__file__),ROOT/'scripts/c67_ranker.py',cp,gp,DATA/'data-freeze.json',DATA/'independent-oof-audit.json',DEV/'independent-oof-audit.json',
        DEV/'runtime-parity-audit.json',ROOT/'outputs/local/c68-reserved-fit-boundary-20261112/boundary-freeze.json',OUT/'fit-run-freeze.json',OUT/'cpu-model.npz',OUT/'gpu-model.npz',DATA/'feature-names.json',
        ROOT/'scripts/legacy_formate_compat.py',ROOT/'outputs/local/c45-complete-parent-validation-20261112/frozen-p16-code/deployment.json']
    write(OUT/'complete-candidate-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),queries_fitted=780,acquisitions_fitted=int(q.loc[q.fitting_eligible,'acquisitions'].sum()),candidate_rows_fitted=18833,all_USED_evaluation_queries=869,all_evaluation_acquisitions=3452,all_candidate_rows=21719,features=101,global_C13_reserved_chemistry_fit_overlap=0,
        CPU_gradient_maximum=grad,GPU_device=torch.cuda.get_device_name(0),GPU_epochs=gpu['epochs'],GPU_peak_memory_allocated=int(torch.cuda.max_memory_allocated()),
        parent='Exact frozen P01 core + P16external overlay with neutral-mass consistency; unmodified public parent/encoder historical exposure remains',
        ranking='CPU/GPU pairwise direct-ranking models reorder all same-formula slots; preserve candidate membership and formula slots; missing evidence keeps parent group',
        CPU_model_sha256=sha(OUT/'cpu-model.npz'),GPU_model_sha256=sha(OUT/'gpu-model.npz'),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        GPU_development_negative_source_mode_intersections={k:v for k,v in read(DEV/'independent-oof-audit.json')['source_mode_intersections'].items() if not v['nonnegative']},
        development_gate='Predeclared whole cohort/source/mode marginal guards passed; report source-mode intersections separately without calling them all nonnegative',
        all869_are_USED=True,new_source_quality_read=False,C13_read=False,scoring_entries=0,maximum_new_scored_entries=2,qualified_for_scoring=False,
        limitation='Development-surviving pair only. Requires new untouched target source panel, independent full production audits, single unused C13confirmation and quota/terminal-score safeguards before scoring.'))
    print('C71 STRICT CLEAN FINAL CPU/GPU PAIR FROZEN; prospective/source/production/C13 gates still held',flush=True)


if __name__=='__main__':main()
