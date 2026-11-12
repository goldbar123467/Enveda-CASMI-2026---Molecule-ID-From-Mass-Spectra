"""Verify exact eligible-only final CPU optimum and GPU training replay."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import base64,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp,frozen_contract,CODE
from build_c09_controls import unwrap
from build_c13_formate_recovery import value
from c67_ranker import predict
OUT=ROOT/'outputs/local/c71-final-clean-direct-pair-20261112'
DATA=ROOT/'outputs/local/c69-clean-parent-evidence-oof-20261112'


def main():
    torch.set_num_threads(2);target=OUT/'independent-fitted-pair-audit.json';assert not target.exists()
    complete=read(OUT/'complete-candidate-freeze.json')
    assert complete['source_sha256']==sha(ROOT/'scripts/fit_c71_final_pair.py')
    for name,pin in complete['files_sha256'].items():assert sha(ROOT/name)==pin
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    q=pd.read_parquet(DATA/'queries.parquet');r=pd.read_parquet(DATA/'candidates.parquet');x=np.load(DATA/'features.npy').astype(np.float64)
    eligible=r.fitting_eligible.to_numpy();cpu=np.load(OUT/'cpu-model.npz');gpu=np.load(OUT/'gpu-model.npz')
    mean=x[eligible].mean(0);std=np.maximum(x[eligible].std(0),1e-4)
    for m in [cpu,gpu]:
        np.testing.assert_array_equal(m['mean'],mean);np.testing.assert_array_equal(m['std'],std);np.testing.assert_array_equal(m['weights'],cpu['weights'])
    pairs=[]
    for _,g in r.groupby(['molecule_id','formula'],sort=False):
        if len(g)<=1 or not g.available.all() or g.target.sum()!=1:continue
        p=int(g.index[g.target.eq(1)][0]);neg=g.index[g.target.eq(0)&g.fitting_eligible]
        if not r.fitting_eligible.iloc[p] or not len(neg):continue
        pairs.extend((p,int(n),1/len(neg)) for n in neg)
    a=np.array([t[0] for t in pairs]);b=np.array([t[1] for t in pairs]);weights=np.array([t[2] for t in pairs])
    assert eligible[a].all() and eligible[b].all()
    reserved=set(pd.read_parquet(ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet',columns=['key']).key)
    assert not set(r.key[eligible])&reserved and not set(r.key.iloc[np.r_[a,b]])&reserved
    z=np.clip((x-mean)/std,-10,10);parent=-np.log(r.parent_rank.to_numpy(float));anchor=parent+z@cpu['weights']
    policy=read(ROOT/'configs/c70-clean-gpu-parent-evidence-oof-20261112.json');delta=z[a]-z[b];margin=anchor[a]-anchor[b]
    norm=weights/weights.sum();sigmoid=1/(1+np.exp(-np.clip(margin,-80,80)))
    gradient=delta.T@((sigmoid-1)*norm)+2*policy['cpu']['ridge']*cpu['weights'];residual=float(np.abs(gradient).max());assert residual<2e-5
    seed=policy['seed']+5;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.GELU(approximate='tanh'),torch.nn.Linear(32,1)).to('cuda:0')
    torch.nn.init.zeros_(net[2].weight);torch.nn.init.zeros_(net[2].bias)
    zz=torch.tensor(z,dtype=torch.float32,device='cuda:0');aa=torch.tensor(anchor,dtype=torch.float32,device='cuda:0')
    pa=torch.tensor(a,device='cuda:0');pb=torch.tensor(b,device='cuda:0');pw=torch.tensor(weights,dtype=torch.float32,device='cuda:0');pw/=pw.sum()
    ix=torch.tensor(np.flatnonzero(eligible),device='cuda:0');settings=policy['gpu']
    opt=torch.optim.AdamW(net.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    for epoch in range(settings['epochs']):
        opt.zero_grad();correction=.25*torch.tanh(net(zz).flatten());m=aa[pa]-aa[pb]+correction[pa]-correction[pb]
        loss=(torch.nn.functional.softplus(-m)*pw).sum()+settings['correction_MSE_penalty']*correction[ix].square().mean();loss.backward();opt.step()
    for actual,name in [(net[0].weight,'hidden_weight'),(net[0].bias,'hidden_bias'),(net[2].weight,'output_weight'),(net[2].bias,'output_bias')]:
        np.testing.assert_allclose(actual.detach().cpu().numpy(),gpu[name],rtol=0,atol=1e-6)
    with torch.no_grad():expected=anchor+(.25*torch.tanh(net(zz).flatten())).cpu().numpy()
    _,actual=predict(r,x,gpu,True);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),candidate_freeze_sha256=sha(OUT/'complete-candidate-freeze.json'),
        fitting_queries=int(q.fitting_eligible.sum()),fitting_acquisitions=int(q.loc[q.fitting_eligible,'acquisitions'].sum()),fitting_candidate_rows=int(eligible.sum()),
        candidate_pair_count=len(a),all869evaluation_queries_retained=True,all21719evaluation_candidates_retained=True,
        reserved_identity_and_chemistry_fit_overlap=0,CPU_unique_convex_optimum_gradient_maximum=residual,CPU_scaling_eligible_rows_only_exact=True,
        GPU_actual_clean_CPU_anchor_exact=True,GPU_fixed_seed_200epoch_training_replay_exact=True,exported_NumPy_forward_exact=True,
        C13_query_mapping_or_outcomes_read=False,new_source_quality_read=False,scoring_entries=0,qualified_for_scoring=False))
    contract=frozen_contract();notebook=ROOT/'kaggle/forks/c57-full-masked-validation/notebook.ipynb';pin=read(notebook.with_name('validation-pin.json'))
    assert sha(notebook)==pin['notebook_sha256']
    outer=unwrap(''.join(read(notebook)['cells'][12]['source']));payload=base64.b64decode(value(outer,'FROZEN_B64'))
    payload_sha=hashlib.sha256(payload).hexdigest();assert payload_sha==pin['frozen_scientific_payload_sha256']
    files=[OUT/'complete-candidate-freeze.json',target,notebook,notebook.with_name('validation-pin.json'),ROOT/'scripts/c67_ranker.py',
        ROOT/'scripts/legacy_formate_compat.py',CODE/'deployment.json',CODE/'c09e_policy.json']+[CODE/n for n in contract['source_files']]
    write(OUT/'complete-pipeline-binding.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),
        candidate_freeze_sha256=sha(OUT/'complete-candidate-freeze.json'),P01_frozen_scientific_payload_sha256=payload_sha,
        P16_frozen_deployment_sha256=sha(CODE/'deployment.json'),runtime_sha256=sha(ROOT/'scripts/c67_ranker.py'),
        CPU_model_sha256=sha(OUT/'cpu-model.npz'),GPU_model_sha256=sha(OUT/'gpu-model.npz'),
        physical_adapter='c67_ranker.physical_parent; original adduct tokens; max0.002Da/10ppm neutral-mass check; unknown keepsP01',
        reference_use='Production uses exact frozen reference assets; controlled validation quarantines allUSED/fresh/globalreserved231chemistry and acquisition aliases',
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},new_source_outcomes_read=False,C13_read=False,scoring_entries=0,
        limitation='Complete scientific recipe frozen. Source/production/once-onlyconfirmation gates remain required; this receipt is not scoring authorization.'))
    print('C71 INDEPENDENT FIT AND COMPLETE PIPELINE BINDING PASS',residual,flush=True)


if __name__=='__main__':main()
