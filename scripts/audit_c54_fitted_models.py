"""Independent clean-fit/normal-equation/anchor/bound checks, no query quality."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import torch
import numpy as np
import pandas as pd
from datetime import datetime,timezone
from c45_complete_parent import ROOT,read,write,sha
from c54_gpu_model import BoundedResidualHead
OUT=ROOT/'outputs/local/c54-isomer-fingerprint-heads-20261112'


def main():
    torch.set_num_threads(2);target=OUT/'independent-fitted-model-audit.json';assert not target.exists()
    cfg=read(ROOT/'configs/c54-isomer-fingerprint-pair-20261112.json');boundary=read(OUT/'fit-boundary-freeze.json');features=read(OUT/'head-feature-freeze.json')
    for name,pin in {**boundary['files_sha256'],**features['files_sha256']}.items():assert sha(OUT/name)==pin
    ids=np.load(OUT/'fitting-identities.npy',allow_pickle=False);rows=np.load(OUT/'fitting-spectrum-rows.npy',allow_pickle=False)
    oldids=np.load(ROOT/'outputs/local/c48-clean-residual-heads-20261112/fitting-identities.npy',allow_pickle=False);assert set(ids)<=set(oldids)
    cache=ROOT/'outputs/local/long-frozen-20261107-r1';molecules=pd.read_parquet(cache/'molecules.parquet',columns=['key'])
    C39=pd.read_csv(ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv',usecols=['key'])
    auxiliary=pd.read_csv(ROOT/'outputs/local/c48-auxiliary-holdout-20261112/labels.csv',usecols=['key'])
    reserved_inventory=pd.read_parquet(ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet',columns=['key'])
    new_source=pd.read_csv(ROOT/'outputs/local/c53-target-holdout-20261112/labels.csv',usecols=['key'])
    forbidden=set(C39.key)|set(auxiliary.key)|set(new_source.key)|set(reserved_inventory.key)
    assert not set(molecules.key.iloc[ids])&forbidden
    selected=pd.read_parquet(cache/'selected.parquet',columns=['row','mol_idx','valid']);assert selected.valid.iloc[rows].all()
    assert set(selected.mol_idx.iloc[rows])==set(ids)
    raw=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14']);rawkeys=set(raw.inchikey14.iloc[selected.row.to_numpy()[rows]].astype(str))
    blocked=set(read(OUT/'engine2-raw-key-exclusion.json')['keys']);assert not rawkeys&blocked
    witnesses=pd.read_parquet(OUT/'original-structure-identity-witnesses.parquet',columns=['raw_key','official_key','exclude'])
    original_official=set(witnesses.loc[witnesses.raw_key.isin(rawkeys),'official_key'].dropna());assert not original_official&forbidden
    weights=np.load(OUT/'cpu-ridge/model.npz')['weights'];x=np.load(OUT/'fit-features.npy',mmap_mode='r');y=np.load(OUT/'fit-targets.npy',mmap_mode='r')
    assert x.shape==(len(ids),1039) and y.shape==(len(ids),2048) and len(ids)==153401
    # Independent float64 sufficient statistics verify the actual CPU model,
    # including an unpenalized intercept and equal weights per identity.
    gram=np.zeros((1040,1040));rhs=np.zeros((1040,2048));target_norm_error=0.
    for start in range(0,len(ids),8192):
        a=np.asarray(x[start:start+8192],np.float64);b=np.asarray(y[start:start+8192],np.float64)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        target_norm_error=max(target_norm_error,float(np.abs(np.linalg.norm(b,axis=1)-1).max()))
        a=np.column_stack([a,np.ones(len(a))]);gram+=a.T@a;rhs+=a.T@b
    gram/=len(ids);rhs/=len(ids);gram+=np.diag([cfg['cpu']['ridge_l2']]*1039+[0.])
    error=float(np.abs(gram@weights.astype(np.float64)-rhs).max());assert error<1e-5 and target_norm_error<1e-5
    state=torch.load(OUT/'gpu-residual/final_model.pt',map_location='cpu',weights_only=True)
    assert state['cfg']==cfg['gpu'] and state['steps']==8000
    assert torch.equal(state['state_dict']['anchor_weights'],torch.from_numpy(weights))
    model=BoundedResidualHead(state['cfg']).eval();model.load_state_dict(state['state_dict'])
    fingerprint=np.load(ROOT/'outputs/local/n0/cache/fingerprints.npy',mmap_mode='r');center=np.zeros(2048,np.float64)
    for start in range(0,len(ids),8192):center+=np.asarray(fingerprint[ids[start:start+8192]],np.float64).sum(0)
    center/=len(ids);stored=np.load(OUT/'fit-scaling.npz')['molecule_center'];assert np.abs(stored-center).max()<1e-7
    for i in np.random.default_rng(54).choice(len(ids),256,replace=False):
        expected=np.asarray(fingerprint[ids[i]],np.float64)-center;expected/=np.linalg.norm(expected);assert np.abs(expected-y[i]).max()<1e-7
    raw_queries=np.load(ROOT/'outputs/local/c48-clean-residual-heads-20261112/C39-predictions/query-features.npz',allow_pickle=False)['features'];scale=np.load(OUT/'fit-scaling.npz')
    z=np.clip((raw_queries-scale['input_mean'])/scale['input_std'],-5,5)
    with torch.inference_mode():
        anchor,delta=model.components(torch.from_numpy(z));ratios=delta.norm(dim=-1)/anchor.norm(dim=-1).clamp_min(1e-12)
        maximum=float(ratios.max());assert maximum<=.500001
        sample=torch.from_numpy(np.asarray(y[:64]).copy());assert torch.equal(model.encode_molecules(sample),torch.nn.functional.normalize(sample,dim=-1))
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),CPU_model_sha256=sha(OUT/'cpu-ridge/model.npz'),
        GPU_model_sha256=sha(OUT/'gpu-residual/final_model.pt'),fit_boundary_sha256=sha(OUT/'fit-boundary-freeze.json'),
        fitting_identities=len(ids),fitting_acquisitions=len(rows),all_original_exclusions_preserved=True,
        C39_C51_C53_and_global_C13_inventory_fit_identity_overlaps=0,original_row_official_identity_overlaps=0,original_raw_quarantine_survivors=0,
        CPU_normal_equation_max_abs_residual=error,target_unit_norm_max_abs_error=target_norm_error,
        GPU_actual_anchor_exactly_clean_CPU=True,GPU_final_steps=8000,GPU_maximum_observed_relative_residual=maximum,
        molecular_geometry_unchanged=True,all_fitting_fingerprint_center_independently_exact=True,independent_target_witnesses=256,quality_outcomes_read=False,C13_labels_read=False,
        limitation='Actual fitted-component audit; public parent/encoder historical exposure is not removed. C13global inventory is used only for exclusion verification, not reserved query label matching.')
    write(target,result);print('C54 INDEPENDENT FITTED MODEL PASS',__import__('json').dumps(result))


if __name__=='__main__':main()
