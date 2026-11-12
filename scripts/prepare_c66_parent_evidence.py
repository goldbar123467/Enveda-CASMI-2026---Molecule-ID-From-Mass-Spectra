"""Add omitted, already-deployed parent rank/metadata evidence to USED OOF data."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp
from prepare_c64_pairwise_data import PANELS
DATA=ROOT/'outputs/local/c64-pairwise-oof-20261112'
OUT=ROOT/'outputs/local/c66-parent-evidence-oof-20261112'
POLICY=ROOT/'configs/c66-parent-evidence-oof-20261112.json'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'data-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    old=read(DATA/'data-freeze.json')
    for name,pin in old['files_sha256'].items():assert sha(ROOT/name)==pin
    q=pd.read_parquet(DATA/'queries.parquet');rows=pd.read_parquet(DATA/'candidates.parquet');x=np.load(DATA/'features.npy');names=read(DATA/'feature-names.json')
    path=ROOT/'outputs/research/pop-prior/pool_popularity.csv'
    pop=pd.read_csv(path,usecols=['inchikey14','pop','np_coconut','np_lotus','src_natural_product']).drop_duplicates('inchikey14',keep='last').set_index('inchikey14')
    pop['np_flag']=(pop.np_coconut|pop.np_lotus|pop.src_natural_product).astype(int)
    raw_frames=[];inputs=[]
    for _,_,_,_,core,*_ in PANELS:
        p=ROOT/core/'c41_neural_p01.parquet';f=pd.read_parquet(p)
        f['parent_neural_rank']=f.groupby('molecule_id',sort=False).neural_score.rank(method='first',ascending=False)
        raw_frames.append(f);inputs.append(p)
    raw=pd.concat(raw_frames,ignore_index=True).set_index(['molecule_id','metric_key'])
    index=pd.MultiIndex.from_arrays([rows.molecule_id,rows.key]);assert index.isin(raw.index).all()
    br=raw.loc[index,'baseline_rank'].to_numpy(float);nr=raw.loc[index,'parent_neural_rank'].to_numpy(float)
    popularity=rows.key.map(pop['pop']).fillna(0).to_numpy(float);np_flag=rows.key.map(pop['np_flag']).fillna(0).to_numpy(float)
    assert rows.library_max.max()<.9,'Inspect actual frozen library bonus before using its inactive-branch reconstruction'
    score=1/(3+br)+.75/(3+nr)+.1*popularity/20+.02*np_flag
    added=dict(public_baseline_rank=br,parent_neural_rank=nr,public_rank_RRF=1/(3+br),parent_neural_RRF=.75/(3+nr),
        documentation_prior=popularity/20,natural_product_prior=np_flag,parent_tail_score=score)
    columns=[x];new_names=list(names)
    for name,value in added.items():
        z=pd.Series(value).groupby(rows.molecule_id).transform(lambda a:(a-a.mean())/max(float(a.std(ddof=0)),1e-6)).to_numpy()
        columns.extend([value[:,None].astype(np.float32),np.clip(z,-10,10)[:,None].astype(np.float32)]);new_names.extend([name,name+'_query_z'])
    augmented=np.concatenate(columns,axis=1);assert np.isfinite(augmented).all()
    violations=[]
    for mid,g in rows.assign(reconstructed_parent_score=score).groupby('molecule_id',sort=False):
        tail=g.sort_values('parent_rank').iloc[1:];v=tail.reconstructed_parent_score.to_numpy()
        for i in np.flatnonzero(np.diff(v)>1e-9):violations.append(dict(molecule_id=mid,earlier_rank=int(tail.parent_rank.iloc[i]),later_rank=int(tail.parent_rank.iloc[i+1]),gap=float(v[i+1]-v[i])))
    # Existing parent can pin its former leader at rank2. Every other ordering
    # must agree with the receipt-pinned tail formula, without looking at truth.
    unexplained=[v for v in violations if v['earlier_rank']!=2]
    assert not unexplained,('Parent evidence reconstruction mismatch',unexplained[:4])
    for name in ['queries.parquet','candidates.parquet','physical-parent-changes.json']:shutil.copyfile(DATA/name,OUT/name)
    np.save(OUT/'features.npy',augmented);write(OUT/'feature-names.json',new_names);write(OUT/'parent-score-reconstruction.json',dict(queries=869,
        parent_formula='1/(3+baseline_rank)+0.75/(3+neural_rank)+0.1*pop/20+0.02*NP; library bonus inactive in every USED query',
        known_rank2_pin_exceptions=violations,unexplained_tail_inversions=0,labels_used_for_reconstruction=False))
    files=[Path(__file__),POLICY,path,DATA/'data-freeze.json',*inputs]+[OUT/n for n in ['queries.parquet','candidates.parquet','features.npy','feature-names.json','physical-parent-changes.json','parent-score-reconstruction.json']]
    write(OUT/'data-freeze.json',dict(prepared_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),queries=869,acquisitions=3452,candidates=len(rows),
        features=augmented.shape[1],chemistry_groups=old['chemistry_groups'],fold_queries=old['fold_queries'],all_original_queries_retained=True,
        no_C56_fitted_projection_reused=True,C13_read=False,model_fitted=False,scoring_entries=0,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('C66 PARENT EVIDENCE READY',augmented.shape,'rank2pins',len(violations),flush=True)


if __name__=='__main__':main()
