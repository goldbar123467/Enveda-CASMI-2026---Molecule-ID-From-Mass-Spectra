"""One fixed CPU-only direct isomer-ranking prototype; all869USED OOF outcomes."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import json
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from evaluate_c45_complete import point
DATA=ROOT/'outputs/local/c66-parent-evidence-oof-20261112'
OUT=ROOT/'outputs/local/c67-gpu-parent-evidence-oof-20261112'
POLICY=ROOT/'configs/c67-gpu-parent-evidence-oof-20261112.json'


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(0)
    assert read(DATA/'independent-oof-audit.json')['CPU_gate_passed']
    assert read(DATA/'cpu-oof-results.json')['CPU_gate_before_GPU_passed']
    assert not (OUT/'gpu-oof-results.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    freeze=read(DATA/'data-freeze.json');policy=read(POLICY)
    assert not policy['can_submit'] and policy['can_fit_GPU']
    for name,pin in freeze['files_sha256'].items():assert sha(ROOT/name)==pin,name
    queries=pd.read_parquet(DATA/'queries.parquet');rows=pd.read_parquet(DATA/'candidates.parquet');x=np.load(DATA/'features.npy').astype(np.float64)
    assert len(queries)==869 and rows.shape[0]==x.shape[0] and set(queries.fold)==set(range(5))
    write(OUT/'gpu-oof-run-freeze.json',dict(started_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),data_freeze_sha256=sha(DATA/'data-freeze.json'),
        selection='Single predeclared prototype; outer labels unavailable to fold fitting/normalization; no outer early stopping',device='cuda:0',CPU_audit_sha256=sha(DATA/'independent-oof-audit.json'),C13_read=False))
    parent=-np.log(rows.parent_rank.to_numpy(float));prediction=np.full(len(rows),np.nan);pairs=[]
    grouped=list(rows.groupby(['molecule_id','formula'],sort=False))
    for _,g in grouped:
        if not g.available.all() or len(g)<=1 or g.target.sum()!=1:continue
        positive=int(g.index[g.target.eq(1)][0]);negatives=g.index[g.target.eq(0)].to_numpy(int)
        for negative in negatives:pairs.append((positive,int(negative),1/len(negatives)))
    pair=np.asarray(pairs,np.float64);assert len(pair)>0;positive=pair[:,0].astype(int);negative=pair[:,1].astype(int);pairweight=pair[:,2]
    fold_audits=[]
    for fold in range(5):
        train=rows.fold.ne(fold).to_numpy();held=~train
        train_query=queries[queries.fold.ne(fold)];held_query=queries[queries.fold.eq(fold)]
        assert not set(train_query.chemistry_group)&set(held_query.chemistry_group)
        selected=train[positive]&train[negative];assert selected.any()
        mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-4);z=np.clip((x-mean)/std,-10,10)
        cpu=np.load(DATA/f'cpu-fold{fold}.npz')
        np.testing.assert_array_equal(mean,cpu['mean']);np.testing.assert_array_equal(std,cpu['std'])
        anchor=parent+z@cpu['weights']
        torch.manual_seed(policy['seed']+fold);torch.cuda.manual_seed_all(policy['seed']+fold)
        net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.GELU(approximate='tanh'),torch.nn.Linear(32,1)).to('cuda:0')
        torch.nn.init.zeros_(net[2].weight);torch.nn.init.zeros_(net[2].bias)
        zz=torch.tensor(z,dtype=torch.float32,device='cuda:0');aa=torch.tensor(anchor,dtype=torch.float32,device='cuda:0')
        pos=torch.tensor(positive[selected],device='cuda:0');neg=torch.tensor(negative[selected],device='cuda:0')
        weights=torch.tensor(pairweight[selected],dtype=torch.float32,device='cuda:0');weights/=weights.sum()
        ix=torch.tensor(np.flatnonzero(train),device='cuda:0')
        opt=torch.optim.AdamW(net.parameters(),lr=policy['gpu']['learning_rate'],weight_decay=policy['gpu']['weight_decay'])
        for epoch in range(policy['gpu']['epochs']):
            opt.zero_grad();correction=.25*torch.tanh(net(zz).flatten())
            margin=aa[pos]-aa[neg]+correction[pos]-correction[neg]
            loss=(torch.nn.functional.softplus(-margin)*weights).sum()+policy['gpu']['correction_MSE_penalty']*correction[ix].square().mean()
            loss.backward();opt.step()
        net.eval()
        with torch.no_grad():correction=(.25*torch.tanh(net(zz).flatten())).cpu().numpy().astype(np.float64)
        assert np.isfinite(correction).all() and np.abs(correction).max()<=.25
        prediction[held]=anchor[held]+correction[held]
        np.savez(OUT/f'gpu-fold{fold}.npz',weights=cpu['weights'],mean=mean,std=std,
            hidden_weight=net[0].weight.detach().cpu().numpy(),hidden_bias=net[0].bias.detach().cpu().numpy(),
            output_weight=net[2].weight.detach().cpu().numpy(),output_bias=net[2].bias.detach().cpu().numpy())
        item=dict(fold=fold,training_queries=len(train_query),held_queries=len(held_query),training_pairs=int(selected.sum()),
            fitted_chemistry_overlap=0,model_sha256=sha(OUT/f'gpu-fold{fold}.npz'),training_objective=float(loss.detach()),epochs=policy['gpu']['epochs'],
            maximum_correction=float(np.abs(correction).max()),GPU_peak_memory_allocated=int(torch.cuda.max_memory_allocated()))
        fold_audits.append(item);print('C67 GPU OOF FOLD',json.dumps(item),flush=True)
    assert np.isfinite(prediction).all() and torch.cuda.is_initialized()
    rows['oof_score']=prediction;rows.to_parquet(OUT/'gpu-oof-scores.parquet',index=False)
    output={mid:g.sort_values('parent_rank').smiles.tolist() for mid,g in rows.groupby('molecule_id',sort=False)}
    changes=leaders=0
    for (mid,formula),g in grouped:
        if len(g)<=1 or not g.available.all():continue
        positions=(g.parent_rank-1).to_numpy(int);order=sorted(g.index,key=lambda i:(-prediction[i],int(rows.parent_rank.iloc[i])))
        strings=rows.smiles.iloc[order].tolist()
        for index,text in zip(positions,strings):output[mid][index]=text
    metric=module('c67_oof_identity','c09e_features.py').structure;identities=dict(zip(rows.smiles,rows.key));formulas=dict(zip(rows.key,rows.formula));ranks={n:[] for n in ['P01','P16','physical_parent','cpu']}
    for n in ['P01','P16','physical_parent','cpu']:
        for q in queries.itertuples(index=False):
            strings=output[q.molecule_id] if n=='cpu' else getattr(q,n).split(';');keys=[]
            for text in strings:
                if text not in identities:identities[text]=metric(text)[0]
                keys.append(identities[text])
            assert len(set(keys))==len(keys)
            ranks[n].append(keys.index(q.key)+1 if q.key in keys else 0)
            if n=='cpu':
                original=q.physical_parent.split(';');original_keys=[identities[s] for s in original]
                assert set(keys)==set(original_keys) and [formulas[k] for k in keys]==[formulas[k] for k in original_keys]
                changes+=strings!=original;leaders+=keys[0]!=original_keys[0]
    ranks={n:np.asarray(a,np.int16) for n,a in ranks.items()};np.savez(OUT/'gpu-oof-ranks.npz',**ranks)
    pd.DataFrame(dict(molecule_id=queries.molecule_id,smiles=[';'.join(output[mid]) for mid in queries.molecule_id])).to_csv(OUT/'gpu-oof.csv',index=False)
    old=pd.read_csv(ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv',dtype=str).set_index('molecule_id')
    original192=queries.molecule_id.isin(old.index[old.partition.eq('development')]).to_numpy();assert original192.sum()==192
    masks={'all869':np.ones(869,bool),'original192':original192,'original352':queries.cohort.eq('C39').to_numpy()}
    for cohort in queries.cohort.unique():
        selected=queries.cohort.eq(cohort);masks[cohort]=selected.to_numpy()
        for field in ['source','mode']:
            for value in queries.loc[selected,field].unique():masks[cohort+'/'+field+'/'+value]=(selected&queries[field].eq(value)).to_numpy()
    comparisons={}
    for baseline in ['P01','P16','physical_parent']:
        comparisons[baseline]={name:point(ranks[baseline],ranks['cpu'],mask) for name,mask in masks.items()}
    passed=all(comparisons[b][name]['strictly_positive'] for b in ['P01','P16'] for name in ['all869','original192','original352'])
    passed=passed and all(entry['nonnegative'] for b in ['P01','P16'] for entry in comparisons[b].values())
    record=dict(completed_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),data_freeze_sha256=sha(DATA/'data-freeze.json'),
        queries=869,acquisitions=3452,folds=fold_audits,all_queries_retained=True,absent_truths_zero=True,OOF_query_chemistry_overlap=0,
        comparisons=comparisons,changed_lists=changes,rank1_changes=leaders,candidate_membership_changes=0,formula_slot_changes=0,
        GPU_development_gate_passed=passed,device=torch.cuda.get_device_name(0),CUDA_initialized=True,C13_read=False,scoring_entries=0,prospective_qualification=False,
        limitation='All869are adaptive USED development, including the rejectedC53source. Query-chemistry OOF is a development check, not new source confirmation or leaderboard calibration. No prospective qualification or scored entry is authorized by this USED development result alone.')
    write(OUT/'gpu-oof-results.json',record);print('C67 GPU OOF RESULT',json.dumps(dict(gate=passed,all_vs_P01=comparisons['P01']['all869'],all_vs_P16=comparisons['P16']['all869'],changed_lists=changes,rank1_changes=leaders)),flush=True)


if __name__=='__main__':main()
