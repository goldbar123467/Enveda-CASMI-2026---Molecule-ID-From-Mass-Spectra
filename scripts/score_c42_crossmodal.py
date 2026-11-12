"""Label-blind C42 final-list scoring, identical locally and on Kaggle.

CPU refers to fitting the ridge head. Both trials use frozen GPU encoders and
the complete P01 parent. Candidate membership, rank1 and formula slots stay fixed.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from pathlib import Path
import argparse,gc,hashlib,json,sys
from datetime import datetime,timezone
from c40_gpu_model import CrossModalHead
from crossmodal_heads import normalize_rows,ridge_predict
from crossmodal_tail import order


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf8')


def preprocess(row):
    allowed={'[M+H]+','[M+Na]+','[M+K]+','[M+NH4]+','[M-H2O+H]+','[M-2H2O+H]+'}
    if str(row.ionization_mode).lower()!='positive' or row.adduct not in allowed or not 0<float(row.precursor_mz)<=1000:return None
    mz=np.asarray(row.ms2_mzs,dtype=np.float64);it=np.asarray(row.ms2_normalized_intensities,dtype=np.float64)
    assert mz.ndim==it.ndim==1 and len(mz)==len(it)
    good=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(mz<=1000)&(it>0)
    mz,it=mz[good],it[good]
    if len(mz)<3:return None
    ix=np.sort(np.argsort(it,kind='stable')[-100:])
    peaks=np.stack([mz[ix],it[ix]],1);peaks=np.pad(peaks,((0,100-len(peaks)),(0,0)))
    peaks[:,1]/=peaks[:,1].max()
    return np.vstack([[float(row.precursor_mz),1.1],peaks]).astype(np.float32)


@torch.inference_mode()
def score(heads,encoders,baseline,queries,output,device='cuda',only='both'):
    torch.set_num_threads(2)
    heads,encoders,output=map(Path,[heads,encoders,output]);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'score-receipt.json').exists(),'Scoring output is immutable'
    manifest=json.loads((heads/'head-manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(heads/n)==h,n
    assert sha(encoders/'frozen-manifest.json')==manifest['encoder_bundle_manifest_sha256']
    encoder_manifest=json.loads((encoders/'frozen-manifest.json').read_text())
    for n in ['molecules.parquet','molecule_embeddings.npy']:assert sha(encoders/n)==encoder_manifest['files'][n],n
    for n,h in manifest['encoders'].items():assert sha(encoders/n)==h,n
    sys.path.insert(0,str(heads));import features
    parent=pd.read_csv(baseline,dtype=str,keep_default_na=False)
    q=pd.read_parquet(queries);q['molecule_id']=q.molecule_id.astype(str)
    assert parent.molecule_id.is_unique and set(parent.molecule_id)==set(q.molecule_id)
    mids=parent.molecule_id.tolist();all_smiles=list(dict.fromkeys(s for x in parent.smiles for s in x.split(';')))
    structures={}
    for s in all_smiles:
        key,canonical,formula,*_=features.structure(s)
        structures[s]=dict(key=key,canonical=canonical,formula=formula)
    for text in parent.smiles:
        keys=[structures[s]['key'] for s in text.split(';')]
        assert 1<=len(keys)<=25 and len(set(keys))==len(keys)
    tokenizer=Tokenizer.from_file(str(encoders/'tokenizer.json'))
    # Count tokens before truncation. Overlong groups must retain parent order.
    tokenizer.no_truncation();tokenizer.no_padding()
    canonical=list(dict.fromkeys(v['canonical'] for v in structures.values()))
    eligibility={s:len(t.ids)<=202 for s,t in zip(canonical,tokenizer.encode_batch(canonical))}
    for v in structures.values():v['eligible']=eligibility[v['canonical']]
    tokenizer.enable_truncation(max_length=202);tokenizer.enable_padding(length=202,pad_id=2,pad_token='<pad>')
    catalogue=pd.read_parquet(encoders/'molecules.parquet',columns=['smiles'])
    known=dict(zip(catalogue.smiles.astype(str),range(len(catalogue))))
    cache=np.load(encoders/'molecule_embeddings.npy',mmap_mode='r')
    vectors={s:np.asarray(cache[known[s]],np.float32) for s in canonical if eligibility[s] and s in known}
    unknown=[s for s in canonical if eligibility[s] and s not in vectors]
    if unknown:
        model=torch.jit.load(str(encoders/'molformer.ts'),map_location=device).eval()
        for start in range(0,len(unknown),8):
            names=unknown[start:start+8];t=tokenizer.encode_batch(names)
            ids=torch.tensor([x.ids for x in t],device=device);mask=torch.tensor([x.attention_mask for x in t],device=device)
            values=model(ids,mask).float().cpu().numpy().astype(np.float16).astype(np.float32)
            assert values.shape==(len(names),768) and np.isfinite(values).all()
            vectors.update(zip(names,values))
        del model;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    x=np.zeros((len(mids),1039),np.float32);supported=np.zeros(len(mids),bool);coverage=[]
    model=torch.jit.load(str(encoders/'dreams.ts'),map_location=device).eval()
    grouped=dict(tuple(q.groupby('molecule_id',sort=False)))
    for i,mid in enumerate(mids):
        g=grouped[mid];values=[];meta=[];adduct=[]
        for row in g.itertuples(index=False):
            p=preprocess(row)
            if p is None:continue
            m,a=features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev)
            values.append(p);meta.append(m);adduct.append(a)
        if values:
            encoded=[]
            for start in range(0,len(values),8):
                t=torch.tensor(np.stack(values[start:start+8]),device=device)
                encoded.append(model(t).float().cpu().numpy().astype(np.float16).astype(np.float32))
            x[i,:1024]=np.concatenate(encoded).mean(0);x[i,1024:1028]=np.asarray(meta).mean(0)
            x[i,1028:]=np.bincount(adduct,minlength=11)/len(adduct);supported[i]=True
        coverage.append(dict(molecule_id=mid,acquisitions=len(g),supported_acquisitions=len(values),supported=bool(values)))
        if i%32==0:print('C42 QUERY FEATURES',i+1,'/',len(mids),flush=True)
    del model;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    assert np.isfinite(x).all()
    np.savez(output/'query-features.npz',features=x,supported=supported)
    pd.DataFrame(coverage).to_csv(output/'query-coverage.csv',index=False)
    scale=np.load(heads/'fit-scaling.npz');z=np.clip((x-scale['input_mean'])/scale['input_std'],-5,5)
    names=[s for s in canonical if s in vectors];loc={s:i for i,s in enumerate(names)}
    y=normalize_rows(np.stack([vectors[s] for s in names])-scale['molecule_center'])
    trials=['cpu','gpu'] if only=='both' else [only];audits={};raw=[]
    for trial in trials:
        if trial=='cpu':
            weights=np.load(heads/'cpu-model.npz')['weights'];queries_encoded=normalize_rows(ridge_predict(z,weights));molecules=y
        else:
            state=torch.load(heads/'gpu-model.pt',map_location='cpu',weights_only=True)
            model=CrossModalHead(state['cfg']).to(device).eval();model.load_state_dict(state['state_dict'])
            queries_encoded=model.encode_spectra(torch.tensor(z,device=device)).cpu().numpy()
            molecules=np.concatenate([model.encode_molecules(torch.tensor(y[j:j+512],device=device)).cpu().numpy() for j in range(0,len(y),512)])
            del model
        rows=[];changed=0
        for i,row in enumerate(parent.itertuples(index=False)):
            ss=row.smiles.split(';');formula=[structures[s]['formula'] for s in ss]
            valid=[structures[s]['eligible'] for s in ss];values=[float(molecules[loc[structures[s]['canonical']]]@queries_encoded[i]) if v else 0. for s,v in zip(ss,valid)]
            chosen=order(formula,values,valid,bool(supported[i]))
            changed+=chosen!=list(range(len(ss)))
            rows.append(dict(molecule_id=row.molecule_id,smiles=';'.join(ss[j] for j in chosen)))
            for j,s in enumerate(ss):raw.append(dict(trial=trial,molecule_id=row.molecule_id,parent_rank=j+1,smiles=s,score=values[j],eligible=valid[j]))
        path=output/(trial+'.csv');pd.DataFrame(rows).to_csv(path,index=False)
        audits[trial]=dict(changed_lists=int(changed),candidate_membership_changes=0,rank1_changes=0,formula_position_changes=0,sha256=sha(path))
    pd.DataFrame(raw).to_parquet(output/'candidate-scores.parquet',index=False)
    pd.DataFrame([dict(smiles=s,**v) for s,v in structures.items()]).to_parquet(output/'candidate-identities.parquet',index=False)
    receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),head_manifest_sha256=sha(heads/'head-manifest.json'),
        baseline_sha256=sha(baseline),query_sha256=sha(queries),queries=len(parent),acquisitions=len(q),supported_queries=int(supported.sum()),
        candidates=len(structures),exact_canonical_cached=len(vectors)-len(unknown),fresh_canonical_encodings=len(unknown),
        overlong_canonical_structures=sum(not v for v in eligibility.values()),trials=audits,quality_labels_read=False,
        files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},preprocessing=manifest['spectrum_preprocessing'])
    write(output/'score-receipt.json',receipt);print('DONE C42 SCORES',json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('heads');p.add_argument('encoders');p.add_argument('baseline');p.add_argument('queries');p.add_argument('output')
    p.add_argument('--device',default='cuda');p.add_argument('--only',choices=['cpu','gpu','both'],default='both');a=p.parse_args()
    score(a.heads,a.encoders,a.baseline,a.queries,a.output,a.device,a.only)
