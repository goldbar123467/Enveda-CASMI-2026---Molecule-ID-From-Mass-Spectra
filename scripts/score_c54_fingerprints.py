"""Frozen DreaMS-to-structure-fingerprint inference; no outcome labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import numpy as np
import pandas as pd
import argparse,gc,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from c54_gpu_model import BoundedResidualHead
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
def score(heads,encoders,baseline,queries,output,device='cuda',only='both',reuse_query_cache=None):
    torch.set_num_threads(2);heads,encoders,output=map(Path,[heads,encoders,output]);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'score-receipt.json').exists();manifest=json.loads((heads/'head-manifest.json').read_text())
    for name,pin in manifest['files'].items():assert sha(heads/name)==pin,name
    assert sha(encoders/'frozen-manifest.json')==manifest['encoder_bundle_manifest_sha256']
    for name,pin in manifest['encoders'].items():assert sha(encoders/name)==pin,name
    sys.path.insert(0,str(heads));import features
    parent=pd.read_csv(baseline,dtype=str,keep_default_na=False);q=pd.read_parquet(queries);q.molecule_id=q.molecule_id.astype(str)
    mids=parent.molecule_id.tolist();assert len(set(mids))==len(parent) and set(mids)==set(q.molecule_id)
    cache=None;cached_structures={};cache_origin=None
    if reuse_query_cache is not None:
        cache=Path(reuse_query_cache);prior=json.loads((cache/'score-receipt.json').read_text())
        assert prior['source_sha256'] in {'60707e1a7406bbe7884cae81eaa3e981214982fc63fdb66b0ade0c490d91c652','040cf0a124709de562cc394b90b452d499362af62715c4c6e39e528f6afd86e7'}
        assert prior['query_sha256']==sha(queries) and not prior['quality_labels_read']
        assert sha(heads/'features.py')=='a97ad9b3ab3711517864d56e497e1672cb29bb05683c72b52db9f9b0a17b1455'
        assert prior['preprocessing']==manifest['spectrum_preprocessing']
        for name,pin in prior['files_sha256'].items():assert sha(cache/name)==pin
        cached_structures={r.smiles:dict(key=r.key,canonical=r.canonical,formula=r.formula) for r in pd.read_parquet(cache/'candidate-identities.parquet').itertuples(index=False)}
        cache_origin=dict(score_receipt_sha256=sha(cache/'score-receipt.json'),query_features_sha256=sha(cache/'query-features.npz'),
            kind='Only original frozen raw DreaMS/acquisition features and deterministic structure identities reused; no old fitted projection or ranking reused.')
        write(output/'query-cache-origin.json',cache_origin)
    structures={};all_smiles=list(dict.fromkeys(s for text in parent.smiles for s in text.split(';')))
    for s in all_smiles:
        if s in cached_structures:structures[s]=cached_structures[s]
        else:
            key,canonical,formula,*_=features.structure(s);structures[s]=dict(key=key,canonical=canonical,formula=formula)
    for text in parent.smiles:
        keys=[structures[s]['key'] for s in text.split(';')];assert 1<=len(keys)<=25 and len(set(keys))==len(keys)
    if cache is None:
        x=np.zeros((len(mids),1039),np.float32);supported=np.zeros(len(mids),bool);coverage=[]
        model=torch.jit.load(str(encoders/'dreams.ts'),map_location=device).eval();grouped=dict(tuple(q.groupby('molecule_id',sort=False)))
        for i,mid in enumerate(mids):
            g=grouped[mid];values=[];meta=[];adduct=[]
            for row in g.itertuples(index=False):
                p=preprocess(row)
                if p is None:continue
                m,a=features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev);values.append(p);meta.append(m);adduct.append(a)
            if values:
                encoded=[]
                for start in range(0,len(values),8):encoded.append(model(torch.tensor(np.stack(values[start:start+8]),device=device)).float().cpu().numpy().astype(np.float16).astype(np.float32))
                x[i,:1024]=np.concatenate(encoded).mean(0);x[i,1024:1028]=np.asarray(meta).mean(0);x[i,1028:]=np.bincount(adduct,minlength=11)/len(adduct);supported[i]=True
            coverage.append(dict(molecule_id=mid,acquisitions=len(g),supported_acquisitions=len(values),supported=bool(supported[i])))
            if i%32==0:print('C54 QUERY FEATURES',i+1,'/',len(mids),flush=True)
        del model;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
        coverage=pd.DataFrame(coverage)
    else:
        raw=np.load(cache/'query-features.npz',allow_pickle=False);x=raw['features'];supported=raw['supported'];coverage=pd.read_csv(cache/'query-coverage.csv')
    assert coverage.molecule_id.tolist()==mids and x.shape==(len(mids),1039) and np.isfinite(x).all()
    np.testing.assert_array_equal(coverage.supported.to_numpy(bool),supported)
    np.savez(output/'query-features.npz',features=x,supported=supported);coverage.to_csv(output/'query-coverage.csv',index=False)
    scale=np.load(heads/'fit-scaling.npz');assert scale['molecule_center'].shape==(2048,)
    z=np.clip((x-scale['input_mean'])/scale['input_std'],-5,5)
    canonical=list(dict.fromkeys(v['canonical'] for v in structures.values()));loc={s:i for i,s in enumerate(canonical)}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);bits=np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in canonical]).astype(np.float32)
    y=normalize_rows(bits-scale['molecule_center']);trials=['cpu','gpu'] if only=='both' else [only];audits={};raw=[]
    for trial in trials:
        if trial=='cpu':query_vectors=normalize_rows(ridge_predict(z,np.load(heads/'cpu-model.npz')['weights']));molecules=y
        else:
            state=torch.load(heads/'gpu-model.pt',map_location='cpu',weights_only=True);model=BoundedResidualHead(state['cfg']).to(device).eval();model.load_state_dict(state['state_dict'])
            query_vectors=model.encode_spectra(torch.tensor(z,device=device)).cpu().numpy();molecules=model.encode_molecules(torch.tensor(y,device=device)).cpu().numpy();del model
        rows=[];changed=0
        for i,row in enumerate(parent.itertuples(index=False)):
            strings=row.smiles.split(';');formulas=[structures[s]['formula'] for s in strings];values=[float(molecules[loc[structures[s]['canonical']]]@query_vectors[i]) for s in strings]
            chosen=order(formulas,values,[True]*len(strings),bool(supported[i]));changed+=chosen!=list(range(len(strings)))
            rows.append(dict(molecule_id=row.molecule_id,smiles=';'.join(strings[j] for j in chosen)))
            for j,s in enumerate(strings):raw.append(dict(trial=trial,molecule_id=row.molecule_id,parent_rank=j+1,smiles=s,score=values[j],eligible=True))
        path=output/(trial+'.csv');pd.DataFrame(rows).to_csv(path,index=False)
        audits[trial]=dict(protected_parent_prefix=1,changed_lists=int(changed),candidate_membership_changes=0,rank1_changes=0,formula_position_changes=0,sha256=sha(path))
    pd.DataFrame(raw).to_parquet(output/'candidate-scores.parquet',index=False)
    pd.DataFrame([dict(smiles=s,**v,eligible=True) for s,v in structures.items()]).to_parquet(output/'candidate-identities.parquet',index=False)
    receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),head_manifest_sha256=sha(heads/'head-manifest.json'),
        baseline_sha256=sha(baseline),query_sha256=sha(queries),queries=len(parent),acquisitions=len(q),supported_queries=int(supported.sum()),candidates=len(structures),
        trials=audits,quality_labels_read=False,feature_cache_used=cache is not None,feature_cache_origin=cache_origin,
        files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},preprocessing=manifest['spectrum_preprocessing'],target='Training-centered unit Morgan2-2048')
    write(output/'score-receipt.json',receipt);print('C54 SCORING COMPLETE',dict(queries=len(parent),trials=audits),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['heads','encoders','baseline','queries','output']:p.add_argument(name)
    p.add_argument('--only',choices=['cpu','gpu','both'],default='both');p.add_argument('--device',default='cuda');p.add_argument('--reuse-query-cache');a=p.parse_args()
    score(a.heads,a.encoders,a.baseline,a.queries,a.output,a.device,a.only,a.reuse_query_cache)
