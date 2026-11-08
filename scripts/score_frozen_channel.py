"""P14: add a verified frozen-feature head to P01 final-list scores only."""
import gc,hashlib,json,sys
from pathlib import Path
import torch  # Load CUDA DLLs before the legacy encoder environment's NumPy/Arrow.
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ALLOWED={'[M+H]+','[M+Na]+','[M+K]+','[M+NH4]+','[M-H2O+H]+','[M-2H2O+H]+'}


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024*1024):value.update(block)
    return value.hexdigest()


def preprocess(row):
    if str(row.ionization_mode).lower()!='positive' or row.adduct not in ALLOWED or not 0<float(row.precursor_mz)<=1000:return None
    mz=np.asarray(row.ms2_mzs,dtype=np.float64);it=np.asarray(row.ms2_normalized_intensities,dtype=np.float64)
    if mz.ndim!=1 or it.ndim!=1 or len(mz)!=len(it):return None
    valid=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(mz<=1000)&(it>0);mz,it=mz[valid],it[valid]
    if len(mz)<3:return None
    # NumPy1/2 quicksort disagree at equal-intensity cutoffs. Fix the tie order.
    indexes=np.sort(np.argsort(it,kind='stable')[-100:]);peaks=np.stack([mz[indexes],it[indexes]],axis=1)
    peaks=np.pad(peaks,((0,100-len(peaks)),(0,0)));peaks[:,1]/=peaks[:,1].max()
    return np.vstack([[float(row.precursor_mz),1.1],peaks]).astype(np.float32)


def fuse(base,new,weight):
    base,new=np.asarray(base,dtype=np.float64),np.asarray(new,dtype=np.float64)
    assert np.isfinite(base).all() and np.isfinite(new).all()
    old_scale=max(float(base.std()),1e-6)
    new_z=(new-new.mean())/max(float(new.std()),1e-6)
    return (1-weight)*base+weight*(new_z*old_scale+base.mean())


@torch.inference_mode()
def score(bundle,legacy_bundle,baseline,queries,output,device='cuda'):
    torch.set_num_threads(2);bundle=Path(bundle);output=Path(output)
    manifest=json.loads((bundle/'frozen-manifest.json').read_text())
    for name,digest in manifest['files'].items():assert sha(bundle/name)==digest,name
    sys.path.insert(0,str(bundle));from frozen_model import ProjectionRanker
    import importlib.util
    spec=importlib.util.spec_from_file_location('_p14_legacy_features',Path(legacy_bundle)/'features.py')
    features=importlib.util.module_from_spec(spec);spec.loader.exec_module(features)
    state=torch.load(bundle/'head.pt',map_location='cpu',weights_only=True)
    expected=json.loads((bundle/'export_receipt.json').read_text())['head']
    cfg=state['cfg'];assert state['seed']==manifest['selected_seed'] and state['step']==expected['best_step']
    model=ProjectionRanker(cfg['hidden'],cfg['dimension'],cfg['dropout']).to(device).eval();model.load_state_dict(state['state_dict'])
    frame=pd.read_parquet(baseline);frame['molecule_id']=frame.molecule_id.astype(str)
    frame['frozen_score']=0.;frame['frozen_ok']=False
    q=pd.read_parquet(queries);q['molecule_id']=q.molecule_id.astype(str)
    records={};coverage=[];errors=[]
    processed=set(frame.loc[frame.processed.astype(bool),'molecule_id'])
    for mid,g in q.groupby('molecule_id',sort=False):
        if mid not in processed:continue
        values=[]
        for row in g.itertuples(index=False):
            try:
                peaks=preprocess(row)
                if peaks is None:continue
                meta,adduct=features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev)
                values.append((peaks,meta,adduct))
            except (ValueError,TypeError) as error:errors.append(dict(molecule_id=mid,stage='spectrum intake',error=str(error)))
        if values:records[mid]=values
        coverage.append(dict(molecule_id=mid,total_spectra=len(g),supported_spectra=len(values)))
    catalogue=pd.read_parquet(bundle/'molecules.parquet');known=dict(zip(catalogue.key,range(len(catalogue))))
    cached=np.load(bundle/'molecule_embeddings.npy',mmap_mode='r');vectors={};canonical={}
    for smiles in frame.loc[frame.molecule_id.isin(records)&frame.neural_ok.astype(bool),'smiles'].astype(str).unique():
        try:
            key,normal,*_=features.structure(smiles)
            if key in known:vectors[smiles]=np.array(cached[known[key]],dtype=np.float32)
            else:canonical[smiles]=normal
        except (ValueError,TypeError) as error:errors.append(dict(smiles=smiles,stage='structure intake',error=str(error)))
    unknown=list(canonical)
    if unknown:
        tokenizer=Tokenizer.from_file(str(bundle/'tokenizer.json'));tokenizer.enable_truncation(max_length=202)
        tokenizer.enable_padding(length=202,pad_id=2,pad_token='<pad>')
        encoder=torch.jit.load(str(bundle/'molformer.ts'),map_location=device).eval()
        for start in range(0,len(unknown),8):
            names=unknown[start:start+8];tokens=tokenizer.encode_batch([canonical[n] for n in names])
            ids=torch.tensor([v.ids for v in tokens],device=device);mask=torch.tensor([v.attention_mask for v in tokens],device=device)
            values=encoder(ids,mask).float().cpu().numpy().astype(np.float16).astype(np.float32)
            assert np.isfinite(values).all()
            vectors.update(zip(names,values))
        del encoder;gc.collect();torch.cuda.empty_cache()
    encoder=torch.jit.load(str(bundle/'dreams.ts'),map_location=device).eval()
    for mid,values in records.items():
        indexes=frame.index[(frame.molecule_id==mid)&frame.neural_ok.astype(bool)&frame.smiles.astype(str).isin(vectors)]
        if not len(indexes):continue
        embeddings=[]
        for start in range(0,len(values),8):
            peaks=torch.tensor(np.stack([v[0] for v in values[start:start+8]]),device=device)
            embeddings.append(encoder(peaks).float().cpu().numpy().astype(np.float16).astype(np.float32))
        embeddings=torch.tensor(np.concatenate(embeddings),device=device)
        meta=torch.tensor(np.stack([v[1] for v in values]),device=device);adduct=torch.tensor([v[2] for v in values],device=device)
        query=model.encode_spectra(embeddings,meta,adduct).mean(0)
        molecules=torch.tensor(np.stack([vectors[str(s)] for s in frame.loc[indexes,'smiles']]),device=device)
        score=(model.encode_molecules(molecules)@query/model.log_temperature.exp().clamp(.01,1)).cpu().numpy()
        assert np.isfinite(score).all()
        frame.loc[indexes,'frozen_score']=score;frame.loc[indexes,'frozen_ok']=True
        # Entire groups are fused together only when every candidate can be scored.
        group=frame.index[frame.molecule_id==mid]
        if frame.loc[group,'frozen_ok'].all() and frame.loc[group,'neural_ok'].all():
            frame.loc[group,'neural_score']=fuse(frame.loc[group,'neural_score'],frame.loc[group,'frozen_score'],manifest['fusion_weight'])
        else:frame.loc[group,'frozen_ok']=False
    assert np.isfinite(frame.neural_score).all()
    frame.to_parquet(output,index=False)
    Path(str(output)+'.errors.json').write_text(json.dumps(errors,indent=2))
    receipt=dict(run_id=manifest['run_id'],selected_seed=manifest['selected_seed'],weight=manifest['fusion_weight'],
                 numpy_version=np.__version__,preprocess_sort='stable_intensity_original_index',
                 manifest_sha256=sha(bundle/'frozen-manifest.json'),head_sha256=manifest['files']['head.pt'],
                 scored_molecules=int(frame.loc[frame.frozen_ok,'molecule_id'].nunique()),known_structures=len(vectors)-len(unknown),new_structures=len(unknown),
                 coverage=coverage,input_errors=len(errors),unsupported_fallback='exact P01 baseline score',scope='Final lists only; P01 outsider scorer unchanged')
    Path(str(output)+'.frozen.json').write_text(json.dumps(receipt,indent=2))
    print('FROZEN CHANNEL SCORED',json.dumps(receipt),flush=True)


if __name__=='__main__':score(*sys.argv[1:])
