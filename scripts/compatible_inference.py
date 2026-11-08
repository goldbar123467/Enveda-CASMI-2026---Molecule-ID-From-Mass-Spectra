"""Pinned C08 scoring on the COMPLETE P01 answer lists; no candidate generation.

Train/serve peak precision and stable ordering are identical. Two same-arm fits
are standardized separately per query, then averaged; no weight averaging.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,sys,types
from compatible_tail import apply


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while value:=stream.read(4*1024*1024):h.update(value)
    return h.hexdigest()


def load_modules(bundle):
    name='_casmi_c08';package=types.ModuleType(name);package.__path__=[str(bundle)];sys.modules[name]=package
    for module in ['model','features']:
        spec=importlib.util.spec_from_file_location(name+'.'+module,bundle/(module+'.py'))
        value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
    return sys.modules[name+'.model'].SpectrumCandidateModel,sys.modules[name+'.features']


def normalize(values):
    values=np.asarray(values,dtype=np.float64)
    return (values-values.mean())/max(float(values.std()),1e-6)


@torch.inference_mode()
def score(bundle,arm,baseline,queries,output,device='cuda',smoke=False,shared_fps=None):
    bundle=Path(bundle);manifest=json.loads((bundle/'c08-manifest.json').read_text())
    expected_steps=200 if smoke else 40000
    assert manifest['kind']=='compatible_scorer' and manifest['steps']==expected_steps and manifest['smoke']==smoke
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':assert torch.cuda.is_available()
    records=[m for m in manifest['models'] if m['arm']==arm];assert [m['fold'] for m in records]==[0,1]
    files=set(['model.py','features.py','fingerprint-cache-smiles.parquet','fingerprint-cache.npy'])|{m['file'] for m in records}
    for file in files:assert digest(bundle/file)==manifest['files'][file],file
    model_class,features=load_modules(bundle)
    cache_smiles=pd.read_parquet(bundle/'fingerprint-cache-smiles.parquet').smiles.astype(str)
    assert cache_smiles.is_unique
    cached=np.load(bundle/'fingerprint-cache.npy',mmap_mode='r');assert cached.shape==(len(cache_smiles),256)
    lookup=dict(zip(cache_smiles,range(len(cache_smiles))));fps={} if shared_fps is None else shared_fps
    frame=pd.read_csv(baseline,dtype=str,keep_default_na=False) if not isinstance(baseline,pd.DataFrame) else baseline.copy()
    q=pd.read_parquet(queries) if not isinstance(queries,pd.DataFrame) else queries.copy();q.molecule_id=q.molecule_id.astype(str)
    query={str(mid):g for mid,g in q.groupby('molecule_id',sort=False)}
    prepared={};parts=[];supported=set();unsupported=[]
    cfg_expected=manifest['architecture'];precision=records[0]['cache_precision'];assert all(m['cache_precision']==precision for m in records)
    for row in frame.itertuples(index=False):
        mid=str(row.molecule_id);candidates=row.smiles.split(';')
        group=pd.DataFrame(dict(molecule_id=mid,smiles=candidates,baseline_rank=np.arange(1,len(candidates)+1),c08_ok=False,c08_score=0.))
        parts.append(group)
        if mid not in query:continue
        spectra=[]
        for spectrum in query[mid].itertuples(index=False):
            try:
                x,n=features.peaks(spectrum.ms2_mzs,spectrum.ms2_normalized_intensities,spectrum.precursor_mz,96)
                if n<3:continue
                metadata,adduct=features.metadata(spectrum.precursor_mz,spectrum.adduct,spectrum.ionization_mode,spectrum.collision_energy_ev)
                assert np.isfinite(x).all() and np.isfinite(metadata).all()
                if precision=='float16':x=x.astype(np.float16).astype(np.float32)
                spectra.append((x,n,metadata,adduct))
            except (ValueError,TypeError):continue
        if not spectra:unsupported.append(mid);continue
        candidate_fp=[]
        for smiles in candidates:
            if smiles not in fps:
                if smiles in lookup:fps[smiles]=np.unpackbits(np.asarray(cached[lookup[smiles]]),bitorder='big')
                else:fps[smiles]=features.structure(smiles,2048)[-1]
            candidate_fp.append(fps[smiles])
        prepared[mid]=(spectra,np.stack(candidate_fp));supported.add(mid)
    result=pd.concat(parts,ignore_index=True)
    for entry in records:
        checkpoint=torch.load(bundle/entry['file'],map_location='cpu',weights_only=True)
        assert checkpoint['steps']==expected_steps and checkpoint['smoke']==smoke and checkpoint['fold']==entry['fold']
        cfg=checkpoint['cfg'];assert all(cfg[k]==v for k,v in cfg_expected.items())
        assert cfg['cache_precision']==precision and cfg['name']==arm and cfg['max_peaks']==96
        model=model_class(cfg).to(device).eval();model.load_state_dict(checkpoint['state_dict']);del checkpoint
        column='c08_raw_fold'+str(entry['fold']);result[column]=0.
        for mid,indices in result.groupby('molecule_id',sort=False).groups.items():
            if mid not in prepared:continue
            spectra,fp=prepared[mid];candidate=torch.as_tensor(fp,device=device,dtype=torch.float32)
            molecule=model.encode_molecules(candidate);zsum=psum=None;nseen=0
            for start in range(0,len(spectra),32):
                batch=spectra[start:start+32]
                x=torch.as_tensor(np.stack([s[0] for s in batch]),device=device)
                lengths=torch.as_tensor([s[1] for s in batch],device=device,dtype=torch.long)
                metadata=torch.as_tensor(np.stack([s[2] for s in batch]),device=device)
                adduct=torch.as_tensor([s[3] for s in batch],device=device,dtype=torch.long)
                z,p=model.encode_spectra(x,lengths,metadata,adduct)
                zsum=z.sum(0) if zsum is None else zsum+z.sum(0);psum=p.sum(0) if psum is None else psum+p.sum(0);nseen+=len(batch)
            values=((molecule@(zsum/nseen))/cfg['temperature']+cfg['fp_score_weight']*(candidate@(psum/nseen))/cfg['fp_bits']).cpu().numpy()
            assert np.isfinite(values).all(),mid
            result.loc[indices,column]=values
            result.loc[indices,'c08_score']+=.5*normalize(values)
            result.loc[indices,'c08_ok']=True
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        print('C08 FOLD SCORED',arm,entry['fold'],len(supported),flush=True)
    result.to_parquet(output,index=False)
    receipt=dict(arm=arm,models=2,folds=[0,1],cache_precision=precision,smoke=smoke,steps=expected_steps,manifest_sha256=digest(bundle/'c08-manifest.json'),
        baseline_content_sha256=hashlib.sha256(frame.to_csv(index=False,lineterminator='\n').encode()).hexdigest(),
        queries_sha256=digest(queries) if not isinstance(queries,pd.DataFrame) else None,
        scored_molecules=len(supported),unsupported_molecules=unsupported,unprocessed_molecules=len(frame)-len(query),
        input_errors=0,peak_selection='stable_intensity_then_mz',spectrum_aggregation='all usable spectra, equal average',
        model_aggregation='mean of per-query standardized raw scores',numpy_version=np.__version__,torch_version=torch.__version__)
    Path(str(output)+'.json').write_text(json.dumps(receipt,indent=2));return result,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--arm',required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--queries',type=Path,required=True)
    parser.add_argument('--scores',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--weight',type=float,default=.1);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    values,receipt=score(args.bundle,args.arm,args.baseline,args.queries,args.scores,args.device)
    if args.output:
        original=pd.read_csv(args.baseline,dtype=str,keep_default_na=False);final,changes=apply(original,values,args.weight)
        final.to_csv(args.output,index=False);receipt.update(changes,csv_sha256=digest(args.output))
        Path(str(args.output)+'.c08.json').write_text(json.dumps(receipt,indent=2))
    print('DONE C08 SCORING',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
