"""Identical CPU preprocessing/prediction probe, using only USED public candidates."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['PYTHONHASHSEED']='0'
import torch
torch.set_num_threads(2)
import argparse,hashlib,importlib.util,json,platform,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

JOBS = [{'smiles': 'CC1=CC(=NC(=N1)N2CC[CH]([CH](CC2)O)N(C)C3CCOCC3)C', 'adduct': '[M+H]+', 'instrument': 'QTOF', 'ce': 40.0}, {'smiles': 'CCOc1ccc(-n2ccnc2SCC(=O)N2CCCc3ccccc32)cc1', 'adduct': '[M+H]+', 'instrument': 'QTOF', 'ce': 40.0}, {'smiles': 'Cc1cc(C(=O)O)nc(N2CCCC(C(=O)c3ccccn3)C2)n1', 'adduct': '[M+H]+', 'instrument': 'QTOF', 'ce': 40.0}, {'smiles': 'CC1=NC(=CC(=N1)N2CCCC2)C3CCCN(C3)CCOCCO', 'adduct': '[M+H]+', 'instrument': 'QTOF', 'ce': 40.0}]  # frozen_public_development_jobs


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path);parser.add_argument('--site',type=Path)
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working'));args=parser.parse_args()
    matches=[args.package/'gl_runner.py'] if args.package else list(Path('/kaggle/input').rglob('casmi26-glacier/gl_runner.py'))
    assert len(matches)==1,matches;pkg=matches[0].parent
    wheels=pkg.parent/'casmi26-iceberg/wheels';assert wheels.is_dir()
    out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'feature-probe.json').exists()
    spec=importlib.util.spec_from_file_location('c75_original_gl',pkg/'gl_runner.py')
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    site=args.site or out/'ice_site'
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(wheels),site=str(site),any_rdkit=True),print)
    model=g.load_model(str(pkg),'cpu',print);fz=g.Featurizer(model)
    items=[];canonical=[]
    for i,job in enumerate(JOBS):
        c=g.canon(job['smiles']);assert c is not None
        canonical.append(c)
        items.append(fz.item(c,job['adduct'],job['ce'],job['instrument'],name='JOB'+str(i)))
    from ms_pred.glacier.dataset import IntenDataset
    batch=IntenDataset.collate_fn(items);arrays={}
    def snapshot(value,path):
        if isinstance(value,torch.Tensor):value=value.detach().cpu().contiguous().numpy()
        if isinstance(value,np.ndarray):
            name='.'.join(path);arrays[name]=value
            return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
        if isinstance(value,dict):return {str(k):snapshot(v,path+[str(k)]) for k,v in sorted(value.items())}
        if isinstance(value,(list,tuple)):return [snapshot(v,path+[str(i)]) for i,v in enumerate(value)]
        if value is None or isinstance(value,(str,bool,int,float)):return value
        raise TypeError(('Unsupported scientific feature object',path,type(value).__name__))
    features=snapshot(batch,['batch'])
    with torch.inference_mode():predicted=model.predict_inten_frag_batch(batch)['spec']
    spectra=[]
    for i,prediction in enumerate(predicted):
        mz,intensity=g.post(prediction.float().cpu().numpy())
        arrays['prediction.'+str(i)+'.mz']=mz;arrays['prediction.'+str(i)+'.intensity']=intensity
        spectra.append(dict(mz=mz.tolist(),intensity=intensity.tolist()))
    assert not torch.cuda.is_initialized(),'CPU only'
    np.savez_compressed(out/'feature-arrays.npz',**arrays)
    import rdkit
    import importlib.metadata as metadata
    versions={name:metadata.version(name) for name in ['torch','numpy','pandas','scipy','numba','networkx','multiprocess','platformdirs']}
    manifest=json.loads((pkg/'MANIFEST.json').read_text())
    scientific={name:sha(pkg/name) for name in manifest['files'] if (pkg/name).is_file()}
    result=dict(python=platform.python_version(),source_sha256=sha(__file__),jobs=JOBS,canonical=canonical,
        features=features,predicted_spectra=spectra,versions=versions,rdkit=rdkit.__version__,
        package_files_sha256=scientific,checkpoint_sha256=sha(pkg/'ckpt/glacier.pt'),
        arrays_sha256=sha(out/'feature-arrays.npz'),GPU_initialized=False,
        USED_only=True,query_truth_or_quality_read=False,C13_read=False,scoring_entries=0)
    (out/'feature-probe.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('C75 CPU FEATURE/PREDICTION PROBE COMPLETE',len(JOBS),'jobs',flush=True)


if __name__=='__main__':main()
