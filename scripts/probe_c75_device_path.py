"""Compare CPU/2070 forward physics with saved T4 scores on fixed USED cases."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
import hashlib,importlib.util,time
from types import SimpleNamespace
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def draw(value):return hashlib.sha256(('C75-device-path-v1:'+value).encode()).hexdigest()


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'device-path-probe-v1';assert not out.exists();out.mkdir()
    pkg=base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    spec=importlib.util.spec_from_file_location('c75_device_gl',pkg/'gl_runner.py')
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(pkg.parent/'casmi26-iceberg/wheels'),site='/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',any_rdkit=True),print)
    original=base/'original-cloud-intermediates/gl_work/gl_in.json';items=read(original)
    raw_path=base/'original-cloud-raw-forward/gl_work/gl_out.json';raw=read(raw_path)
    selected=[]
    for item in sorted(items,key=lambda item:draw(item['mid'])):
        plan=g.molecule_plan(item)
        kinds={(ad,ins,float(ce)) for _,_,ad,ins,ce in plan if np.isfinite(ce)}
        if not plan or len(kinds)!=1 or any(not np.isfinite(ce) for _,_,_,_,ce in plan):continue
        candidates=[s for s in sorted(item['cands'],key=draw) if raw[item['mid']].get(s) is not None and g.smiles_ok(s)[0]][:4]
        if len(candidates)!=4:continue
        selected.append((dict(item,cands=candidates),plan))
        if len(selected)==4:break
    assert len(selected)==4
    jobs=set().union(*(g.molecule_jobs(plan,item['cands']) for item,plan in selected))
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),original_input_sha256=sha(original),original_T4_raw_sha256=sha(raw_path),
        runner_sha256=sha(pkg/'gl_runner.py'),checkpoint_sha256=sha(pkg/'ckpt/glacier.pt'),
        selection='First four fixed-hash USED queries having one finite original CE/adduct/instrument kind; first4formalvalid candidates by hash. No truth or quality selection.',
        molecules=4,candidates=16,unique_jobs=len(jobs),selected=[item for item,_ in selected],
        USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    model=g.load_model(str(pkg),'cpu',print)
    results={};predictions={}
    for device in ['cpu','cuda']:
        if device=='cuda':assert '2070' in torch.cuda.get_device_name(0)
        model=model.to(device).eval();p=g.Predictor(model,device,32,print)
        finished=p.predict(jobs,time.time()+600);assert finished and p.n_fail==p.n_feat_fail==0
        assert all(job in p.cache and p.cache[job] is not None for job in jobs)
        scores={}
        for item,plan in selected:
            scores[item['mid']]=g.score_molecule(plan,item['cands'],p.cache)
        results[device]=scores
        predictions[device]={str(i):dict(mz=p.cache[job][0].tolist(),intensity=p.cache[job][1].tolist()) for i,job in enumerate(sorted(jobs))}
        print('C75 DEVICE PATH FINISHED',device,len(jobs),flush=True)
    comparisons={}
    for device in ['cpu','cuda']:
        pairs=[(score,raw[mid][s]) for mid,v in results[device].items() for s,score in v.items()]
        assert all(x is not None for x,y in pairs)
        comparisons[device]=dict(compared_scores=len(pairs),maxdiff_to_T4=max(abs(x-y) for x,y in pairs),
            outside_frozen_tolerance=sum(abs(x-y)>1e-5+1e-5*abs(y) for x,y in pairs))
    pairs=[(score,results['cuda'][mid][s]) for mid,v in results['cpu'].items() for s,score in v.items()]
    comparisons['CPU_vs_2070']=dict(maxdiff=max(abs(x-y) for x,y in pairs),outside_frozen_tolerance=sum(abs(x-y)>1e-5+1e-5*abs(y) for x,y in pairs))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),scores=results,predicted_spectra=predictions,
        comparison=comparisons,USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 DEVICE PATH COMPARISON',comparisons,flush=True)


if __name__=='__main__':main()
