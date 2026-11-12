"""Pin captured fragment decisions to separate selection from mass arithmetic.

Two frozen16USED-job counterfactuals, no labels, training or qualification.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
import importlib.util,types
from types import SimpleNamespace
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    b=ROOT/'outputs/local/c75-local-core-port-20261112';out=b/'fragment-counterfactual-v1'
    assert not out.exists();out.mkdir()
    pkg=b/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    sp=importlib.util.spec_from_file_location('c75_cf_gl',pkg/'gl_runner.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(pkg.parent/'casmi26-iceberg/wheels'),site='/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',any_rdkit=True),print)
    selected=read(b/'device-path-probe-v1/intent.json')['selected'];jobs=set()
    for item in selected:jobs.update(g.molecule_jobs(g.molecule_plan(item),item['cands']))
    jobs=sorted(jobs);keys=[(g.canon(s),ad,ins,ce) for s,ad,ins,ce in jobs]
    tensors=np.load(b/'fragment-tie-probe-v1/breakpoint-tensors.npz')
    original=np.load(b/'raw-mass-precision-probe-v1/raw-predictions.npz')
    frozen_scores=read(b/'device-path-probe-v1/results.json')['scores']
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),fixed_cases_sha256=sha(b/'device-path-probe-v1/intent.json'),
        tensors_sha256=sha(b/'fragment-tie-probe-v1/breakpoint-tensors.npz'),raw_predictions_sha256=sha(b/'raw-mass-precision-probe-v1/raw-predictions.npz'),
        variants=['CPU_with_captured_GPU_decisions','GPU_with_captured_CPU_decisions'],
        criterion='Causal separation of fragment selection and mass arithmetic against saved stock outputs; never truth',
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    post=g.post;captured=[]
    def capture(value,k=100):captured.append(value.copy());return post(value,k)
    g.post=capture
    model=g.load_model(str(pkg),'cpu',print);results={};raw_arrays={}
    for device,pattern_device in [('cpu','cuda'),('cuda','cpu')]:
        captured.clear();pattern=tensors['stock.'+pattern_device+'.pattern']
        def pin(self,logit,card,num_atoms,debug=False):
            assert tuple(logit.shape)==pattern.shape
            assert np.array_equal(torch.argmax(card,dim=-1).cpu().numpy(),tensors['stock.'+pattern_device+'.card'])
            return torch.as_tensor(pattern,device=logit.device)
        model.breakpoint_inference=types.MethodType(pin,model)
        model=model.to(device).eval();predictor=g.Predictor(model,device,32,print)
        values=predictor._run(keys);assert len(captured)==len(values)==16
        fixed_masses=[];records=[]
        for i,value in enumerate(captured):
            stock=original[pattern_device+'.'+str(i)];assert stock.shape==value.shape,(i,stock.shape,value.shape)
            counter=value.copy();counter[:,0]=stock[:,0];fixed_masses.append(post(counter))
            mz,it=post(stock);fm,fi=fixed_masses[-1]
            mass_equal=np.array_equal(mz,fm)
            records.append(dict(case=i,rows=len(value),max_raw_mass_diff=float(np.abs(value[:,0]-stock[:,0]).max(initial=0)),
                max_raw_intensity_diff=float(np.abs(value[:,1]-stock[:,1]).max(initial=0)),
                fixed_mass_bins_equal=mass_equal,fixed_mass_intensity_max_diff=float(np.abs(it-fi).max(initial=0)) if mass_equal else None))
            raw_arrays[device+'.'+str(i)]=value
        score_variants={}
        for name,predictions in [('fixed_fragments',values),('fixed_fragments_and_masses',fixed_masses)]:
            cache=dict(zip(jobs,predictions));scores={item['mid']:g.score_molecule(g.molecule_plan(item),item['cands'],cache) for item in selected}
            pairs=[(v,frozen_scores[pattern_device][mid][s]) for mid,items in scores.items() for s,v in items.items()]
            score_variants[name]=dict(max_diff_to_decision_source_device=max(abs(a-z) for a,z in pairs),
                outside_tolerance=sum(abs(a-z)>1e-5+1e-5*abs(z) for a,z in pairs),scores=scores)
        results[device+'_with_'+pattern_device+'_decisions']=dict(raw_cases=records,score_comparisons=score_variants)
        print('C75 FRAGMENT COUNTERFACTUAL DONE',device,{n:{k:v for k,v in data.items() if k!='scores'} for n,data in score_variants.items()},flush=True)
    np.savez_compressed(out/'raw-predictions.npz',**raw_arrays)
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),
        results=results,raw_predictions_sha256=sha(out/'raw-predictions.npz'),experimental_only=True,qualified_for_scoring=False,
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))


if __name__=='__main__':main()
