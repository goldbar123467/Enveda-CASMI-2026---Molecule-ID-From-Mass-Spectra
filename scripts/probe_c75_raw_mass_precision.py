"""Instrument unchanged GL forward output to test mass-bin amplification."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[n]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
import importlib.util
from types import SimpleNamespace
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    b=ROOT/'outputs/local/c75-local-core-port-20261112';out=b/'raw-mass-precision-probe-v1';assert not out.exists();out.mkdir()
    pkg=b/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    spec=importlib.util.spec_from_file_location('c75_mass_gl',pkg/'gl_runner.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(pkg.parent/'casmi26-iceberg/wheels'),site='/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',any_rdkit=True),print)
    selected=read(b/'device-path-probe-v1/intent.json')['selected'];jobs=set()
    for item in selected:jobs.update(g.molecule_jobs(g.molecule_plan(item),item['cands']))
    original_keys=sorted(jobs)
    keys=[(g.canon(s),ad,ins,ce) for s,ad,ins,ce in original_keys];assert all(k[0] for k in keys)
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),prior_fixed_case_intent_sha256=sha(b/'device-path-probe-v1/intent.json'),
        runner_sha256=sha(pkg/'gl_runner.py'),jobs=len(keys),instrumentation='Capture copies at original post() boundary; no prediction/scoring/model modification',
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    post=g.post;captured=[]
    def capture(sp,k=100):
        captured.append(sp.copy());return post(sp,k)
    g.post=capture
    model=g.load_model(str(pkg),'cpu',print);raw={};posted={};arrays={}
    for device in ['cpu','cuda']:
        captured.clear();model=model.to(device).eval();predictor=g.Predictor(model,device,32,print)
        values=predictor._run(keys);assert len(values)==len(captured)==len(keys)
        raw[device]=[v.copy() for v in captured];posted[device]=values
        for i,v in enumerate(raw[device]):arrays[device+'.'+str(i)]=v
        print('C75 RAW OUTPUT CAPTURED',device,len(values),flush=True)
    np.savez_compressed(out/'raw-predictions.npz',**arrays)
    records=[]
    for i,(a,z) in enumerate(zip(raw['cpu'],raw['cuda'])):
        assert a.shape==z.shape,(i,a.shape,z.shape)
        pinned=z.copy();pinned[:,0]=a[:,0]
        pm,pi=post(pinned);cm,ci=posted['cpu'][i]
        same_mass_bins=bool(np.array_equal(pm,cm))
        records.append(dict(job=i,rows=len(a),maximum_raw_mass_difference=float(np.abs(a[:,0]-z[:,0]).max(initial=0)),
            maximum_raw_intensity_difference=float(np.abs(a[:,1]-z[:,1]).max(initial=0)),
            four_decimal_bin_changes=int((np.round(a[:,0].astype(float),4)!=np.round(z[:,0].astype(float),4)).sum()),
            counterfactual_fixed_mass_bins_equal=same_mass_bins,
            fixed_mass_intensity_max_difference=float(np.abs(pi-ci).max(initial=0)) if same_mass_bins and pi.shape==ci.shape else None))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),raw_arrays_sha256=sha(out/'raw-predictions.npz'),
        cases=records,USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 RAW MASS/POSTPROCESSING DIAGNOSIS',records,flush=True)


if __name__=='__main__':main()
