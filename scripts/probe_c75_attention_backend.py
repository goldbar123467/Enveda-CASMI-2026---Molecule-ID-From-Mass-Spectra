"""Localize CPU/CUDA numerical divergence on the frozen16 USED jobs.

The stock parent remains unchanged. Enforcing the SDPA math backend is an
engineering counterfactual, not a qualified pipeline or a quality experiment.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True,warn_only=True)
torch.backends.cudnn.benchmark=False
import contextlib,importlib.util,types
from types import SimpleNamespace
import numpy as np
from torch.nn.attention import sdpa_kernel,SDPBackend
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'attention-backend-probe-v1';assert not out.exists();out.mkdir()
    pkg=base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    sp=importlib.util.spec_from_file_location('c75_attention_gl',pkg/'gl_runner.py')
    g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(pkg.parent/'casmi26-iceberg/wheels'),site='/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',any_rdkit=True),print)
    selected=read(base/'device-path-probe-v1/intent.json')['selected'];jobs=set()
    for item in selected:jobs.update(g.molecule_jobs(g.molecule_plan(item),item['cands']))
    jobs=sorted(jobs);keys=[(g.canon(s),ad,ins,ce) for s,ad,ins,ce in jobs]
    from rdkit import Chem
    counts=np.array([Chem.MolFromSmiles(s).GetNumAtoms() for s,ad,ins,ce in keys])
    variants=[('stock_cpu','cpu',False),('stock_cuda','cuda',False),('math_cuda','cuda',True)]
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),fixed_cases_sha256=sha(base/'device-path-probe-v1/intent.json'),
        variants=variants,criterion='Numerical layer/pattern/score comparison on the same16USEDjobs, no truth or quality',
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    model=g.load_model(str(pkg),'cpu',print)
    originals={n:getattr(type(model),n) for n in ['molecular_embedding','breakpoint_forward','breakpoint_inference']}
    arrays={};scores={}
    for variant,device,enforce_math in variants:
        captured={}
        def embedding(self,*args,**kwargs):
            result=originals['molecular_embedding'](self,*args,**kwargs)
            for name,value in result.items():captured[name]=value.detach().cpu().numpy()
            return result
        def forward(self,*args,**kwargs):
            result=originals['breakpoint_forward'](self,*args,**kwargs)
            for name,value in result.items():captured[name]=value[-1].detach().cpu().numpy()
            return result
        def inference(self,*args,**kwargs):
            result=originals['breakpoint_inference'](self,*args,**kwargs)
            captured['pattern']=result.detach().cpu().numpy()
            return result
        model.molecular_embedding=types.MethodType(embedding,model)
        model.breakpoint_forward=types.MethodType(forward,model)
        model.breakpoint_inference=types.MethodType(inference,model)
        model=model.to(device).eval();predictor=g.Predictor(model,device,32,print)
        with sdpa_kernel(SDPBackend.MATH) if enforce_math else contextlib.nullcontext():
            predicted=predictor._run(keys)
        assert len(predicted)==16 and set(captured)=={'root_tokens','node_embeddings','frag_logits','frag_card_logits','pattern'}
        for name,value in captured.items():arrays[variant+'.'+name]=value
        cache=dict(zip(jobs,predicted))
        scores[variant]={item['mid']:g.score_molecule(g.molecule_plan(item),item['cands'],cache) for item in selected}
        print('C75 ATTENTION BACKEND DONE',variant,flush=True)
    previous=read(base/'device-path-probe-v1/results.json')['scores']
    for variant,device,enforce_math in variants[:2]:
        assert scores[variant]==previous[device],('Stock reproduction failed',variant)
    raw=read(base/'original-cloud-raw-forward/gl_work/gl_out.json')
    comparisons={}
    for variant,device,enforce_math in variants:
        detail={}
        for name in ['root_tokens','node_embeddings','frag_logits','frag_card_logits']:
            a=arrays['stock_cpu.'+name];z=arrays[variant+'.'+name]
            diff=np.abs(a-z)
            if name in ['node_embeddings','frag_logits']:
                valid=np.arange(a.shape[1 if name=='node_embeddings' else 2])[None,:]<counts[:,None]
                diff=diff[np.broadcast_to(valid[:,:,None] if name=='node_embeddings' else valid[:,None,:],a.shape)]
            detail[name]=dict(maxdiff_to_CPU=float(diff.max(initial=0)),median_diff_to_CPU=float(np.median(diff)))
        pairs=[(v,raw[mid][s]) for mid,entries in scores[variant].items() for s,v in entries.items()]
        detail.update(pattern_bits_different_from_CPU=int((arrays['stock_cpu.pattern']!=arrays[variant+'.pattern']).sum()),
            max_score_diff_to_T4=max(abs(a-z) for a,z in pairs),scores_outside_T4_tolerance=sum(abs(a-z)>1e-5+1e-5*abs(z) for a,z in pairs))
        comparisons[variant]=detail
    np.savez_compressed(out/'tensors.npz',**arrays)
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),
        tensors_sha256=sha(out/'tensors.npz'),comparisons=comparisons,scores=scores,stock_reproduced_exactly=True,
        experimental_only=True,qualified_for_scoring=False,USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 ATTENTION BACKEND COMPARISON',comparisons,flush=True)


if __name__=='__main__':main()
