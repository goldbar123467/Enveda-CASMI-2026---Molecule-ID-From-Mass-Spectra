"""Test projected tie-breaking versus the strictly monotone raw-logit order.

Engineering experiment only on the previously frozen16USED jobs. No candidate
qualification, model fitting, query truth or fresh-panel outcomes are involved.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[n]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
import importlib.util,inspect,types
from types import SimpleNamespace
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    b=ROOT/'outputs/local/c75-local-core-port-20261112';out=b/'fragment-tie-probe-v1';assert not out.exists();out.mkdir()
    pkg=b/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    sp=importlib.util.spec_from_file_location('c75_tie_gl',pkg/'gl_runner.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(pkg.parent/'casmi26-iceberg/wheels'),site='/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',any_rdkit=True),print)
    selected=read(b/'device-path-probe-v1/intent.json')['selected'];jobs=set()
    for item in selected:jobs.update(g.molecule_jobs(g.molecule_plan(item),item['cands']))
    jobs=sorted(jobs);keys=[(g.canon(s),ad,ins,ce) for s,ad,ins,ce in jobs]
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),fixed_cases_sha256=sha(b/'device-path-probe-v1/intent.json'),
        variants=['stock','stable_projected','stable_raw_logit'],criterion='Cross-device choices/scores and preservation of raw learned ordering, never truth/public score',
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    model=g.load_model(str(pkg),'cpu',print);original=type(model).breakpoint_inference
    (out/'original-breakpoint-method.py').write_text(inspect.getsource(original),encoding='utf8')
    all_arrays={};scores={};statistics={}
    for variant in ['stock','stable_projected','stable_raw_logit']:
        for device in ['cpu','cuda']:
            captured=[]
            def controlled(self,logit,card,num_atoms,debug=False):
                projected=self.node_ranking(logit,num_atoms)
                B,N,A=projected.shape[:3];k=torch.argmax(card,dim=-1)
                gathered=torch.gather(projected,index=k[:,:,None,None].expand(B,N,A,1),dim=-1).squeeze(-1)
                flat=gathered.reshape(-1,A);kk=k.reshape(-1)
                sorted_values=torch.sort(flat,dim=-1,descending=True,stable=True).values
                valid=(kk>0)&(kk<A)
                boundary=valid&(sorted_values.gather(1,(kk-1).clamp(min=0)[:,None]).squeeze(1)==sorted_values.gather(1,kk.clamp(max=A-1)[:,None]).squeeze(1))
                if variant=='stock':patterns=original(self,logit,card,num_atoms,debug)
                else:
                    if variant=='stable_raw_logit':
                        mask=torch.arange(A,device=logit.device).unsqueeze(0)>=num_atoms.unsqueeze(1)
                        flat=logit.masked_fill(mask.unsqueeze(1),-1e9).reshape(-1,A)
                    order=torch.argsort(flat,dim=-1,descending=True,stable=True)
                    rank=torch.empty_like(order);rank.scatter_(1,order,torch.arange(A,device=order.device).expand_as(order))
                    patterns=(rank<kk[:,None]).view(B,N,A)
                captured.append(dict(raw=logit.detach().cpu().numpy(),card=k.detach().cpu().numpy(),pattern=patterns.detach().cpu().numpy(),
                    projected=gathered.detach().cpu().numpy(),boundary_tie_count=int(boundary.sum()),nonzero_cardinality=int((kk>0).sum())))
                return patterns
            model.breakpoint_inference=types.MethodType(controlled,model)
            model=model.to(device).eval();predictor=g.Predictor(model,device,32,print)
            predicted=predictor._run(keys);assert len(predicted)==len(keys) and len(captured)==1
            c=captured[0];tag=variant+'.'+device
            for name in ['raw','card','pattern','projected']:all_arrays[tag+'.'+name]=c[name]
            statistics[tag]={name:c[name] for name in ['boundary_tie_count','nonzero_cardinality']}
            cache={job:value for job,value in zip(jobs,predicted)}
            scores[tag]={item['mid']:g.score_molecule(g.molecule_plan(item),item['cands'],cache) for item in selected}
            print('C75 FRAGMENT TIE EXPERIMENT DONE',tag,statistics[tag],flush=True)
    np.savez_compressed(out/'breakpoint-tensors.npz',**all_arrays)
    comparisons={}
    for variant in ['stock','stable_projected','stable_raw_logit']:
        a=variant+'.cpu';z=variant+'.cuda'
        pairs=[(v,scores[z][mid][s]) for mid,entries in scores[a].items() for s,v in entries.items()]
        comparisons[variant]=dict(pattern_bits_different=int((all_arrays[a+'.pattern']!=all_arrays[z+'.pattern']).sum()),
            cardinality_classes_different=int((all_arrays[a+'.card']!=all_arrays[z+'.card']).sum()),
            maximum_raw_logit_difference=float(np.abs(all_arrays[a+'.raw']-all_arrays[z+'.raw']).max(initial=0)),
            score_max_difference=max(abs(x-y) for x,y in pairs),scores_outside_tolerance=sum(abs(x-y)>1e-5+1e-5*abs(y) for x,y in pairs))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),
        tensors_sha256=sha(out/'breakpoint-tensors.npz'),statistics=statistics,comparisons=comparisons,scores=scores,
        inference_method_changes_only_experimental=True,qualified_for_scoring=False,USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 FRAGMENT TIE CAUSAL COMPARISON',comparisons,flush=True)


if __name__=='__main__':main()
