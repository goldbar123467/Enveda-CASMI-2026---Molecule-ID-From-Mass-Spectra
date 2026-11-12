"""Test CPU fragment decisions plus GPU intensities and float64 mass reduction.

Keep original learned functions, weights and top-k semantics. The small decision
path has one CPU implementation on both devices; only expensive intensity work
uses CUDA. This prospective engineering candidate does not pass the old gate.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
import importlib.util,inspect,textwrap,types
from types import SimpleNamespace
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    return value


def main():
    b=ROOT/'outputs/local/c75-local-core-port-20261112';out=b/'portable-breakpoints-v1'
    assert not out.exists();out.mkdir()
    pkg=b/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    sp=importlib.util.spec_from_file_location('c75_portable_gl',pkg/'gl_runner.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(pkg.parent/'casmi26-iceberg/wheels'),site='/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',any_rdkit=True),print)
    selected=read(b/'device-path-probe-v1/intent.json')['selected'];jobs=set()
    for item in selected:jobs.update(g.molecule_jobs(g.molecule_plan(item),item['cands']))
    jobs=sorted(jobs);keys=[(g.canon(s),ad,ins,ce) for s,ad,ins,ce in jobs]
    model=g.load_model(str(pkg),'cpu',print);decision_model=g.load_model(str(pkg),'cpu',print)
    original=type(model).inten_calculation;before=textwrap.dedent(inspect.getsource(original))
    needle='net_fragment_mass_flat = (masses_expanded * frag_targs_f).sum(dim=-1)'
    replacement='net_fragment_mass_flat = (masses_expanded.double() * frag_targs_f.double()).sum(dim=-1)'
    assert before.count(needle)==1;after=before.replace(needle,replacement)
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),fixed_cases_sha256=sha(b/'device-path-probe-v1/intent.json'),
        checkpoint_sha256=sha(pkg/'ckpt/glacier.pt'),changes=['Original molecular_embedding/breakpoint_forward/breakpoint_inference always CPU','Existing atom-mass reduction float64'],
        untouched=['weights','neural function definitions','stock top-k','fragment graph algorithm','intensity neural network','postprocess','score function'],
        predeclared_acceptance='16of16scores at original atol=rtol=1e-5 acrossCPU/CUDA; all16mass bins exact',
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    namespace={};exec(compile(after,'c75_mass_reduction_repair','exec'),original.__globals__,namespace)
    model.inten_calculation=types.MethodType(namespace['inten_calculation'],model)
    captured={}
    def embedding(self,*args,**kwargs):
        device=args[0].device
        value=decision_model.molecular_embedding(*move(args,'cpu'),**move(kwargs,'cpu'))
        return move(value,device)
    def forward(self,*args,**kwargs):
        device=args[0].device
        return move(decision_model.breakpoint_forward(*move(args,'cpu'),**move(kwargs,'cpu')),device)
    def inference(self,logit,card,num_atoms,debug=False):
        value=decision_model.breakpoint_inference(logit.cpu(),card.cpu(),num_atoms.cpu(),debug)
        captured['pattern']=value.cpu().numpy().copy()
        return value.to(logit.device)
    model.molecular_embedding=types.MethodType(embedding,model)
    model.breakpoint_forward=types.MethodType(forward,model)
    model.breakpoint_inference=types.MethodType(inference,model)
    scores={};predictions={};patterns={};timings={}
    import time
    for device in ['cpu','cuda']:
        start=time.monotonic();model=model.to(device).eval();predictor=g.Predictor(model,device,32,print)
        values=predictor._run(keys);assert len(values)==16 and all(v is not None for v in values)
        timings[device]=time.monotonic()-start;patterns[device]=captured['pattern']
        cache=dict(zip(jobs,values));scores[device]={item['mid']:g.score_molecule(g.molecule_plan(item),item['cands'],cache) for item in selected}
        predictions[device]=values
        assert next(decision_model.parameters()).device.type=='cpu'
        print('C75 PORTABLE BREAKPOINTS DONE',device,timings[device],flush=True)
    pairs=[(v,scores['cuda'][mid][s]) for mid,items in scores['cpu'].items() for s,v in items.items()]
    bins=[np.array_equal(a[0],z[0]) for a,z in zip(predictions['cpu'],predictions['cuda'])]
    comparison=dict(scores=16,mass_bins_exact=sum(bins),max_score_difference=max(abs(a-z) for a,z in pairs),
        scores_outside_original_tolerance=sum(abs(a-z)>1e-5+1e-5*abs(z) for a,z in pairs),
        pattern_bits_different=int((patterns['cpu']!=patterns['cuda']).sum()),
        maximum_intensity_difference=max(float(np.abs(a[1]-z[1]).max(initial=0)) for a,z,equal in zip(predictions['cpu'],predictions['cuda'],bins) if equal))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),comparison=comparison,
        timings_seconds=timings,scores=scores,predeclared_component_criterion_passed=comparison['scores_outside_original_tolerance']==0 and all(bins),
        original_C75_fixture_gate_passed=False,qualified_for_scoring=False,complete_pipeline_recipe_revision_required=True,
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 PORTABLE BREAKPOINTS COMPARISON',comparison,flush=True)


if __name__=='__main__':main()
