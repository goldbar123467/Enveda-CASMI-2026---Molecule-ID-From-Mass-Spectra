"""Test one minimal float64 mass-reduction repair on the frozen16USED jobs.

Only the summation of existing float32 atom masses is promoted. Those inputs
are exactly representable in float64, avoiding device-dependent float32 sums.
Weights, neural/intensity calculations, top-k rules and scoring stay unchanged.
This is an engineering candidate, not a passed C75 gate or qualified submission.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=True)
import hashlib,importlib.util,inspect,textwrap,types
from types import SimpleNamespace
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    b=ROOT/'outputs/local/c75-local-core-port-20261112';out=b/'mass-reduction-repair-v1'
    assert not out.exists();out.mkdir()
    pkg=b/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    sp=importlib.util.spec_from_file_location('c75_repair_gl',pkg/'gl_runner.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
    g.setup_env(SimpleNamespace(pkg=str(pkg),wheels=str(pkg.parent/'casmi26-iceberg/wheels'),site='/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',any_rdkit=True),print)
    selected=read(b/'device-path-probe-v1/intent.json')['selected'];jobs=set()
    for item in selected:jobs.update(g.molecule_jobs(g.molecule_plan(item),item['cands']))
    jobs=sorted(jobs);keys=[(g.canon(s),ad,ins,ce) for s,ad,ins,ce in jobs]
    model=g.load_model(str(pkg),'cpu',print);original=type(model).inten_calculation
    before=textwrap.dedent(inspect.getsource(original))
    needle='net_fragment_mass_flat = (masses_expanded * frag_targs_f).sum(dim=-1)'
    replacement='net_fragment_mass_flat = (masses_expanded.double() * frag_targs_f.double()).sum(dim=-1)'
    assert before.count(needle)==1
    after=before.replace(needle,replacement);assert after.replace(replacement,needle)==before
    (out/'original-intensity-method.py').write_text(before,encoding='utf8')
    (out/'repaired-intensity-method.py').write_text(after,encoding='utf8')
    write(out/'intent.json',dict(started_utc=stamp(),source_sha256=sha(__file__),fixed_cases_sha256=sha(b/'device-path-probe-v1/intent.json'),
        checkpoint_sha256=sha(pkg/'ckpt/glacier.pt'),original_method_sha256=sha(out/'original-intensity-method.py'),
        repaired_method_sha256=sha(out/'repaired-intensity-method.py'),only_change='Promote existing atom-mass multiplication/reduction to float64; stock output float32 cast remains',
        predeclared_acceptance='All16CPU/CUDA scores within original atol=rtol=1e-5; no claim of exact previousT4scores or complete-list parity',
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    namespace={};exec(compile(after,'c75_mass_reduction_repair','exec'),original.__globals__,namespace)
    model.inten_calculation=types.MethodType(namespace['inten_calculation'],model)
    scores={};predictions={}
    for device in ['cpu','cuda']:
        model=model.to(device).eval();predictor=g.Predictor(model,device,32,print)
        values=predictor._run(keys);assert len(values)==16 and all(v is not None for v in values)
        cache=dict(zip(jobs,values));scores[device]={item['mid']:g.score_molecule(g.molecule_plan(item),item['cands'],cache) for item in selected}
        predictions[device]=values
        print('C75 MASS REDUCTION REPAIR DONE',device,flush=True)
    pairs=[(v,scores['cuda'][mid][s]) for mid,items in scores['cpu'].items() for s,v in items.items()]
    bins=[np.array_equal(a[0],z[0]) for a,z in zip(predictions['cpu'],predictions['cuda'])]
    comparison=dict(scores=16,mass_bins_exact=sum(bins),max_score_difference=max(abs(a-z) for a,z in pairs),
        scores_outside_original_tolerance=sum(abs(a-z)>1e-5+1e-5*abs(z) for a,z in pairs),
        maximum_intensity_difference=max(float(np.abs(a[1]-z[1]).max(initial=0)) for a,z,equal in zip(predictions['cpu'],predictions['cuda'],bins) if equal))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),intent_sha256=sha(out/'intent.json'),
        comparison=comparison,scores=scores,predeclared_component_criterion_passed=comparison['scores_outside_original_tolerance']==0,
        original_C75_fixture_gate_passed=False,qualified_for_scoring=False,complete_pipeline_recipe_revision_required=True,
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 MASS REDUCTION REPAIR COMPARISON',comparison,flush=True)


if __name__=='__main__':main()
