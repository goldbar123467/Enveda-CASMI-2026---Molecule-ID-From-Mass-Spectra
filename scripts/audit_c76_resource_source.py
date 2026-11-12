"""Audit only the two declared local resource changes, before any retry."""
import base64,json
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp,unwrap,value,replace_value


def main():
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';prior=base/'fixture-port';out=base/'fixture-resource-port'
    dest=out/'independent-source-audit.json';assert not dest.exists()
    pin=read(out/'fixture-freeze.json');assert pin['source_sha256']==sha(ROOT/'scripts/prepare_c76_resource_retry.py')
    assert pin['operational_resource_policy_sha256']==sha(ROOT/'configs/c76-local-resource-adapter-v1.json')
    a=read(prior/'notebook.ipynb');z=read(out/'notebook.ipynb')
    assert [i for i,(x,y) in enumerate(zip(a['cells'],z['cells'])) if x['source']!=y['source']]==[12]
    x=unwrap(''.join(a['cells'][12]['source']));y=unwrap(''.join(z['cells'][12]['source']))
    for name in ['FROZEN_B64','FROZEN_SHA256']:y=replace_value(y,name,value(x,name))
    assert x==y
    a=read(out/'parent-inner.ipynb');z=read(out/'resource-inner.ipynb')
    assert [i for i,(x,y) in enumerate(zip(a['cells'],z['cells'])) if x['source']!=y['source']]==[41]
    x=''.join(a['cells'][41]['source']);y=''.join(z['cells'][41]['source'])
    addition="    gc.collect(); torch.cuda.empty_cache()  # C76 local resource adapter: release unused parent cache before ICE\n"
    assert y.count(addition)==1;y=y.replace(addition,'')
    needle="site='/kaggle/working/ice_site', extra_args=['--any-rdkit','--bs','8'])";assert y.count(needle)==1
    y=y.replace(needle,"site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])");assert y==x
    assert read(prior/'independent-source-audit.json')['passed']
    assert pin['parent_fixture_freeze_sha256']==sha(prior/'fixture-freeze.json')
    pkg=Path(pin['owned_GL_package'])
    assert sha(pkg/'gl_runner.py')==pin['candidate_gl_runner_sha256'] and sha(pkg/'c76_gl_portability.py')==pin['helper_sha256']
    assert sha(pkg/'ckpt/glacier.pt')==pin['original_checkpoint_sha256']
    assert pin['generated_executor_sha256']==sha(ROOT/'scripts/run_c76_resource_fixture.py')
    assert pin['generated_namespace_activator_sha256']==sha(ROOT/'scripts/activate_c76_resource_namespace.py')
    write(dest,dict(completed_utc=stamp(),source_sha256=sha(__file__),fixture_freeze_sha256=sha(out/'fixture-freeze.json'),
        passed=True,only_new_changes=['Release unused parent CUDA cache before ICE','Supported ICE CLI --bs8'],
        parent_GL_numerical_source_audit_sha256=sha(prior/'independent-source-audit.json'),
        model_weights_and_functions_unchanged=True,original_queries_candidates_and_reference_masks_retained=True,
        exact_ICE_parity_across_batch_sizes_not_assumed=True,original_C75_gate_passed=False,
        quality_read=False,C13_read=False,scoring_entries=0))
    print('C76 RESOURCE-ONLY SOURCE AUDIT PASSED',flush=True)


if __name__=='__main__':main()
