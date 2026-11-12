"""Freeze one explicit local resource recovery; retain the rejected attempt."""
import base64,copy,hashlib,json,platform
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp,unwrap,value,replace_value


def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old));return text.replace(old,new)


def main():
    assert platform.system()=='Linux'
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';prior=base/'fixture-port'
    assert read(prior/'supervisor-retry-failed.json')['status']=='failed'
    stopped=read(prior/'intentional-performance-stop.json');assert not stopped['queries_or_candidates_changed']
    assert not (prior/'run-complete.json').exists()
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    out=base/'fixture-resource-port';assert not out.exists();out.mkdir()
    working=Path('/home/imsa/.local/share/enveda-c76/fixture-resource/working')
    assert not working.parent.exists();working.mkdir(parents=True)
    parent=read(prior/'notebook.ipynb');nb=copy.deepcopy(parent)
    outer=unwrap(''.join(nb['cells'][12]['source']));payload=base64.b64decode(value(outer,'FROZEN_B64'))
    inner=json.loads(payload);before=''.join(inner['cells'][41]['source'])
    target="site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])"
    after=replace(before,target,"site='/kaggle/working/ice_site', extra_args=['--any-rdkit','--bs','8'])")
    line='    ICE_SCORES = ice_fuse.run_ice('
    addition="    gc.collect(); torch.cuda.empty_cache()  # C76 local resource adapter: release unused parent cache before ICE\n"
    after=replace(after,line,addition+line);inner['cells'][41]['source']=after.splitlines(keepends=True)
    compiled=json.dumps(inner,ensure_ascii=False,indent=1).encode()
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(compiled).decode())
    outer=replace_value(outer,'FROZEN_SHA256',hashlib.sha256(compiled).hexdigest())
    nb['cells'][12]['source']=outer.splitlines(keepends=True)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),'c76_resource_outer_'+str(i),'exec')
    write(out/'notebook.ipynb',nb);(out/'parent-inner.ipynb').write_bytes(payload);(out/'resource-inner.ipynb').write_bytes(compiled)
    run=ROOT/'scripts/run_c76_fixture.py';assert sha(run)==read(prior/'run-intent.json')['source_sha256']
    text=run.read_text(encoding='utf8')
    for old,new in [("out=base/'fixture-port'","out=base/'fixture-resource-port'"),
        ('scripts/prepare_c76_fixture.py','scripts/prepare_c76_resource_retry.py'),
        ('scripts/audit_c76_fixture_source.py','scripts/audit_c76_resource_source.py'),
        ('scripts/activate_c76_fixture_namespace.py','scripts/activate_c76_resource_namespace.py'),
        ("Path('/home/imsa/.local/share/enveda-c76/fixture/working')","Path('/home/imsa/.local/share/enveda-c76/fixture-resource/working')")]:text=replace(text,old,new)
    newrun=ROOT/'scripts/run_c76_resource_fixture.py';assert not newrun.exists();newrun.write_text(text,encoding='utf8')
    activation=ROOT/'scripts/activate_c76_fixture_namespace.py';text=activation.read_text(encoding='utf8')
    text=replace(text,"/fixture-port'","/fixture-resource-port'")
    text=replace(text,"assert read(base/'supervisor-failed.json')['status']=='failed' and not (base/'run-intent.json').exists()",
        "prior=ROOT/'outputs/local/c76-numerical-portability-20261112/fixture-port'\n    assert (prior/'intentional-performance-stop.json').exists() and read(prior/'supervisor-retry-failed.json')['status']=='failed'\n    assert not (base/'run-intent.json').exists()")
    text=replace(text,"Path('/home/imsa/.local/share/enveda-c75/fixture/working')","Path('/home/imsa/.local/share/enveda-c76/fixture/working')")
    text=replace(text,"Path('/home/imsa/.local/share/enveda-c76/fixture/working') and not working.is_symlink()",
        "Path('/home/imsa/.local/share/enveda-c76/fixture-resource/working') and not working.is_symlink()")
    newactivation=ROOT/'scripts/activate_c76_resource_namespace.py';assert not newactivation.exists();newactivation.write_text(text,encoding='utf8')
    pin=read(prior/'fixture-freeze.json');pin.update(frozen_utc=stamp(),source_sha256=sha(__file__),
        parent_notebook_sha256=sha(prior/'notebook.ipynb'),notebook_sha256=sha(out/'notebook.ipynb'),
        parent_fixture_freeze_sha256=sha(prior/'fixture-freeze.json'),inner_parent_sha256=hashlib.sha256(payload).hexdigest(),
        inner_candidate_sha256=hashlib.sha256(compiled).hexdigest(),owned_working=str(working),
        operational_resource_policy_sha256=sha(ROOT/'configs/c76-local-resource-adapter-v1.json'),
        intentional_performance_stop_sha256=sha(prior/'intentional-performance-stop.json'),
        operational_changes=['Empty unused parent CUDA cache before ICE','Original ICE --bs8'],
        generated_executor_sha256=sha(newrun),generated_namespace_activator_sha256=sha(newactivation),
        models_and_input_data_unchanged=True,exact_ICE_parity_across_batch_sizes_not_assumed=True)
    write(out/'fixture-freeze.json',pin)
    print('C76 FRESH RESOURCE RETRY FROZEN; ORIGINAL ATTEMPT AND ALL INPUTS RETAINED',flush=True)


if __name__=='__main__':main()
