"""Prepare a fresh complete USED12 run with only the explicit GL adapter.

Original C75 failed reproduction artifacts stay intact. This candidate changes
only inner cell41's GL package path and its expected runner hash; the package
is an owned clone with read-only links to the original payload and one clearly
audited loader extension. Both ordinary GL and N1 proxy GL use that loader.
"""
import base64,copy,hashlib,json,platform
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp,unwrap,value,replace_value


def main():
    assert platform.system()=='Linux'
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';old=ROOT/'outputs/local/c75-local-core-port-20261112'
    pin=read(base/'candidate-freeze-r1.json')
    for name,digest in pin['files_sha256'].items():assert sha(ROOT/name)==digest,name
    assert read(base/'extended-used-nine-r1/results.json')['component_passed']
    cpu=read(base/'cpu-feature-parity/independent-comparison.json')
    assert cpu['checkpoint_exact'] and cpu['canonical_strings_exact'] and not cpu['different_package_files']
    assert len(cpu['array_comparisons'])==26 and all(v['exact'] for v in cpu['array_comparisons'].values())
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    out=base/'fixture-port';assert not out.exists();out.mkdir()
    native=Path('/home/imsa/.local/share/enveda-c76/fixture');assert not native.exists()
    assert native.is_absolute() and native.parent==Path('/home/imsa/.local/share/enveda-c76')
    native.mkdir(parents=True);working=native/'working';working.mkdir();pkg=native/'glacier';pkg.mkdir()
    original=old/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    runner=original/'gl_runner.py';assert sha(runner)=='5c4f74313ac54263d05d987c39d012d81ec57c5f35e1095ecd49a5bcb6440fca'
    helper=ROOT/'scripts/c76_gl_portability.py';assert sha(helper)==pin['files_sha256']['scripts/c76_gl_portability.py']
    for item in original.iterdir():
        if item.name in ['gl_runner.py','__pycache__']:continue
        (pkg/item.name).symlink_to(item.resolve(),target_is_directory=item.is_dir())
    (pkg/'c76_gl_portability.py').write_bytes(helper.read_bytes())
    before=runner.read_text(encoding='utf8');needle="\nif __name__ == '__main__':\n    sys.exit(main())\n";assert before.count(needle)==1
    extension='''
# C76 explicit numerical adapter; original functions above retained unchanged.
_C76_original_load_model = load_model
def load_model(pkg, dev, log, ckpt=None):
    import hashlib,importlib.util,json
    from pathlib import Path
    checkpoint=Path(ckpt) if ckpt else Path(pkg)/'ckpt/glacier.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12'
    helper=Path(pkg)/'c76_gl_portability.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==HELPER_SHA
    spec=importlib.util.spec_from_file_location('c76_pinned_portability',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=_C76_original_load_model(pkg,dev,log,ckpt)
    provider=_C76_original_load_model(pkg,'cpu',log,ckpt)
    receipt=module.install(model,provider)
    Path(str(sys.argv[3])+'.c76_adapter.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    log('C76 EXPLICIT NUMERICAL ADAPTER',json.dumps(receipt))
    return model
'''.replace('HELPER_SHA',repr(sha(helper)))
    after=before.replace(needle,extension+needle);assert after.replace(extension,'')==before
    compile(after,'c76_explicit_gl_loader','exec');(pkg/'gl_runner.py').write_text(after,encoding='utf8')
    original_nb=read(old/'fixture-port/notebook.ipynb');nb=copy.deepcopy(original_nb)
    outer=unwrap(''.join(nb['cells'][12]['source']));payload=base64.b64decode(value(outer,'FROZEN_B64'))
    assert hashlib.sha256(payload).hexdigest()==value(outer,'FROZEN_SHA256')
    inner=json.loads(payload);inner_original=copy.deepcopy(inner);source=''.join(inner['cells'][41]['source'])
    line="GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))";assert source.count(line)==1
    source=source.replace(line,"GL_PKG = os.environ['CASMI_C76_GL_PACKAGE']")
    assert value(source,'N1_GL_RUNNER_SHA256')==sha(runner)
    source=replace_value(source,'N1_GL_RUNNER_SHA256',sha(pkg/'gl_runner.py'))
    inner['cells'][41]['source']=source.splitlines(keepends=True)
    assert [i for i,(a,z) in enumerate(zip(inner_original['cells'],inner['cells'])) if a['source']!=z['source']]==[41]
    encoded=json.dumps(inner,ensure_ascii=False,indent=1).encode()
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(encoded).decode())
    outer=replace_value(outer,'FROZEN_SHA256',hashlib.sha256(encoded).hexdigest())
    nb['cells'][12]['source']=outer.splitlines(keepends=True)
    assert [i for i,(a,z) in enumerate(zip(original_nb['cells'],nb['cells'])) if a['source']!=z['source']]==[12]
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),'c76_outer_'+str(i),'exec');cell['outputs']=[];cell['execution_count']=None
    write(out/'notebook.ipynb',nb);(out/'c75-parent-inner.ipynb').write_bytes(payload);(out/'c76-inner.ipynb').write_bytes(encoded)
    write(out/'fixture-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),candidate_freeze_sha256=sha(base/'candidate-freeze-r1.json'),
        parent_notebook_sha256=sha(old/'fixture-port/notebook.ipynb'),notebook_sha256=sha(out/'notebook.ipynb'),
        inner_parent_sha256=hashlib.sha256(payload).hexdigest(),inner_candidate_sha256=hashlib.sha256(encoded).hexdigest(),
        original_gl_runner_sha256=sha(runner),candidate_gl_runner_sha256=sha(pkg/'gl_runner.py'),helper_sha256=sha(helper),
        inner_changed_cells=[41],outer_changed_cells=[12],parent_scientific_functions_unchanged=True,
        original_checkpoint_sha256=sha(pkg/'ckpt/glacier.pt'),
        owned_GL_package=str(pkg),owned_working=str(working),original_input_target=read(old/'fixture-port/namespace-activation.json')['input_target'],
        query_package=str(old/'fixture-port/query-package'),queries=12,acquisitions=51,
        both_classic_GL_and_N1_use_same_explicit_adapter=True,original_reference_masks_preserved=True,
        C75_gate_still_failed=True,quality_read=False,C13_read=False,scoring_entries=0,qualified_for_scoring=False))
    print('C76 COMPLETE USED12 FIXTURE FROZEN; BOTH GL BRANCHES USE EXPLICIT LOADER',flush=True)


if __name__=='__main__':main()
