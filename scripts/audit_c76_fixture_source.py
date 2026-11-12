"""Independently verify the declared complete-fixture source delta before GPU."""
import ast,base64,copy
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp,unwrap,value,replace_value


def main():
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';out=base/'fixture-port'
    dest=out/'independent-source-audit.json';assert not dest.exists()
    pin=read(out/'fixture-freeze.json');old=ROOT/'outputs/local/c75-local-core-port-20261112/fixture-port'
    original=read(old/'notebook.ipynb');candidate=read(out/'notebook.ipynb')
    assert sha(out/'notebook.ipynb')==pin['notebook_sha256'] and sha(old/'notebook.ipynb')==pin['parent_notebook_sha256']
    assert len(original['cells'])==len(candidate['cells'])
    assert [i for i,(a,z) in enumerate(zip(original['cells'],candidate['cells'])) if a['source']!=z['source']]==[12]
    before=unwrap(''.join(original['cells'][12]['source']));after=unwrap(''.join(candidate['cells'][12]['source']))
    for name in ['FROZEN_B64','FROZEN_SHA256']:after=replace_value(after,name,value(before,name))
    assert after==before,'Outer orchestrator changed beyond the declared inner payload'
    original_inner=read(out/'c75-parent-inner.ipynb');inner=read(out/'c76-inner.ipynb')
    assert [i for i,(a,z) in enumerate(zip(original_inner['cells'],inner['cells'])) if a['source']!=z['source']]==[41]
    before=''.join(original_inner['cells'][41]['source']);after=''.join(inner['cells'][41]['source'])
    after=after.replace("GL_PKG = os.environ['CASMI_C76_GL_PACKAGE']","GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))")
    after=replace_value(after,'N1_GL_RUNNER_SHA256',value(before,'N1_GL_RUNNER_SHA256'))
    assert after==before,'Scientific branch/source rules changed beyond package path and its honest new runner hash'
    original_pkg=ROOT/'outputs/local/c75-local-core-port-20261112/input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    pkg=Path(pin['owned_GL_package']);assert sha(pkg/'gl_runner.py')==pin['candidate_gl_runner_sha256']
    assert sha(pkg/'c76_gl_portability.py')==pin['helper_sha256']==sha(ROOT/'scripts/c76_gl_portability.py')
    for item in original_pkg.iterdir():
        if item.name in ['gl_runner.py','__pycache__']:continue
        assert (pkg/item.name).is_symlink() and (pkg/item.name).resolve()==item.resolve(),item.name
    a=ast.parse((original_pkg/'gl_runner.py').read_text()).body;z=ast.parse((pkg/'gl_runner.py').read_text()).body
    dump=lambda node:ast.dump(node,include_attributes=False)
    assert [dump(n) for n in a[:-1]]==[dump(n) for n in z[:-3]]
    assert dump(a[-1])==dump(z[-1])
    assert isinstance(z[-3],ast.Assign) and z[-3].targets[0].id=='_C76_original_load_model'
    assert isinstance(z[-2],ast.FunctionDef) and z[-2].name=='load_model' and len(z[-2].args.args)==4
    assert sha(pkg/'ckpt/glacier.pt')==pin['original_checkpoint_sha256']
    write(dest,dict(completed_utc=stamp(),source_sha256=sha(__file__),fixture_freeze_sha256=sha(out/'fixture-freeze.json'),
        outer_change='Only re-encoded inner payload in cell12',inner_change='Cell41 GL package path and exact new GL runner hash only',
        all_original_GL_function_definitions_retained=True,only_added_GL_loader_wrapper=True,original_weight_and_asset_targets_exact=True,
        both_classic_and_N1_GL_use_new_loader=True,original_query_package_and_reference_masks_retained=True,
        original_C75_gate_passed=False,quality_read=False,C13_read=False,scoring_entries=0,passed=True))
    print('C76 COMPLETE FIXTURE INDEPENDENT SOURCE AUDIT PASSED',flush=True)


if __name__=='__main__':main()
