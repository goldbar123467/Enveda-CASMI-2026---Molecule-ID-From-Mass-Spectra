"""Prepare the full actual-T4 C76 source run; never launch or score it.

Original C73 remains the separate legacy control. Only its inner GL integration
changes here. The exact locally tested loader/helper bytes are materialized in
an owned package with original asset links. No fresh labels are mounted/read.
"""
import base64,copy,hashlib
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp,unwrap,value,replace_value


def clone_block(runner,helper):
    return '''
    # C76 explicit numerical candidate: original assets plus pinned loader/helper.
    import base64 as _c76_b64,hashlib as _c76_hash
    from pathlib import Path as _c76_Path
    _c76_original=_c76_Path(GL_PKG)
    assert _c76_hash.sha256((_c76_original/'gl_runner.py').read_bytes()).hexdigest()==ORIGINAL_SHA
    _c76_clone=_c76_Path('/kaggle/working/c76_glacier')
    assert not _c76_clone.exists(),'Fresh owned C76 GL package required'
    _c76_clone.mkdir()
    for _c76_item in _c76_original.iterdir():
        if _c76_item.name in ['gl_runner.py','__pycache__']:continue
        (_c76_clone/_c76_item.name).symlink_to(_c76_item.resolve(),target_is_directory=_c76_item.is_dir())
    (_c76_clone/'gl_runner.py').write_bytes(_c76_b64.b64decode(RUNNER_B64))
    (_c76_clone/'c76_gl_portability.py').write_bytes(_c76_b64.b64decode(HELPER_B64))
    assert _c76_hash.sha256((_c76_clone/'gl_runner.py').read_bytes()).hexdigest()==RUNNER_SHA
    assert _c76_hash.sha256((_c76_clone/'c76_gl_portability.py').read_bytes()).hexdigest()==HELPER_SHA
    for _c76_name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[_c76_name]='2'
    GL_PKG=str(_c76_clone)
    print('C76 EXPLICIT GL PACKAGE',dict(original_runner=ORIGINAL_SHA,adapter_runner=RUNNER_SHA,helper=HELPER_SHA,cpu_decision_threads=2),flush=True)
'''.replace('ORIGINAL_SHA',repr('5c4f74313ac54263d05d987c39d012d81ec57c5f35e1095ecd49a5bcb6440fca')).replace(
        'RUNNER_B64',repr(base64.b64encode(runner.read_bytes()).decode())).replace(
        'HELPER_B64',repr(base64.b64encode(helper.read_bytes()).decode())).replace(
        'RUNNER_SHA',repr(sha(runner))).replace('HELPER_SHA',repr(sha(helper)))


def main():
    auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_used']==0
    base=ROOT/'outputs/local/c76-numerical-portability-20261112'
    assert read(base/'extended-used-nine-r1/results.json')['component_passed']
    cpu=read(base/'cpu-feature-parity/independent-comparison.json')
    assert all(v['exact'] for v in cpu['array_comparisons'].values()) and len(cpu['array_comparisons'])==26
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    original=ROOT/'kaggle/forks/c73-full-masked-validation/notebook.ipynb'
    nb=read(original);prior=copy.deepcopy(nb);outer=unwrap(''.join(nb['cells'][12]['source']))
    payload=base64.b64decode(value(outer,'FROZEN_B64'))
    assert hashlib.sha256(payload).hexdigest()==value(outer,'FROZEN_SHA256')=='3f973abbb1593e356db4f6dec149c8ad99f5afa5ded8da8bab59152bd49e60e7'
    inner=__import__('json').loads(payload);before=copy.deepcopy(inner)
    native_pin=read(base/'fixture-port/fixture-freeze.json');pkg=Path(native_pin['owned_GL_package'])
    # Windows plan preparation also works without importing any ML extensions.
    if not pkg.exists():
        runner_bytes=__import__('subprocess').check_output(['wsl.exe','-d','Ubuntu-24.04','--','cat',native_pin['owned_GL_package']+'/gl_runner.py'])
        archive=base/'cloud-validation-preparation';assert not archive.exists();archive.mkdir()
        runner=archive/'pinned-gl_runner.py';runner.write_bytes(runner_bytes)
    else:
        archive=base/'cloud-validation-preparation';assert not archive.exists();archive.mkdir();runner=pkg/'gl_runner.py'
    helper=ROOT/'scripts/c76_gl_portability.py'
    assert sha(runner)==native_pin['candidate_gl_runner_sha256'] and sha(helper)==native_pin['helper_sha256']
    source=''.join(inner['cells'][41]['source'])
    needle="    GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))"
    assert source.count(needle)==1
    addition=clone_block(runner,helper);source=source.replace(needle,needle+addition)
    source=replace_value(source,'N1_GL_RUNNER_SHA256',sha(runner))
    inner['cells'][41]['source']=source.splitlines(keepends=True)
    assert [i for i,(a,z) in enumerate(zip(before['cells'],inner['cells'])) if a['source']!=z['source']]==[41]
    candidate=__import__('json').dumps(inner,ensure_ascii=False,indent=1).encode()
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(candidate).decode())
    outer=replace_value(outer,'FROZEN_SHA256',hashlib.sha256(candidate).hexdigest())
    nb['cells'][12]['source']=outer.splitlines(keepends=True)
    assert [i for i,(a,z) in enumerate(zip(prior['cells'],nb['cells'])) if a['source']!=z['source']]==[12]
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),'c76_cloud_outer_'+str(i),'exec');cell['outputs']=[];cell['execution_count']=None
    folder=ROOT/'kaggle/forks/c76-full-masked-validation';assert not folder.exists();folder.mkdir()
    write(folder/'notebook.ipynb',nb)
    meta=read(original.with_name('kernel-metadata.json'));meta.update(id='clarkkitchen/casmi26-c76-full-masked-validation',title='casmi26 C76 full masked validation')
    write(folder/'kernel-metadata.json',meta)
    (archive/'original-inner.ipynb').write_bytes(payload);(archive/'candidate-inner.ipynb').write_bytes(candidate)
    write(archive/'clone-addition.txt',addition)
    files=[original,folder/'notebook.ipynb',folder/'kernel-metadata.json',helper,ROOT/'configs/c76-source-controls-addendum-v1.json',
        ROOT/'outputs/local/c71-final-clean-direct-pair-20261112/complete-pipeline-binding.json',
        ROOT/'outputs/local/c72-clean-target-holdout-20261112/preparation-protocol.json']
    write(folder/'validation-pin.json',dict(prepared_utc=stamp(),source_sha256=sha(__file__),queries=512,acquisitions=2215,
        original_payload_sha256=hashlib.sha256(payload).hexdigest(),candidate_payload_sha256=hashlib.sha256(candidate).hexdigest(),
        original_C73_notebook_sha256=sha(original),notebook_sha256=sha(folder/'notebook.ipynb'),metadata_sha256=sha(folder/'kernel-metadata.json'),
        original_GL_runner_sha256='5c4f74313ac54263d05d987c39d012d81ec57c5f35e1095ecd49a5bcb6440fca',
        candidate_GL_runner_sha256=sha(runner),helper_sha256=sha(helper),clone_addition_sha256=sha(archive/'clone-addition.txt'),
        exact_original_C73_source_datasets_and_query_contract_retained=True,changed_outer_cells=[12],changed_inner_cells=[41],
        original_ICE_default_retained_on_actual_T4=True,local_only_resource_adapter_not_shipped_to_T4=True,
        original_C73_required_as_separate_legacy_control=True,complete_C76_pipeline_binding_still_required=True,
        files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        quality_read=False,C13_read=False,scoring_entries=0,release_qualified=False,private_source_inference_only=True))
    print('C76 ACTUAL-T4 FULL512 SOURCE NOTEBOOK PREPARED; NOT RELEASED OR SCORED',flush=True)


if __name__=='__main__':main()
