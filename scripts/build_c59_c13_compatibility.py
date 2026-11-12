"""Prepare, but do NOT release, the faithful C13 main/fragment repair.

No reserved per-query labels or prediction quality are read. All120original
queries and their existing query-only contract remain unchanged.
"""
import ast,base64,copy,hashlib,json
from c45_complete_parent import ROOT,read,write,sha,stamp
from build_c13_formate_recovery import value,replace_value
from build_c45_parent_ablation import unwrap,scientific
from build_general_anchor import guarded_source
OUT=ROOT/'outputs/local/c59-c13-compatibility-20261112'
FOLDER=ROOT/'kaggle/forks/c59-c13-main-compatibility'


def main():
    OUT.mkdir(exist_ok=True);assert not FOLDER.exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    old=ROOT/'kaggle/forks/c13-formate-recovery';oldpin=read(old/'recovery-pin.json')
    assert sha(old/'notebook.ipynb')==oldpin['notebook_sha256']
    notebook=read(old/'notebook.ipynb');original=copy.deepcopy(notebook)
    outer=unwrap(''.join(notebook['cells'][12]['source']));frozen=json.loads(base64.b64decode(value(outer,'FROZEN_B64')));before=copy.deepcopy(frozen)
    helper=(ROOT/'scripts/c41_reference_guard.py').read_text(encoding='utf8')
    node=next(n for n in ast.parse(helper).body if isinstance(n,ast.FunctionDef) and n.name=='patch_legacy_tree')
    patcher='import hashlib\nfrom pathlib import Path\n\n'+ast.get_source_segment(helper,node)+'\n'
    compat=(ROOT/'scripts/legacy_formate_compat.py').read_text(encoding='utf8')
    code=''.join(frozen['cells'][27]['source']);needle="V4CODE = V43_DETERMINISM['work_source_root']"
    assert code.count(needle)==1 and 'patch_legacy_tree' not in code
    injection="""
from c59_legacy_copy import patch_legacy_tree as _c59_patch_legacy
V3CODE, _c59_compat_v3 = _c59_patch_legacy(V3CODE, '/kaggle/working/c59_legacy_v3_code')
V4CODE, _c59_compat_v4 = _c59_patch_legacy(V4CODE, '/kaggle/working/c59_legacy_v4_code')
Path('/kaggle/working/c59_legacy_compatibility.json').write_text(json.dumps(dict(v3=_c59_compat_v3,v4=_c59_compat_v4),indent=2))
"""
    frozen['cells'][27]['source']=code.replace(needle,needle+'\n'+injection).splitlines(keepends=True)
    assert all(frozen['cells'][i]['source']==before['cells'][i]['source'] for i in range(len(frozen['cells'])) if i!=27)
    assert frozen['cells'][32]['source']==before['cells'][32]['source'],'PV-v2repair must remain exact'
    payload=json.dumps(frozen,ensure_ascii=False,indent=1).encode('utf8')
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(payload).decode());outer=replace_value(outer,'FROZEN_SHA256',hashlib.sha256(payload).hexdigest())
    for name,text in [('c59_legacy_copy.py',patcher),('legacy_formate_compat.py',compat)]:
        compile(text,name,'exec');outer+='\nMODULES['+repr(name)+']='+repr(text)+'\nMODULE_SHA256['+repr(name)+']='+repr(hashlib.sha256(text.encode()).hexdigest())+'\n'
    notebook['cells'][12]['source']=guarded_source(outer,'c59_faithful_main_fragment_compatibility').splitlines(keepends=True)
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),'c59_'+str(i),'exec');cell['outputs']=[];cell['execution_count']=None
    assert all(notebook['cells'][i]['source']==original['cells'][i]['source'] for i in range(len(notebook['cells'])) if i!=12)
    FOLDER.mkdir();(FOLDER/'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
    metadata=read(old/'kernel-metadata.json');metadata.update(id='clarkkitchen/casmi26-c59-c13-main-compatibility',title='casmi26 C59 faithful C13 main compatibility')
    assert metadata['is_private'] and not metadata['enable_internet'] and metadata['machine_shape']=='NvidiaTeslaT4'
    write(FOLDER/'kernel-metadata.json',metadata)
    query=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet';contract=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-contract.json'
    preflight=read(ROOT/'outputs/local/c52-unused-target-inventory-20261112/C13-query-only-compatibility-preflight.json')
    write(FOLDER/'compatibility-pin.json',dict(prepared_utc=stamp(),source_sha256=sha(__file__),old_v2_notebook_sha256=sha(old/'notebook.ipynb'),
        notebook_sha256=sha(FOLDER/'notebook.ipynb'),scientific_sha256=scientific(notebook),frozen_scientific_payload_sha256=hashlib.sha256(payload).hexdigest(),
        changed_outer_cells=[12],changed_inner_cells=[27],PV_v2_repair_unchanged=True,all_other_scientific_cells_exact=True,
        source_copy_function_sha256=hashlib.sha256(patcher.encode()).hexdigest(),compat_adapter_sha256=sha(ROOT/'scripts/legacy_formate_compat.py'),
        query_sha256=sha(query),query_contract_sha256=sha(contract),queries=120,query_modes=dict(positive=60,negative=60),
        query_tokens_or_selection_changed=False,no_new_reference_intervention=True,query_labels_mounted=False,prediction_quality_read=False,
        main_scalar_preflight=dict(old_finite=preflight['old_main_finite_masses'],patched_finite=preflight['adapter_main_finite_masses'],
            note='One original trailing-space formate remains unsupported by the exact production main adapter; retain it and its Engine2fallback. No query rewrite or favorable omission.'),
        release_authorized_only_after='C56actualP30/P31production audits AND C58fresh482full/incremental/source/mode admission. This build alone cannot push, submit, consume C13or clear scientific hold.',
        private=True,internet=False,scoring_submission=False,cloud_run_released=False))
    print('C59 FAITHFUL C13COMPATIBILITY PREPARED ONLY',sha(FOLDER/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
