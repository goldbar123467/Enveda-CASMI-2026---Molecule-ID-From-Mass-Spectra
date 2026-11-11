"""Build private real-12 smoke and full352 reference-masked P01/P14 inference.

No scoring submission or quality metrics. Dataset packages contain queries,
global reference masks and attribution only; labels remain local.
"""
import ast,base64,hashlib,json,shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from build_c09_controls import unwrap
from build_c13_formate_recovery import value,replace_value
from c41_reference_guard import patch_engine2_source

OUT=ROOT/'outputs/local/c41-masked-pipeline-validation-20261111'
PREFIX='c41'
TEMPLATE=ROOT/'kaggle/forks/c20-generated-recovery/notebook.ipynb'
C39=ROOT/'outputs/local/c39-source-supported-validation-20261111'
QUERIES=ROOT/'outputs/publish/c39-source-supported-validation-20261111/validation-queries.parquet'


def build(mode,query_ids,helper,compat):
    count=len(query_ids);folder=ROOT/('kaggle/forks/'+PREFIX+'-'+mode+'-masked-validation')
    package=ROOT/('outputs/publish/'+PREFIX+'-'+mode+'-masked-validation-20261111')
    folder.mkdir(parents=True,exist_ok=True);package.mkdir(parents=True,exist_ok=True)
    assert not (folder/'validation-pin.json').exists(),'Immutable validation build exists'
    q=pd.read_parquet(QUERIES);q=q[q.molecule_id.isin(query_ids)].copy()
    assert q.molecule_id.nunique()==count and q.spectrum_id.is_unique
    q.to_parquet(package/'validation-queries.parquet',index=False)
    ids=list(dict.fromkeys(q.molecule_id));pd.DataFrame(dict(molecule_id=ids,smiles='CCO')).to_csv(package/'validation-sample.csv',index=False)
    for name in ['main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json']:(package/name).write_bytes((OUT/name).read_bytes())
    (package/'ATTRIBUTION.txt').write_text('Validation acquisitions derived privately from the official Enveda CASMI26 competition training data (Enveda-180 and Enveda natural-product examples) and primary MassBank2026.11 records.\n'
        'Enveda: https://github.com/enveda/enveda-180 and https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra\n'
        'MassBank primary archive: https://github.com/MassBank/MassBank-data/releases/tag/2026.11 ; commit72980b3af504cfd1e89ac54d7698d3da45d85a58. Original record attribution retained in the private project provenance ledger.\n'
        'This query-only diagnostic is not an unseen-by-all-models benchmark. Public pretraining exposure is disclosed. No query structure names or per-query identity labels are included.\n',encoding='utf8')
    files=['validation-queries.parquet','validation-sample.csv','main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json','ATTRIBUTION.txt']
    contract=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),queries=count,spectra=len(q),mode=mode,
        truth_injection=False,query_labels_visible_to_pipeline=False,reference_mask_required=True,
        source_query_freeze_sha256=sha(C39/'validation-freeze.json'),reference_exclusion_freeze_sha256=sha(OUT/'reference-exclusion-freeze.json'),
        controlled_in_corpus=True,pretrained_model_exposure_not_removed=True,quality_metrics_computed=False,
        files={n:sha(package/n) for n in files})
    write(package/'validation-contract.json',contract)
    dataset_id='clarkkitchen/casmi26-'+PREFIX+'-'+mode+'-masked-validation-20261111'
    write(package/'dataset-metadata.json',dict(id=dataset_id,title='CASMI26 C41 '+mode+' masked validation',licenses=[dict(name='CC-BY-4.0')]))
    nb=json.loads(TEMPLATE.read_text(encoding='utf8'));before=json.loads(json.dumps(nb))
    outer=unwrap(''.join(nb['cells'][12]['source']))
    frozen=json.loads(base64.b64decode(value(outer,'FROZEN_B64')));original=json.loads(json.dumps(frozen))
    changes={}
    for i,cell in enumerate(frozen['cells']):
        if cell['cell_type']!='code':continue
        s=''.join(cell['source'])
        marker="V4CODE = V43_DETERMINISM['work_source_root']"
        if marker in s:
            assert s.count(marker)==1
            injection='''
from c41_reference_guard import patch_legacy_tree as _c41_patch_legacy
V3CODE, _c41_compat_v3 = _c41_patch_legacy(V3CODE, '/kaggle/working/c41_legacy_v3_code')
V4CODE, _c41_compat_v4 = _c41_patch_legacy(V4CODE, '/kaggle/working/c41_legacy_v4_code')
Path('/kaggle/working/c41_legacy_compatibility.json').write_text(json.dumps(dict(v3=_c41_compat_v3,v4=_c41_compat_v4),indent=2))
'''
            s=s.replace(marker,marker+'\n'+injection);changes['legacy_source_boundary']=i
        if any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENG_FILES' for t in n.targets) for n in ast.parse(s).body):
            engine=value(s,'ENG_FILES');engine['casmi_engine.py']=patch_engine2_source(engine['casmi_engine.py']);engine['c41_reference_guard.py']=helper
            s=replace_value(s,'ENG_FILES',engine);changes['engine2_reference_index']=i
        marker="L = _external_reference_library(_EXTERNAL_REFERENCE_STATE); P = Pool('work')"
        if marker in s:
            assert s.count(marker)==1
            injection='''from c41_reference_guard import install_library_guard as _c41_install_library_guard
_c41_sid_mask_path = os.environ.get('CASMI_C41_SID_MASK')
assert _c41_sid_mask_path, 'C41 reference mask is mandatory'
_c41_install_library_guard(Library, np.load(_c41_sid_mask_path,allow_pickle=False))
'''
            s=s.replace(marker,injection+'\n'+marker);changes['main_reference_constructor']=i
        cell['source']=s.splitlines(keepends=True)
    assert set(changes)=={'legacy_source_boundary','engine2_reference_index','main_reference_constructor'},changes
    actual=[i for i in range(len(frozen['cells'])) if frozen['cells'][i]['source']!=original['cells'][i]['source']]
    assert set(actual)==set(changes.values())
    for c in frozen['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'c41_frozen_cell','exec')
    payload=json.dumps(frozen,ensure_ascii=False,indent=1).encode('utf8')
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(payload).decode())
    outer=replace_value(outer,'FROZEN_SHA256',hashlib.sha256(payload).hexdigest())
    export_suffix='''
_c41_original_export=export
def export(scope,root):
    _c41_original_export(scope,root)
    root=Path(root)
    library=scope['L']
    assert hasattr(library,'_c41_guard_audit'),'Main reference guard was not installed'
    audit=dict(library._c41_guard_audit)
    assert audit['excluded_reference_rows']>0 and audit['window_calls']>0 and audit['representative_calls']>0
    assert audit['forbidden_rows_retained']==0
    (root/'c41_main_reference_mask_audit.json').write_text(json.dumps(audit,indent=2))
    engine=json.loads(Path('/kaggle/working/c41_engine2_reference_mask_audit.json').read_text())
    assert engine['excluded_reference_rows']>0 and engine['forbidden_rows_retained']==0
    (root/'c41_engine2_reference_mask_receipt.json').write_text(json.dumps(engine,indent=2))
    assert Path('/kaggle/working/c41_legacy_compatibility.json').is_file()
'''
    extra='\nMODULES["c41_reference_guard.py"] = '+repr(helper)+'\nMODULES["legacy_formate_compat.py"] = '+repr(compat)+'\n'
    extra+='MODULES["export_evidence.py"] += '+repr(export_suffix)+'\n'
    extra+='import hashlib as _c41_module_hash\nfor _c41_name in ["c41_reference_guard.py","legacy_formate_compat.py","export_evidence.py"]:\n    MODULE_SHA256[_c41_name]=_c41_module_hash.sha256(MODULES[_c41_name].encode()).hexdigest()\n'
    outer+=extra
    nb['cells'][12]['source']=guarded_source(outer,'c41_reference_masked_payload').splitlines(keepends=True)
    oldpin='7f9a0c51de6a7613cb4a708d39608ca9429af182e3f8ae7c18b271ecbd93f14e'
    for i in [0,6,27]:
        source=unwrap(''.join(nb['cells'][i]['source']))
        source=source.replace(oldpin,sha(package/'validation-contract.json')).replace('c20_','c41_').replace('_c20t4','_c41t4').replace('C20 ','C41 ')
        source=source.replace('==80','=='+str(count)).replace('queries=80','queries='+str(count)).replace('unlabelled80','unlabelled'+str(count))
        if i==6:
            marker="os.environ['CASMI_C08_VALIDATION_COMP']=str(COMP)";assert source.count(marker)==1
            injection="\nos.environ['CASMI_C41_EXCLUSION_FILE']=str(C08_VALIDATION_SOURCE/'engine2-raw-key-exclusion.json')\nos.environ['CASMI_C41_SID_MASK']=str(C08_VALIDATION_SOURCE/'main-reference-sid-exclusion.npy')"
            source=source.replace(marker,marker+injection)
        nb['cells'][i]['source']=guarded_source(source,'c41_controls_'+str(i)).splitlines(keepends=True)
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),'c41_cell_'+str(i),'exec');c['outputs']=[];c['execution_count']=None
    changed=[i for i in range(len(nb['cells'])) if nb['cells'][i]['source']!=before['cells'][i]['source']]
    assert changed==[0,6,12,27],changed
    (folder/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8')
    metadata=json.loads(TEMPLATE.with_name('kernel-metadata.json').read_text())
    metadata['id']='clarkkitchen/casmi26-'+PREFIX+'-'+mode+'-masked-validation'
    metadata['title']='casmi26 '+PREFIX.upper()+' '+mode+' masked validation'
    metadata['is_private']=True
    metadata['dataset_sources']=[dataset_id if x=='clarkkitchen/casmi26-c20-negative-casmi2022-20261110' else x for x in metadata['dataset_sources']]
    write(folder/'kernel-metadata.json',metadata)
    pin=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=sha(__file__),template_sha256=sha(TEMPLATE),
        notebook_sha256=sha(folder/'notebook.ipynb'),query_contract_sha256=sha(package/'validation-contract.json'),
        frozen_scientific_payload_sha256=hashlib.sha256(payload).hexdigest(),frozen_changed_cells=changes,outer_changed_cells=changed,
        guard_source_sha256=sha(ROOT/'scripts/c41_reference_guard.py'),compat_source_sha256=sha(ROOT/'scripts/legacy_formate_compat.py'),
        mode=mode,queries=count,spectra=len(q),labels_mounted=False,private=True,internet=False,scoring_submission=False)
    write(folder/'validation-pin.json',pin)
    return pin


def main():
    import argparse
    global OUT,PREFIX
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default=PREFIX);parser.add_argument('--references',default=str(OUT));args=parser.parse_args()
    PREFIX=args.prefix;OUT=__import__('pathlib').Path(args.references)
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    ref=json.loads((OUT/'reference-exclusion-freeze.json').read_text())
    for n,h in ref['files_sha256'].items():assert sha(OUT/n)==h,n
    q=pd.read_parquet(QUERIES);lab=pd.read_csv(C39/'labels.csv').set_index('molecule_id')
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda x:'mixed' if x.nunique()>1 else x.iloc[0])
    smoke=[]
    for source in lab.source.unique():
        mids=sorted(lab.index[lab.source.eq(source)]);chosen=[]
        for mode in ['positive','mixed','negative']:
            options=[m for m in mids if modes.loc[m]==mode]
            if options:chosen.append(options[0])
        chosen+= [m for m in mids if m not in chosen][:4-len(chosen)]
        assert len(chosen)==4;smoke.extend(chosen)
    assert len(smoke)==12
    helper=(ROOT/'scripts/c41_reference_guard.py').read_text(encoding='utf8')
    compat=(ROOT/'scripts/legacy_formate_compat.py').read_text(encoding='utf8')
    a=build('smoke',smoke,helper,compat);b=build('full',lab.index.tolist(),helper,compat)
    assert a['frozen_scientific_payload_sha256']==b['frozen_scientific_payload_sha256']
    write(OUT/'validation-builds.json',dict(smoke=a,full=b,scientific_payload_exact_between_modes=True,metrics_read=False,C13_read=False))
    print('BUILT C41 PRIVATE VALIDATION',json.dumps(dict(smoke_queries=12,full_queries=352,scientific_payload_exact=True)),flush=True)


if __name__=='__main__':main()
