"""Stage competition-path production source from C29, without pushing or sealing."""
import ast
import base64
import hashlib
import json
from datetime import datetime,timezone

import pandas as pd
from precision_cache import ROOT,sha,write,features
from build_c09_controls import unwrap
from build_general_anchor import guarded_source
from build_c13_formate_recovery import value,replace_value,neutral_function
from submit_c09_external_once import scientific_sha
from public_deployment_support import intake

OUT=ROOT/'outputs/local/c33-public-production-20261110'
FORK=ROOT/'kaggle/forks/p17-public-mass-dualref'
TEMPLATE=ROOT/'kaggle/forks/c29-public-replay/notebook.ipynb'
PUBLIC=ROOT/'outputs/research/2026-11-10-silver-refresh/notebooks/alexchilton__casmi26-lb-0-487-peaks-to-structures-scorer/casmi26-lb-0-487-peaks-to-structures-scorer.ipynb'


def main():
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert not (OUT/'production-source-freeze.json').exists(),'Production source sealed; do not mutate'
    assert not (OUT/'push-intent.json').exists(),'Never replace an active/deployed notebook'
    assert sha(TEMPLATE)=='02304e42bc27cbbad98b2995333ccf6d16e1599838d1de3eef13c964d85db428'
    assert sha(PUBLIC)=='380e2c820c2a47cfad268b75cdd0dfc564350b599b9b052a1915d292987113c5'
    OUT.mkdir(exist_ok=True);FORK.mkdir(exist_ok=True)
    nb=json.loads(TEMPLATE.read_text(encoding='utf-8'));original=json.loads(TEMPLATE.read_text(encoding='utf-8'))
    public=json.loads(PUBLIC.read_text(encoding='utf-8'))
    module=unwrap(''.join(nb['cells'][12]['source']))
    frozen=json.loads(base64.b64decode(value(module,'FROZEN_B64')));prior=json.loads(json.dumps(frozen))
    public_module=''.join(public['cells'][12]['source'])
    public_frozen=json.loads(base64.b64decode(value(public_module,'FROZEN_B64')))
    s=''.join(frozen['cells'][26]['source']);needle="COMP = os.environ['CASMI_C08_VALIDATION_COMP']\n";assert s.count(needle)==1
    frozen['cells'][26]['source']=s.replace(needle,'').splitlines(keepends=True)
    before=''.join(frozen['cells'][28]['source']);after=''.join(public_frozen['cells'][28]['source'])
    needle="assert comp == os.path.realpath(os.environ['CASMI_C08_VALIDATION_COMP'])"
    assert before.count(needle)==1
    expected=before.replace(needle,"assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}")
    assert expected==after,'Only production path preflight may differ'
    frozen['cells'][28]['source']=after.splitlines(keepends=True)
    assert all(frozen['cells'][i]['source']==prior['cells'][i]['source'] for i in range(len(frozen['cells'])) if i not in [26,28])
    engine_files=value(''.join(frozen['cells'][32]['source']),'ENG_FILES');_,adducts=neutral_function(engine_files['pv.py'])
    known=sorted(adducts)
    # Show the safety guard adds no exclusion on either already fixed cohort.
    intake_reports={}
    for name,folder in [('development240',ROOT/'outputs/local/c21-negative-tail-20261110'),('reserved120',ROOT/'outputs/local/c25-reserved-negative-20261110')]:
        parent=pd.read_csv(folder/('parent.csv' if name=='development240' else 'P01.csv'),dtype=str,keep_default_na=False)
        q=pd.read_parquet(folder/'queries.parquet');eligible,audit=intake(parent,q,features,known)
        assert len(eligible)==len(parent),audit.reason.value_counts().to_dict()
        intake_reports[name]=dict(queries=len(parent),eligible=len(eligible),reasons=audit.reason.value_counts().to_dict(),labels_used=False)
    payload=json.dumps(frozen,ensure_ascii=False,indent=1).encode()
    module=replace_value(module,'FROZEN_B64',base64.b64encode(payload).decode())
    module=replace_value(module,'FROZEN_SHA256',hashlib.sha256(payload).hexdigest())
    nodes=[n for n in ast.parse(module).body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="MODULES['runner.py']"]
    assert len(nodes)==2,'Original runner plus the explicit validation override are required'
    runner=ast.literal_eval(nodes[-1].value);old="mode='external_validation',source_adaptations=['fixed_external_test_path','external_input_preflight']"
    assert runner.count(old)==1
    runner=runner.replace(old,"mode='submission',source_adaptations=['formate_neutral_mass_alias','real_empty_engine2_pool_recovery']")
    statement=ast.get_source_segment(module,nodes[-1]);module=module.replace(statement,"MODULES['runner.py'] = "+repr(runner))
    hash_nodes=[n for n in ast.parse(module).body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="MODULE_SHA256['runner.py']"]
    assert len(hash_nodes)==1
    module=module.replace(ast.get_source_segment(module,hash_nodes[0]),"MODULE_SHA256['runner.py'] = "+repr(hashlib.sha256(runner.encode()).hexdigest()))
    nb['cells'][12]['source']=guarded_source(module,'c33_production_payload').splitlines(keepends=True)
    nb['cells'][6]['source']=guarded_source(''.join(public['cells'][6]['source']),'c33_competition_session').splitlines(keepends=True)
    config=unwrap(''.join(nb['cells'][5]['source']));assert value(config,'COMMIT_MODE')=='smoke'
    preamble="""import os,json,torch
from pathlib import Path
torch.set_num_threads(2)
_c33_devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]
assert len(_c33_devices)==2 and all(d['name']=='Tesla T4' for d in _c33_devices)
Path('/kaggle/working/c33_t4_allocation.json').write_text(json.dumps(dict(devices=_c33_devices),indent=2))
os.environ['CUDA_VISIBLE_DEVICES']='0'
"""
    nb['cells'][0]['source']=guarded_source(preamble,'c33_actual_t4_allocation').splitlines(keepends=True)
    # Preserve the entire P01 prefix, remove the validation-only P14 exporter.
    nb['cells']=nb['cells'][:27]
    deployment_sources={n:(ROOT/'scripts'/n).read_text(encoding='utf-8') for n in
        ['apply_external_reference.py','external_reference_overlay.py','c09_spectrum_audit.py','reference_model_audit.py','public_deployment_support.py']}
    deployment_sources['c09e_features.py']=(ROOT/'outputs/local/kit/features.py').read_text(encoding='utf-8')
    policy=ROOT/'configs/c09-external-reference-20261109.json'
    dev=json.loads((ROOT/'outputs/local/c09-external-reference-20261109/development.json').read_text())
    parent_contract=dict(source_files={n:hashlib.sha256(s.encode()).hexdigest() for n,s in deployment_sources.items()},
        policy_sha256=sha(policy),reference_files=dev['reference_sha256'],development_evidence_sha256=sha(ROOT/'outputs/local/c09-external-reference-20261109/development.json'))
    reference=ROOT/'outputs/research/2026-11-09-silver/extlib'
    ref_pins=dict(parent_contract['reference_files'],**{'ext_mass.parquet':sha(reference/'ext_mass.parquet')})
    massbank=ROOT/'outputs/research/2026-11-09-silver/massbank-coverage-audit/spectra.parquet'
    ref_pins['spectra.parquet']=sha(massbank)
    core=unwrap(''.join(original['cells'][29]['source']))
    overlay=value(core,'C29_OVERLAY')
    needle="te = pd.read_parquet(COMP/'test.parquet')\n    assert te.molecule_id.nunique()==240";assert overlay.count(needle)==1
    overlay=overlay.replace(needle,"te = pd.read_parquet(C33_OVERLAY_QUERY_PATH)\n    assert te.molecule_id.nunique()==C33_ELIGIBLE_COUNT")
    core=replace_value(core,'C29_OVERLAY',overlay)
    row_loop=next(n for n in ast.parse(core).body if isinstance(n,ast.For) and ast.unparse(n.target)=='(_c29_view, _c29_exclude)')
    nested=next(n for n in row_loop.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(_row, _old)')
    block='_rows=[];_dedup=0\n    '+ast.get_source_segment(core,nested)
    assert core.count(block)==1
    core=core.replace(block,"_unique,_identity_report=_c33_finalize(_raw,_c29_parent,_c29_key,C33_FALLBACK_PARENT)\n    _rows=_unique.to_dict('records');_dedup=_identity_report['metric_identity_dedup_events']")
    before='    assert _c29_id[s]';assert core.count(before)==1
    core=core.replace(before,"    if not _c29_id[s]:raise ValueError('Missing official identity')")
    before="_audit['metric_identity_dedup_events']=_dedup";assert core.count(before)==1
    core=core.replace(before,"_audit.update(_identity_report)")
    before="assert not (C08_VALIDATION_SOURCE/'labels.csv').exists()\n";assert core.count(before)==1;core=core.replace(before,'')
    core=core.replace('queries=240','queries=C33_ELIGIBLE_COUNT')
    before="controls_report_sha256=hashlib.sha256((ROOT/'c29_report.json').read_bytes()).hexdigest()"
    assert core.count(before)==1;core=core.replace(before,"base_execution_receipt_sha256=hashlib.sha256((ROOT/'execution_receipt.json').read_bytes()).hexdigest()")
    before="(ROOT/'submission.csv').unlink()";assert core.count(before)==1
    core=core.replace(before,"shutil.copy2(ROOT/'c29_all_reference.csv',ROOT/'submission.csv')")
    compile(core,'c33_public_core','exec')
    ending="""# Required production final stage; no validation labels are attached.
assert MERGE_REPORT.get('status')=='applied'
_c33_code=ROOT/'c33_code';_c33_code.mkdir(exist_ok=True)
for _name,_text in C33_SOURCES.items():(_c33_code/_name).write_text(_text,encoding='utf-8')
(_c33_code/'c09e_policy.json').write_bytes(C33_POLICY)
(_c33_code/'deployment.json').write_text(json.dumps(C33_PARENT_CONTRACT,indent=2))
sys.path.insert(0,str(_c33_code))
from public_deployment_support import intake as _c33_intake,protect as _c33_protect,finalize as _c33_finalize
from reference_model_audit import verify as _c33_verify_models
_c33_models=_c33_verify_models(PUB_BUNDLE,OURS_BUNDLE,MODEL_PINS,'c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46')
_c33_qpath=ROOT/'effective_query_spectra.parquet'
_c33_all_queries=pd.read_parquet(_c33_qpath)
_c33_p01=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
_c33_eligible,_c33_intake_report=_c33_intake(_c33_p01,_c33_all_queries,__import__('c09e_features'),C33_KNOWN_ADDUCTS)
_c33_intake_report.to_csv(ROOT/'c33_intake.csv',index=False)
shutil.copy2(ROOT/'submission.csv',ROOT/'c33_p01.csv')
shutil.copy2(ROOT/'neural_candidate_features.parquet',ROOT/'c29_neural_p01.parquet')
_c33_supported=_c33_all_queries[_c33_all_queries.molecule_id.astype(str).isin(_c33_eligible)].copy()
C33_OVERLAY_QUERY_PATH=ROOT/'c33_supported_queries.parquet';_c33_supported.to_parquet(C33_OVERLAY_QUERY_PATH,index=False)
C33_ELIGIBLE_COUNT=len(_c33_eligible)
for _name,_pin in C33_REFERENCE_PINS.items():
    _hits=list(INPUT.rglob(_name))
    if _name=='spectra.parquet':_hits=[p for p in _hits if 'massbank' in str(p).lower()]
    assert len(_hits)==1 and hashlib.sha256(_hits[0].read_bytes()).hexdigest()==_pin,_name
_c33_qpath.rename(ROOT/'c33_original_queries.parquet')
try:
    _c33_supported.to_parquet(_c33_qpath,index=False)
    if _c33_eligible:
        run_stage([sys.executable,_c33_code/'apply_external_reference.py','--work',ROOT,'--input-root',INPUT,'--contract',_c33_code/'deployment.json'],ROOT/'c33_parent.log','Pinned P16 parent')
        _c33_parent_receipt=json.loads((ROOT/'c09e_external_receipt.json').read_text())
        assert _c33_parent_receipt['required_stage_completed'] and _c33_parent_receipt['official_identity_duplicates']==0
        _c33_p16=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
    else:
        _c33_p16=_c33_p01.copy();_c33_parent_receipt=dict(skipped='all queries unsupported',protected_rows=len(_c33_p16))
    _c33_p16.to_csv(ROOT/'c33_p16_parent.csv',index=False)
finally:
    _c33_qpath.unlink(missing_ok=True);(ROOT/'c33_original_queries.parquet').rename(_c33_qpath)
_c33_original_comp=COMP
C33_FALLBACK_PARENT=_c33_p16
try:
    if _c33_eligible:
        _c33_comp=ROOT/'c33_supported_comp';_c33_comp.mkdir(exist_ok=True)
        _c33_train=_c33_comp/'train.parquet'
        if not _c33_train.exists():_c33_train.symlink_to((COMP/'train.parquet').resolve())
        _c33_supported.to_parquet(_c33_comp/'test.parquet',index=False)
        shutil.copy2(COMP/'sample_submission.csv',_c33_comp/'sample_submission.csv')
        COMP=_c33_comp
        exec(compile(C33_PUBLIC_CORE,'c33_fixed_public_post','exec'),globals())
        for _view in ['all_reference','input_alias_excluded']:
            _raw=pd.read_csv(ROOT/('c29_'+_view+'.csv'),dtype=str,keep_default_na=False)
            _protected=_c33_protect(_c33_p16,_raw,_c33_eligible)
            _protected.to_csv(ROOT/('c33_'+_view+'.csv'),index=False)
    else:
        for _view in ['all_reference','input_alias_excluded']:_c33_p16.to_csv(ROOT/('c33_'+_view+'.csv'),index=False)
finally:COMP=_c33_original_comp
shutil.copy2(ROOT/'c33_all_reference.csv',ROOT/'submission.csv')
_c33_final=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
assert _c33_final.molecule_id.tolist()==SAMPLE_IDS and _c33_final.molecule_id.is_unique
assert _c33_final.smiles.str.split(';').map(len).between(1,25).all()
_c33_execution=json.loads((ROOT/'execution_receipt.json').read_text())
assert _c33_execution['is_full_run']==IS_RERUN
assert _c33_execution['processed_molecules']==_c33_all_queries.molecule_id.nunique()
if not IS_RERUN:assert _c33_execution['processed_molecules']==12
(ROOT/'c33_execution_receipt.json').write_text(json.dumps(dict(kind='C29_fixed_public_mass_dualref_production',
    pipeline=_c33_execution,models=_c33_models,P16_parent=_c33_parent_receipt,reference_sha256=C33_REFERENCE_PINS,
    total_rows=len(_c33_final),real_queries=int(_c33_all_queries.molecule_id.nunique()),eligible_queries=C33_ELIGIBLE_COUNT,
    intake_reasons=_c33_intake_report.reason.value_counts().to_dict(),selected_reference_view='all_reference',
    unsupported_parent_rows_exact=int((~_c33_p16.molecule_id.isin(_c33_eligible)).sum()),
    csv_sha256=hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest(),
    source_deployment_sha256=C33_SOURCE_PINS,quality_metrics_computed=False,automatic_submission=False),indent=2))
print('DONE C33 PRODUCTION SOURCE EXECUTION',flush=True)
"""
    constants=dict(C33_SOURCES=deployment_sources,C33_POLICY=policy.read_bytes(),C33_PARENT_CONTRACT=parent_contract,
        C33_REFERENCE_PINS=ref_pins,C33_KNOWN_ADDUCTS=known,C33_PUBLIC_CORE=core,
        C33_SOURCE_PINS={n:hashlib.sha256(s.encode()).hexdigest() for n,s in deployment_sources.items()})
    ending='\n'.join(n+'='+repr(v) for n,v in constants.items())+'\n'+ending
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(ending,'c33_required_fixed_public_recipe').splitlines(keepends=True)))
    for c in nb['cells']:
        if c['cell_type']=='code':compile(unwrap(''.join(c['source'])),'c33_build','exec');c['execution_count']=None;c['outputs']=[]
    nb['cells'].insert(0,dict(cell_type='markdown',metadata={},source=['P17/C33 staged production source. Same fixed C29 recipe, competition paths, declared input safeguards and pinned P16 parent. Not qualified, pushed or submitted. Credits retained below.']))
    (FORK/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
    meta=json.loads(TEMPLATE.with_name('kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-p17-public-mass-dualref',title='casmi26 p17 public mass dualref',is_private=True,enable_internet=False)
    meta['dataset_sources']=[s for s in meta['dataset_sources'] if s not in ['clarkkitchen/casmi26-c29-used240-casmi2022-20261110','clarkkitchen/casmi26-frozen-p14-20261107']]
    write(FORK/'kernel-metadata.json',meta)
    write(OUT/'staging-audit.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),notebook_sha256=sha(FORK/'notebook.ipynb'),
        scientific_sha256=scientific_sha(nb),builder_sha256=sha(__file__),validation_template_sha256=sha(TEMPLATE),
        frozen_payload_changed_cells=[26,28],formate_and_empty_pool_recovery_exact=True,coverage_guard_exact=True,
        deployment_intake=intake_reports,reference_sha256=ref_pins,private=True,internet=False,
        production_source_sealed=False,independent_full_source_audit_pending=True,required_changed_path_fixture_pending=True,
        mandatory_actual_Kaggle_smoke=True,quality_metrics_read=False,automatic_submission=False))
    print('STAGED C33 PRODUCTION SOURCE; NOT SEALED/PUSHED',sha(FORK/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
