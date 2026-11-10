"""Build query-only full240 P01 controls + one fixed public-recipe replay."""
import ast
import json
from datetime import datetime,timezone

import pandas as pd
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from build_c09_controls import unwrap

OUT=ROOT/'outputs/local/c29-public-replay-20261110'
PACKAGE=ROOT/'outputs/publish/c29-used240-casmi2022-20261110'
FORK=ROOT/'kaggle/forks/c29-public-replay'
INPUT=ROOT/'outputs/local/c21-negative-tail-20261110'
PLAN=ROOT/'configs/c29-public-replay-20261110.json'
PUBLIC=ROOT/'outputs/research/2026-11-10-silver-refresh/notebooks/alexchilton__casmi26-lb-0-487-peaks-to-structures-scorer/casmi26-lb-0-487-peaks-to-structures-scorer.ipynb'


def main():
    OUT.mkdir(exist_ok=True);PACKAGE.mkdir(exist_ok=True);FORK.mkdir(exist_ok=True)
    assert not (FORK/'validation-pin.json').exists(),'Existing build sealed'
    plan=json.loads(PLAN.read_text());template=ROOT/'kaggle/forks/c20-generated-recovery/notebook.ipynb'
    assert sha(PUBLIC)==plan['public_notebook_sha256'] and sha(template)==plan['template_notebook_sha256']
    prep=json.loads((INPUT/'preparation.json').read_text())
    for name in ['queries.parquet','parent.csv','P01.csv']:
        assert sha(INPUT/name)==prep['files_sha256'][name]
    q=pd.read_parquet(INPUT/'queries.parquet').drop(columns=['nm'],errors='ignore')
    assert len(q)==q.molecule_id.nunique()==240 and not {'key','smiles','formula','inchikey14'}&set(q.columns)
    assert q.ionization_mode.value_counts().to_dict()=={'positive':160,'negative':80}
    q.to_parquet(PACKAGE/'validation-queries.parquet',index=False)
    pd.DataFrame(dict(molecule_id=q.molecule_id,smiles='')).to_csv(PACKAGE/'validation-sample.csv',index=False)
    write(PACKAGE/'validation-contract.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),queries=240,
        development=240,confirmation=0,modes={'positive':160,'negative':80},truth_injection=False,
        source_doi='10.5281/zenodo.8371051',declared_license='CC-BY-4.0',query_labels_visible_to_pipeline=False,
        prediction_metrics_computed=False,all_queries_previously_used_development=True,
        original_input_sha256=sha(INPUT/'queries.parquet'),plan_sha256=sha(PLAN),
        files={n:sha(PACKAGE/n) for n in ['validation-queries.parquet','validation-sample.csv']}))
    write(PACKAGE/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c29-used240-casmi2022-20261110',
        title='CASMI26 C29 used240 development queries',licenses=[dict(name='CC-BY-4.0')]))
    (PACKAGE/'README.md').write_text('Private query-only union of already-used C10positive160/C20negative80 CASMI2022 development acquisitions, source Zenodo8371051 CC-BY-4.0/MassFormer attribution retained. No query structure labels or reserved C13 acquisitions. No public scoring submission.\n',encoding='utf-8')
    nb=json.loads(template.read_text(encoding='utf-8'));old='7f9a0c51de6a7613cb4a708d39608ca9429af182e3f8ae7c18b271ecbd93f14e'
    for i in [0,6,27]:
        source=unwrap(''.join(nb['cells'][i]['source']))
        source=source.replace(old,sha(PACKAGE/'validation-contract.json')).replace('c20_','c29_').replace('_c20t4','_c29t4').replace('C20 ','C29 ')
        source=source.replace('==80','==240').replace('queries=80','queries=240').replace('unlabelled80','unlabelled240')
        nb['cells'][i]['source']=guarded_source(source,'c29_controls_'+str(i)).splitlines(keepends=True)
    public=json.loads(PUBLIC.read_text(encoding='utf-8'))
    snippets={i:''.join(public['cells'][i]['source']) for i in [5,16,18,20,21,23,24]}
    assert all(public['cells'][i]['cell_type']=='code' for i in snippets)
    tree=ast.parse(snippets[24]);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_apex_extlib_overlay')
    overlay=ast.get_source_segment(snippets[24],fn)
    def replace(before,after):
        nonlocal overlay
        assert overlay.count(before)==1,before
        overlay=overlay.replace(before,after)
    replace('te = pd.read_parquet(test_paths[0])',"te = pd.read_parquet(COMP/'test.parquet')\n    assert te.molecule_id.nunique()==240")
    replace('    # Check Source B:', '    _c29_sourceA = len(refs)\n    # Check Source B:')
    replace('    if not refs:', "    _c29_sourceB = len(refs)-_c29_sourceA\n    assert _c29_sourceA>0 and _c29_sourceB>0, 'Both actual public libraries are required'\n    if not refs:")
    replace('    # Sort references by precursor mass for fast binary search',
        "    refs,_c29_alias = reference_view(refs,queries,_c29_features,_c29_exclude,raw_queries=te)\n    # Sort references by precursor mass for fast binary search")
    replace('audit = dict(molecules=len(sub), hits=hits_count, promoted=promoted_count, elapsed_s=elapsed)',
        'audit = dict(molecules=len(sub), hits=hits_count, promoted=promoted_count, elapsed_s=elapsed, sourceA_valid_rows=_c29_sourceA,sourceB_valid_rows=_c29_sourceB,alias_audit=_c29_alias,sourceB_casmi_subset_dropped=True)')
    compile(overlay,'c29_required_overlay','exec')
    alias=(ROOT/'scripts/c29_reference_audit.py').read_text(encoding='utf-8')
    post="""# One fixed changed recipe; labels never mounted or evaluated here.
assert not (C08_VALIDATION_SOURCE/'labels.csv').exists()
shutil.copy2(ROOT/'c29_neural_p01.parquet',ROOT/'neural_candidate_features.parquet')
exec(compile(C29_CONFIG,'c29_public_configuration','exec'),globals())
assert V8=='MASS' and MASS_ON and not RR8
CONSENSUS_OK,REPORT,MOVES=build_consensus();assert CONSENSUS_OK
SUBMITTED=choose_and_submit();assert SUBMITTED==VARIANT
shutil.copy2(ROOT/'submission.csv',ROOT/'submission_control_V2.csv')
for _i,_src in C29_POST:
    exec(compile(_src,'c29_public_post_'+str(_i),'exec'),globals())
assert PARALLAX_DIAGNOSTICS.get('status') not in ['skipped_due_to_runtime_budget']
assert not PARALLAX_DIAGNOSTICS.get('error')
assert MERGE_REPORT.get('status')=='applied'
assert MASS_REPORT['status']=='on' and MASS_LEAD_REPORT['status']=='ran'
assert RR8_REPORT['status']=='off in this preset'
shutil.copy2(ROOT/'submission.csv',ROOT/'c29_pre_external.csv')
import importlib.util
_spec=importlib.util.spec_from_file_location('_c29_actual_features',Path(PUB_BUNDLE)/'features.py')
_c29_features=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_c29_features)
exec(compile(C29_ALIAS,'c29_reference_audit','exec'),globals())
exec(compile(C29_OVERLAY,'c29_required_overlay','exec'),globals())
_c29_results={};_c29_id={}
_c29_parent=pd.read_csv(ROOT/'c29_pre_external.csv',dtype=str,keep_default_na=False)
def _c29_key(s):
    if s not in _c29_id:_c29_id[s]=_c29_features.structure(s)[0]
    assert _c29_id[s]
    return _c29_id[s]
for _c29_view,_c29_exclude in [('all_reference',False),('input_alias_excluded',True)]:
    shutil.copy2(ROOT/'c29_pre_external.csv',ROOT/'submission.csv')
    _apex_extlib_overlay()
    _raw=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
    assert _raw.molecule_id.tolist()==SAMPLE_IDS and _raw.molecule_id.is_unique
    _rows=[];_dedup=0
    for _row,_old in zip(_raw.itertuples(index=False),_c29_parent.itertuples(index=False)):
        _seen=set();_smiles=[]
        for _s in _row.smiles.split(';')+_old.smiles.split(';'):
            _k=_c29_key(_s)
            if _k not in _seen:_seen.add(_k);_smiles.append(_s)
            elif _s in _row.smiles.split(';'):_dedup+=1
            if len(_smiles)>=25:break
        assert 1<=len(_smiles)<=25 and len(_smiles)==len(_seen)
        _rows.append(dict(molecule_id=_row.molecule_id,smiles=';'.join(_smiles)))
    _path=ROOT/('c29_'+_c29_view+'.csv');pd.DataFrame(_rows).to_csv(_path,index=False)
    _audit=json.loads((ROOT/'apex_extlib_audit.json').read_text())
    _audit['metric_identity_dedup_events']=_dedup
    (ROOT/('c29_'+_c29_view+'_audit.json')).write_text(json.dumps(_audit,indent=2))
    _c29_results[_c29_view]=dict(predictions_sha256=hashlib.sha256(_path.read_bytes()).hexdigest(),audit=_audit)
_query=pd.read_parquet(COMP/'test.parquet')
_supported=_query.adduct.isin(['[M+H]+','[M-H]-','[M+Na]+','[M+NH4]+','[M+K]+','[M+CH2O2-H]-','[M+Cl]-'])
(ROOT/'c29_public_report.json').write_text(json.dumps(dict(queries=240,public_source_sha256=C29_PUBLIC_PIN,
    plan_sha256=C29_PLAN_PIN,controls_report_sha256=hashlib.sha256((ROOT/'c29_report.json').read_bytes()).hexdigest(),
    results=_c29_results,mass=MASS_REPORT,mass_lead=MASS_LEAD_REPORT,rr=RR8_REPORT,
    public_mass_adduct_supported=int(_supported.sum()),public_mass_adduct_unsupported=_query.loc[~_supported,'adduct'].value_counts().to_dict(),
    query_labels_mounted=False,metrics_computed=False,scoring_submission=False,
    sourceB_attribution='samartalwar MassBank harmonized CC-BY-NC-SA-4.0 compilation and original MassBank record terms; all is_casmi_subset rows excluded.',
    limitation='Documented input/formate/empty-pool compatibility and metric-identity output dedup differ from literal public notebook. Author score is not reproduced by this prediction-only run.'),indent=2))
(ROOT/'submission.csv').unlink()
print('DONE C29 FIXED PUBLIC REPLAY',json.dumps(_c29_results),flush=True)
"""
    constants=dict(C29_CONFIG=snippets[5],C29_POST=[(i,snippets[i]) for i in [16,18,20,21,23]],
        C29_ALIAS=alias,C29_OVERLAY=overlay,C29_PUBLIC_PIN=sha(PUBLIC),C29_PLAN_PIN=sha(PLAN))
    post='\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n'+post
    nb['cells'].append(dict(cell_type='markdown',metadata={},source=['C29 private fixed public-recipe validation. Attribution: Alex Chilton, guhongbin, popwenman, ozertuu, romanrozen, zz3t3rn41 and original pipeline authors; source/data notices retained. No public score claim.']))
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(post,'c29_fixed_public_replay').splitlines(keepends=True)))
    for c in nb['cells']:
        if c['cell_type']=='code':compile(unwrap(''.join(c['source'])),'c29_compilation','exec');c['execution_count']=None;c['outputs']=[]
    (FORK/'notebook.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8',newline='\n')
    meta=json.loads(template.with_name('kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-c29-public-replay-development',title='casmi26 C29 fixed public replay development',is_private=True,enable_internet=False)
    meta['dataset_sources']=[s.replace('clarkkitchen/casmi26-c20-negative-casmi2022-20261110','clarkkitchen/casmi26-c29-used240-casmi2022-20261110') for s in meta['dataset_sources']]
    meta['dataset_sources']+=['takumuhata/casmi26-extlib','samartalwar/casmi-2026-spectral-library-massbankharmonized']
    write(FORK/'kernel-metadata.json',meta)
    write(FORK/'validation-pin.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),notebook_sha256=sha(FORK/'notebook.ipynb'),
        builder_sha256=sha(__file__),public_notebook_sha256=sha(PUBLIC),template_notebook_sha256=sha(template),plan_sha256=sha(PLAN),
        input_contract_sha256=sha(PACKAGE/'validation-contract.json'),alias_helper_sha256=sha(ROOT/'scripts/c29_reference_audit.py'),
        full240_P01_controls_before_changed_stages=True,query_labels_mounted=False,metrics_computed=False,scoring_submission=False,
        unchanged_template_cells=[i for i in range(28) if i not in [0,6,27]],private=True,internet=False))
    print('BUILT C29 QUERY-ONLY FULL240 REPLAY',sha(FORK/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
