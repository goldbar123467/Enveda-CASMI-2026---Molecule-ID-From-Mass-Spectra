"""P13: proven P01 baseline plus public-source outsider blend consistency."""
import ast
import hashlib
import json
from pathlib import Path

from build_general_anchor import ROOT,replace_function,guarded_source


def build():
    checkpoint=ROOT/'outputs/research/v80-output/casmi_spectrum_model_ours_v80/final_model.pt'
    assert checkpoint.is_file()
    pin=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    nb=json.loads((ROOT/'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    swap='''def _swap_to_ours(base_bundle):
    _hits=[p for p in Path('/kaggle/input').rglob('final_model.pt')
           if 'casmi_spectrum_model_ours_v80' in str(p) and hashlib.sha256(p.read_bytes()).hexdigest()==MODEL_PIN]
    assert len(_hits)==1,'Exactly one byte-pinned P01 v80 scorer is required'
    _dst=ROOT/'ours_bundle';_dst.mkdir(exist_ok=True)
    for _name in ['model.py','features.py','config.json']:shutil.copy2(Path(base_bundle)/_name,_dst/_name)
    shutil.copy2(_hits[0],_dst/'final_model.pt')
    _audit=dict(checkpoint_sha256=MODEL_PIN,lineage='unchanged P01 v80',original_weight=.5,private_weight=.5)
    (ROOT/'p13_model_audit.json').write_text(json.dumps(_audit,indent=2))
    return str(_dst)'''.replace('MODEL_PIN',repr(pin))
    replace_function(nb,'_swap_to_ours',swap)
    scorer='''def score_final_lists():
    run_stage([sys.executable,CODE/'neural_scores.py',BUNDLE,'final_model.pt',
               ROOT/'frozen_candidate_evidence.parquet',ROOT/'effective_query_spectra.parquet',
               ROOT/'neural_candidate_features.parquet','cuda'],ROOT/'submission_neural.log','Required P01 scorer')
    return True'''
    replace_function(nb,'score_final_lists',scorer)
    added=0;patched=0
    helper=(ROOT/'scripts/outside_blend.py').read_text()
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        if "MODULES['neural_scores.py']" in source:
            source+='\nMODULES["outside_blend.py"]='+repr(helper)+'\nMODULE_SHA256["outside_blend.py"]='+repr(hashlib.sha256(helper.encode()).hexdigest())+'\n';added+=1
        trigger='OUTSIDERS_OK = score_outside_candidates(needed=True)'
        if trigger in source:
            addition='''
if OUTSIDERS_OK:
    shutil.copy2(ROOT/'outsider_neural_features.parquet',ROOT/'outsider_ours.parquet')
    run_stage([sys.executable,CODE/'neural_scores.py',PUB_BUNDLE,'final_model.pt',
               ROOT/'outsider_evidence.parquet',ROOT/'effective_query_spectra.parquet',
               ROOT/'outsider_public.parquet','cuda'],ROOT/'outsiders_public.log','Required public outsider scorer')
    from outside_blend import combine as _blend_outsiders
    _outside_a=pd.read_parquet(ROOT/'outsider_public.parquet')
    _outside_b=pd.read_parquet(ROOT/'outsider_ours.parquet')
    _blend_outsiders(_outside_a,_outside_b).to_parquet(ROOT/'outsider_neural_features.parquet',index=False)
    print('P13 OUTSIDERS public50 + v80-50',len(_outside_a),flush=True)
'''
            source=source.replace(trigger,trigger+'\n'+addition);patched+=1
        cell['source']=source.splitlines(keepends=True)
    assert added==patched==1
    seal='''_a=pd.read_parquet(ROOT/'neural_public.parquet').set_index(['molecule_id','smiles'])
_b=pd.read_parquet(ROOT/'neural_ours.parquet').set_index(['molecule_id','smiles']).reindex(_a.index)
_final=pd.read_parquet(ROOT/'neural_candidate_features.parquet').set_index(['molecule_id','smiles']).reindex(_a.index)
_error=float((_final.neural_score-(.5*_z(_a.neural_score.fillna(0))+.5*_z(_b.neural_score.fillna(0)))).abs().max())
assert _error<1e-6 and MERGE_REPORT.get('status')=='applied'
assert OUTSIDERS_OK,'Real outsider scoring is required for this trial'
from outside_blend import keyed as _keyed,z as _outside_z
_oa=_keyed(pd.read_parquet(ROOT/'outsider_public.parquet'))
_ob=_keyed(pd.read_parquet(ROOT/'outsider_ours.parquet')).reindex(_oa.index)
_of=_keyed(pd.read_parquet(ROOT/'outsider_neural_features.parquet')).reindex(_oa.index)
_ok=_oa.neural_ok.astype(bool)&_ob.neural_ok.astype(bool)
assert _ok.any()
_expected=.5*_outside_z(_oa.neural_score.fillna(0))+.5*_outside_z(_ob.neural_score.fillna(0))
_outside_error=float((_of.loc[_ok,'neural_score']-_expected.loc[_ok]).abs().max())
assert _outside_error<1e-6
_receipt=dict(model=json.loads((ROOT/'p13_model_audit.json').read_text()),
    neural_score_max_abs_error=_error,outside_blend_max_abs_error=_outside_error,
    real_scored_molecules=int(_a.loc[_a.neural_ok.astype(bool)].index.get_level_values(0).nunique()),
    outside_scored_molecules=int(_of.loc[_ok].index.get_level_values(0).nunique()),
    pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),merge_status=MERGE_REPORT['status'],is_hidden_rerun=IS_RERUN)
(ROOT/'p13_qualification_receipt.json').write_text(json.dumps(_receipt,indent=2))
print('P13 QUALIFIED',json.dumps(_receipt),flush=True)
'''
    nb['cells'].append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=seal.splitlines(keepends=True)))
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            source=''.join(cell['source']);compile(source,'p13','exec');cell['source']=guarded_source(source,f'p13_{i}').splitlines(keepends=True)
            cell['outputs']=[];cell['execution_count']=None
        cell.setdefault('id',hashlib.sha256((str(i)+''.join(cell['source'])).encode()).hexdigest()[:12])
    nb['nbformat_minor']=5
    nb['cells'].insert(0,dict(cell_type='markdown',id='p13-header',metadata={},source=[
        '# P13: P01 anchor with consistent outsider neural blending\n',
        'Keeps the verified0.480 P01 primary scorer and physical/prior/dedup settings.\n',
        'Tests the public peaks-v2 idea of scoring outsiders with the same public50/private50 neural blend.\n',
        'No new rank1 thresholds or unscored candidate fills. Hidden-score improvement is unproven.\n',
        'Upstream credits are retained. Source idea: https://www.kaggle.com/code/zz3t3rn41/casmi26-peaks-to-structures-v2\n']))
    out=ROOT/'kaggle/forks/p13-outside-blend';out.mkdir(parents=True,exist_ok=True)
    (out/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'kaggle/forks/p01-public480/kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-p13-outside-blend',title='casmi26 p13 outside blend',is_private=True)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (out/'model-pin.json').write_text(json.dumps(dict(checkpoint_sha256=pin,source='P01 v80',primary_recipe_unchanged=True,public_trial=1),indent=2))
    print('PACKAGED P13',pin)


if __name__=='__main__':build()
