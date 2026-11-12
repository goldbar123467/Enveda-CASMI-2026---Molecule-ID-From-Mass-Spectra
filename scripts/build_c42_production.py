"""Freeze CPU/GPU production notebooks before source or reserved outcomes.

The parent differs from P16 only by the tested legacy formate source adapter.
No validation-reference masks or query labels are mounted in production.
"""
import base64,hashlib,inspect,json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from build_c09_controls import unwrap
from build_c13_formate_recovery import value,replace_value
from c41_reference_guard import patch_legacy_tree
from submit_c09_external_once import scientific_sha


def main():
    parent=ROOT/'kaggle/forks/p16-external-reference-c09/notebook.ipynb'
    assert sha(parent)=='d6b9c9625ad8fb7d503341c6fd702bd909101d719ca55e400145fb96f2b2af6a'
    bundle=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111'
    manifest=json.loads((bundle/'head-manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(bundle/n)==h,n
    assert not manifest['quality_read'] and not manifest['C13_read']
    helper='from pathlib import Path\nimport hashlib\n'+inspect.getsource(patch_legacy_tree)
    compat=(ROOT/'scripts/legacy_formate_compat.py').read_text(encoding='utf8')
    original=json.loads(parent.read_text(encoding='utf8'));prepared=json.loads(json.dumps(original))
    outer=unwrap(''.join(prepared['cells'][12]['source']))
    frozen=json.loads(base64.b64decode(value(outer,'FROZEN_B64')));prior=json.loads(json.dumps(frozen));changed=[]
    for i,c in enumerate(frozen['cells']):
        if c['cell_type']!='code':continue
        s=''.join(c['source']);needle="V4CODE = V43_DETERMINISM['work_source_root']"
        if needle in s:
            assert s.count(needle)==1
            injected='''
from legacy_runtime_compat import patch_legacy_tree as _c42_patch_legacy
V3CODE,_c42_v3=_c42_patch_legacy(V3CODE,'/kaggle/working/c42_legacy_v3_code')
V4CODE,_c42_v4=_c42_patch_legacy(V4CODE,'/kaggle/working/c42_legacy_v4_code')
Path('/kaggle/working/c42_legacy_compatibility.json').write_text(json.dumps(dict(v3=_c42_v3,v4=_c42_v4),indent=2))
'''
            c['source']=s.replace(needle,needle+'\n'+injected).splitlines(keepends=True);changed.append(i)
    assert changed==[27]
    assert all(frozen['cells'][i]['source']==prior['cells'][i]['source'] for i in range(len(prior['cells'])) if i not in changed)
    payload=json.dumps(frozen,ensure_ascii=False,indent=1).encode('utf8')
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(payload).decode())
    outer=replace_value(outer,'FROZEN_SHA256',hashlib.sha256(payload).hexdigest())
    for name,source in [('legacy_runtime_compat.py',helper),('legacy_formate_compat.py',compat)]:
        outer+='\nMODULES['+repr(name)+']='+repr(source)+'\nMODULE_SHA256['+repr(name)+']='+repr(hashlib.sha256(source.encode()).hexdigest())+'\n'
    prepared['cells'][12]['source']=guarded_source(outer,'c42_formate_only_parent_payload').splitlines(keepends=True)
    for trial,slot in [('cpu','p20-c42-cpu-alignment'),('gpu','p21-c42-gpu-alignment')]:
        folder=ROOT/'kaggle/forks'/slot;folder.mkdir(exist_ok=True)
        assert not (folder/'production-pin.json').exists()
        nb=json.loads(json.dumps(prepared))
        source='''# Every required candidate stage is covered by one cleanup guard.
import os,subprocess
import numpy as np
shutil.copy2(ROOT/'submission.csv',ROOT/'c42_parent.csv')
(ROOT/'submission.csv').unlink()
assert MERGE_REPORT.get('status')=='applied'
assert (ROOT/'p16_execution_receipt.json').is_file()
assert (ROOT/'c42_legacy_compatibility.json').is_file()
_c42_devices=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
assert len(_c42_devices)==2 and all('T4' in d['name'] for d in _c42_devices)
_hits=[p for p in INPUT.rglob('head-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest()==C42_HEAD_PIN]
assert len(_hits)==1
_c42_heads=_hits[0].parent;_c42_manifest=json.loads(_hits[0].read_text())
for _name,_pin in _c42_manifest['files'].items():assert hashlib.sha256((_c42_heads/_name).read_bytes()).hexdigest()==_pin,_name
_hits=[p for p in INPUT.rglob('frozen-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest()==_c42_manifest['encoder_bundle_manifest_sha256']]
assert len(_hits)==1;_c42_encoders=_hits[0].parent
_c42_parent=pd.read_csv(ROOT/'c42_parent.csv',dtype=str,keep_default_na=False)
_c42_queries=pd.read_parquet(ROOT/'effective_query_spectra.parquet')
_c42_active=set(_c42_queries.molecule_id.astype(str));assert _c42_active and _c42_active<=set(_c42_parent.molecule_id)
_c42_parent[_c42_parent.molecule_id.isin(_c42_active)].to_csv(ROOT/'c42_active_parent.csv',index=False)
_c42_output=ROOT/'c42_scores'
_env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (ROOT/'c42_inference.log').open('w') as _log:
    _run=subprocess.run([sys.executable,str(_c42_heads/'score_c42_crossmodal.py'),str(_c42_heads),str(_c42_encoders),
        str(ROOT/'c42_active_parent.csv'),str(ROOT/'effective_query_spectra.parquet'),str(_c42_output),'--only',C42_TRIAL],
        stdout=_log,stderr=subprocess.STDOUT,env=_env)
assert _run.returncode==0,'Required C42 inference failed; see c42_inference.log'
_r=json.loads((_c42_output/'score-receipt.json').read_text())
assert _r['head_manifest_sha256']==C42_HEAD_PIN and _r['source_sha256']==_c42_manifest['files']['score_c42_crossmodal.py']
assert _r['queries']==len(_c42_active) and not _r['quality_labels_read']
_recipe=_r['trials'][C42_TRIAL]
assert _recipe['rank1_changes']==_recipe['candidate_membership_changes']==_recipe['formula_position_changes']==0
_scored=pd.read_csv(_c42_output/(C42_TRIAL+'.csv'),dtype=str,keep_default_na=False).set_index('molecule_id')
assert set(_scored.index)==_c42_active and _scored.index.is_unique
_answer=_c42_parent.copy();_mask=_answer.molecule_id.isin(_c42_active)
_answer.loc[_mask,'smiles']=_answer.loc[_mask,'molecule_id'].map(_scored.smiles)
assert _answer.molecule_id.tolist()==SAMPLE_IDS and _answer.molecule_id.is_unique and _answer.smiles.notna().all()
for _old,_new in zip(_c42_parent.smiles,_answer.smiles):
    _a,_b=_old.split(';'),_new.split(';');assert _a[0]==_b[0] and len(_a)==len(_b) and set(_a)==set(_b)
assert _answer.loc[~_mask].equals(_c42_parent.loc[~_mask])
if not IS_RERUN:
    assert len(_c42_active)==12 and _r['supported_queries']>0
    assert _recipe['changed_lists']>0,'Real preview must exercise a changed learned ranking'
_answer.to_csv(ROOT/'c42_staged.csv',index=False)
_receipt=dict(trial=C42_TRIAL,head_manifest_sha256=C42_HEAD_PIN,parent_notebook_sha256=C42_PARENT_PIN,
    parent_csv_sha256=hashlib.sha256((ROOT/'c42_parent.csv').read_bytes()).hexdigest(),
    csv_sha256=hashlib.sha256((ROOT/'c42_staged.csv').read_bytes()).hexdigest(),valid_rows=len(_answer),
    actual_query_count=len(_c42_active),exact_unprocessed_fallback_rows=int((~_mask).sum()),
    devices=_c42_devices,is_hidden_rerun=IS_RERUN,pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),
    score_receipt=_r,parent_only_change='Tested legacy scalar formate alias source adapter',
    validation_reference_masks_mounted=False,query_labels_mounted=False,training_performed=False)
(ROOT/'c42_execution.json').write_text(json.dumps(_receipt,indent=2))
shutil.copy2(ROOT/'c42_staged.csv',ROOT/'submission.csv')
assert hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()==_receipt['csv_sha256']
print('DONE C42 REQUIRED PRODUCTION',json.dumps(_receipt),flush=True)
'''
        source='C42_TRIAL='+repr(trial)+'\nC42_HEAD_PIN='+repr(sha(bundle/'head-manifest.json'))+'\nC42_PARENT_PIN='+repr(sha(parent))+'\n'+source
        nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(source,'c42_all_required_stages').splitlines(keepends=True)))
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),'c42_production_'+str(i),'exec');c['execution_count']=None;c['outputs']=[]
        assert all(nb['cells'][i]['source']==original['cells'][i]['source'] for i in range(len(original['cells'])) if i!=12)
        (folder/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
        meta=json.loads(parent.with_name('kernel-metadata.json').read_text())
        meta.update(id='clarkkitchen/casmi26-'+slot,title='casmi26 '+slot.replace('-',' '),is_private=True,enable_internet=False,docker_image_pinning_type='original')
        meta['dataset_sources']+=['clarkkitchen/casmi26-c42r1-crossmodal-heads-20261111','clarkkitchen/casmi26-frozen-p14-20261107']
        assert not any('c41' in s for s in meta['dataset_sources'])
        write(folder/'kernel-metadata.json',meta)
        write(folder/'production-pin.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),trial=trial,
            parent_notebook_sha256=sha(parent),notebook_sha256=sha(folder/'notebook.ipynb'),scientific_sha256=scientific_sha(nb),
            head_manifest_sha256=sha(bundle/'head-manifest.json'),builder_sha256=sha(__file__),
            parent_outer_changed_cells=[12],parent_frozen_changed_cells=[27],
            parent_only_change='Legacy scalar formate alias source adapter',new_candidate_stage_cell=len(nb['cells'])-1,
            legacy_helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),compat_source_sha256=sha(ROOT/'scripts/legacy_formate_compat.py'),
            scoring_submission=False,quality_metrics_read=False,C13_read=False,mandatory_real_smoke=True,qualified_for_submission=False))
        print('BUILT C42 PRODUCTION',trial,sha(folder/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
