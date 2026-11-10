"""Package unchanged P16 plus the two retained fitted tail recipes; no training."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from submit_c09_external_once import scientific_sha

OUT=ROOT/'outputs/local/two-past-trials-20261110'
FORK=ROOT/'kaggle/forks/p18-p19-past-trained-trials'

def main():
    assert not (OUT/'push-intent.json').exists(),'Do not alter a deployed build'
    selected=json.loads((OUT/'selection.json').read_text())
    assert selected['selected']==['C22-peak-dose_allmode','C18-peak-drop_allmode']
    parent=ROOT/'kaggle/forks/p16-external-reference-c09/notebook.ipynb'
    assert sha(parent)=='d6b9c9625ad8fb7d503341c6fd702bd909101d719ca55e400145fb96f2b2af6a'
    notebook=json.loads(parent.read_text(encoding='utf-8'));original=json.loads(parent.read_text(encoding='utf-8'))
    models={}
    for slot,run,arm,steps,file in [('P18','c22-peak-models-20261110','peak-drop-dose',55000,'p18_c22_submission.csv'),('P19','c18-peak-models-20261110','peak-drop',45000,'p19_c18_submission.csv')]:
        bundle=ROOT/'outputs/publish'/run;manifest=json.loads((bundle/'c08-manifest.json').read_text())
        assert manifest['steps']==steps and len(manifest['models'])==2 and not manifest['smoke']
        for name,pin in manifest['files'].items():assert sha(bundle/name)==pin,name
        assert sha(bundle/'compatible_tail.py')==sha(ROOT/'scripts/compatible_tail.py')
        models[slot]=dict(dataset='clarkkitchen/casmi26-'+run,manifest_sha256=sha(bundle/'c08-manifest.json'),arm=arm,steps=steps,output=file,
            files_sha256=manifest['files'],local_bundle=str(bundle.relative_to(ROOT)))
    source=r'''import os,subprocess,concurrent.futures,hashlib
import numpy as np
_past_parent=ROOT/'past_trials_parent.csv'
shutil.copy2(ROOT/'submission.csv',_past_parent)
_past_before=pd.read_csv(_past_parent,dtype=str,keep_default_na=False)
assert MERGE_REPORT['status']=='applied'
assert hashlib.sha256((ROOT/'p16_execution_receipt.json').read_bytes()).hexdigest()
_past_devices=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
assert len(_past_devices)==2 and all(d['name']=='Tesla T4' for d in _past_devices)
_past_entries={}
for _slot,_entry in PAST_MODELS.items():
    _hits=[p for p in INPUT.rglob('c08-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest()==_entry['manifest_sha256']]
    assert len(_hits)==1,(_slot,len(_hits))
    _bundle=_hits[0].parent;_manifest=json.loads(_hits[0].read_text())
    assert _manifest['steps']==_entry['steps'] and not _manifest['smoke']
    assert len(_manifest['models'])==2 and [m['fold'] for m in _manifest['models']]==[0,1]
    for _name,_pin in _entry['files_sha256'].items():assert hashlib.sha256((_bundle/_name).read_bytes()).hexdigest()==_pin
    _past_entries[_slot]=(_bundle,_entry)
def _past_score(_slot,_gpu):
    _bundle,_entry=_past_entries[_slot]
    _env=os.environ.copy();_env['CUDA_VISIBLE_DEVICES']=str(_gpu);_env['OMP_NUM_THREADS']='2';_env['MKL_NUM_THREADS']='2'
    _target=ROOT/(_slot+'_staged.csv')
    _cmd=[sys.executable,str(_bundle/'inference.py'),'--bundle',str(_bundle),'--arm',_entry['arm'],
          '--baseline',str(_past_parent),'--queries',str(ROOT/'effective_query_spectra.parquet'),
          '--scores',str(ROOT/(_slot+'_scores.parquet')),'--output',str(_target),'--weight','0.2']
    with (ROOT/(_slot+'_inference.log')).open('w') as _log:
        _run=subprocess.run(_cmd,stdout=_log,stderr=subprocess.STDOUT,env=_env)
    assert _run.returncode==0,(_slot,'Required inference failed')
    return _slot,_target
_past_ready={}
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
        for _slot,_target in _pool.map(lambda args:_past_score(*args),[('P18',0),('P19',1)]):_past_ready[_slot]=_target
    _past_receipts={}
    for _slot,_target in _past_ready.items():
        _entry=PAST_MODELS[_slot]
        _after=pd.read_csv(_target,dtype=str,keep_default_na=False)
        _r=json.loads(Path(str(_target)+'.c08.json').read_text())
        assert _r['manifest_sha256']==_entry['manifest_sha256'] and _r['arm']==_entry['arm'] and _r['steps']==_entry['steps']
        assert _r['models']==2 and _r['folds']==[0,1] and _r['input_errors']==0
        assert _r['weight']==.2 and _r['offset']==3. and _r['rank_one_changes']==_r['candidate_set_changes']==0
        assert _after.molecule_id.tolist()==_past_before.molecule_id.tolist()==SAMPLE_IDS and _after.molecule_id.is_unique
        for _old,_new in zip(_past_before.smiles,_after.smiles):
            _a,_b=_old.split(';'),_new.split(';');assert _a[0]==_b[0] and len(_a)==len(_b) and set(_a)==set(_b)
        if not IS_RERUN:assert _r['scored_molecules']==12 and _r['changed_rows']>0
        _r.update(allocated_gpu=0 if _slot=='P18' else 1,output=_entry['output'],csv_sha256=hashlib.sha256(_target.read_bytes()).hexdigest())
        _past_receipts[_slot]=_r
    _past_report=dict(kind='Two_explicit_past_fitted_trials',models=PAST_MODELS,recipes=_past_receipts,devices=_past_devices,
        parent_csv_sha256=hashlib.sha256(_past_parent.read_bytes()).hexdigest(),parent_notebook_sha256=PAST_PARENT_SHA,
        selection_sha256=PAST_SELECTION_SHA,valid_rows=len(_past_before),pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),
        is_hidden_rerun=IS_RERUN,weight=.2,offset=3.,reserved_labels_mounted=False,training_performed=False,statistical_promotion=False)
    (ROOT/'past_trials_execution.json').write_text(json.dumps(_past_report,indent=2))
    for _slot,_target in _past_ready.items():shutil.copy2(_target,ROOT/PAST_MODELS[_slot]['output'])
    (ROOT/'submission.csv').unlink()
except BaseException:
    for _file in ['submission.csv','p18_c22_submission.csv','p19_c18_submission.csv']:(ROOT/_file).unlink(missing_ok=True)
    raise
print('DONE TWO PAST FITTED TRIALS; BOTH REQUIRED STAGES SUCCEEDED',flush=True)
'''
    constants='PAST_MODELS='+repr(models)+'\nPAST_PARENT_SHA='+repr(sha(parent))+'\nPAST_SELECTION_SHA='+repr(sha(OUT/'selection.json'))+'\n'
    source=constants+source;compile(source,'two_past_trials','exec')
    notebook['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(keepends=True)))
    assert notebook['cells'][:-1]==original['cells'],'The complete P16 source must remain exact'
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'past_trials_build','exec');cell['execution_count']=None;cell['outputs']=[]
    FORK.mkdir(exist_ok=True)
    (FORK/'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
    meta=json.loads(parent.with_name('kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-p18-p19-past-trained-trials',title='casmi26 p18 p19 past trained trials',is_private=True,enable_internet=False,docker_image_pinning_type='original')
    meta['dataset_sources'] += [v['dataset'] for v in models.values()]
    write(FORK/'kernel-metadata.json',meta)
    write(OUT/'production-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),notebook_sha256=sha(FORK/'notebook.ipynb'),
        scientific_sha256=scientific_sha(notebook),parent_notebook_sha256=sha(parent),parent_code_exact=True,models=models,
        selection_sha256=sha(OUT/'selection.json'),builder_sha256=sha(__file__),weight=.2,offset=3.,mandatory_real_smoke=True,
        statistical_promotion=False,explicit_user_trials=2,maximum_pending=1,reserved_labels_read=False,training_performed=False))
    print('BUILT TWO PAST TRIALS',sha(FORK/'notebook.ipynb'),flush=True)

if __name__=='__main__':main()
