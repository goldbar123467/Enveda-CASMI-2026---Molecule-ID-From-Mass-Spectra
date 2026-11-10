"""Close preflight cleanup gaps without changing either retained scoring body."""
import hashlib,inspect,json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from build_c09_controls import unwrap
from submit_c09_external_once import scientific_sha

OUT=ROOT/'outputs/local/two-past-trials-20261110'
OLD=ROOT/'kaggle/forks/p19-c18-past-trained-trial'
FORK=ROOT/'kaggle/forks/p19-c18-guarded-export'

def main():
    assert not (OUT/'guarded-export-freeze.json').exists()
    nb=json.loads((OLD/'notebook.ipynb').read_text(encoding='utf-8'));original=json.loads(json.dumps(nb));index=len(nb['cells'])-2
    raw=''.join(nb['cells'][index]['source']);assert 'PAST_MODELS=' in raw and '_past_parent' in raw
    nb['cells'][index]['source']=guarded_source(raw,'all_required_retained_model_stages').splitlines(keepends=True)
    assert unwrap(''.join(nb['cells'][index]['source']))==raw
    for i in range(len(original['cells'])):
        if i!=index:assert nb['cells'][i]==original['cells'][i]
    probe=r'''# Exercise the changed failure branch with synthetic files only.
if not IS_RERUN:
    from types import SimpleNamespace
    _guard_factory={};exec(GUARD_FACTORY_SOURCE,_guard_factory)
    _before_guard_probe=hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()
    _guard_results={}
    for _label,_guarded in [('original',False),('guarded',True)]:
        _dir=ROOT/('guard_probe_'+_label);_dir.mkdir(exist_ok=True)
        pd.DataFrame([dict(molecule_id='synthetic_probe',smiles='C')]).to_csv(_dir/'submission.csv',index=False)
        (_dir/'p16_execution_receipt.json').write_text('{}')
        _scope=dict(ROOT=_dir,INPUT=_dir/'empty-input',pd=pd,shutil=shutil,
            torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:0)),MERGE_REPORT={'status':'applied'})
        _test_source=_guard_factory['guarded_source'](RAW_JOINT_BODY,'forced_preflight',str(_dir/'submission.csv')) if _guarded else RAW_JOINT_BODY
        try:exec(compile(_test_source,'controlled_preflight_failure','exec'),_scope)
        except AssertionError:pass
        else:raise AssertionError('Expected a forced preflight failure')
        _guard_results[_label]=dict(default_file_left=(_dir/'submission.csv').exists(),protected_parent_retained=(_dir/'past_trials_parent.csv').exists())
    assert _guard_results['original']['default_file_left'] and not _guard_results['guarded']['default_file_left']
    assert all(v['protected_parent_retained'] for v in _guard_results.values())
    assert hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()==_before_guard_probe
    (ROOT/'past_trial_cleanup_probe.json').write_text(json.dumps(dict(results=_guard_results,
        actual_scoring_file_unchanged=True,synthetic_only=True,quality_labels_used=False,training_performed=False),indent=2))
'''
    probe='GUARD_FACTORY_SOURCE='+repr(inspect.getsource(guarded_source))+'\nRAW_JOINT_BODY='+repr(raw)+'\n'+probe
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(probe,'required_preflight_cleanup_probe').splitlines(keepends=True)))
    FORK.mkdir(exist_ok=True)
    (FORK/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
    meta=json.loads((OLD/'kernel-metadata.json').read_text());meta.update(id='clarkkitchen/casmi26-p19-c18-guarded-export',title='casmi26 p19 c18 guarded export');write(FORK/'kernel-metadata.json',meta)
    write(OUT/'guarded-export-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),slot='P19',kernel=meta['id'],version=1,
        notebook_path=str((FORK/'notebook.ipynb').relative_to(ROOT)),notebook_sha256=sha(FORK/'notebook.ipynb'),scientific_sha256=scientific_sha(nb),
        prior_export_notebook_sha256=sha(OLD/'notebook.ipynb'),joint_cell_index=index,raw_joint_body_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        raw_joint_prefix_exact=True,parent_and_export_cells_exact=True,model_scoring_code_changed=False,recipe_changed=False,
        full_stage_cleanup_guard=True,synthetic_real_Kaggle_failure_probe_required=True,builder_sha256=sha(__file__),
        output='submission.csv',source_file='p19_c18_submission.csv',prior_P18_submission_immutable=58212322))
    print('BUILT GUARDED SAME-RECIPE C18 EXPORT',sha(FORK/'notebook.ipynb'),flush=True)

if __name__=='__main__':main()
