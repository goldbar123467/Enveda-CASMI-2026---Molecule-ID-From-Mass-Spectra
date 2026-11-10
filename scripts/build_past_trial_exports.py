"""Competition requires submission.csv; append only an exact-file export alias."""
import hashlib,json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from submit_c09_external_once import scientific_sha

OUT=ROOT/'outputs/local/two-past-trials-20261110'
JOINT=ROOT/'kaggle/forks/p18-p19-past-trained-trials'

def main():
    assert not (OUT/'export-freeze.json').exists()
    original=json.loads((JOINT/'notebook.ipynb').read_text(encoding='utf-8'))
    old=json.loads((OUT/'production-freeze.json').read_text())
    assert sha(JOINT/'notebook.ipynb')==old['notebook_sha256'] and scientific_sha(original)==old['scientific_sha256']
    exports={}
    for slot,slug in [('P18','p18-c22-past-trained-trial'),('P19','p19-c18-past-trained-trial')]:
        nb=json.loads(json.dumps(original));file=old['models'][slot]['output']
        ending="""# Competition-mandated output filename; no scoring or recipe change.
_export_slot=EXPORT_SLOT
_export_source=EXPORT_SOURCE
_export_hash=_past_receipts[_export_slot]['csv_sha256']
assert hashlib.sha256((ROOT/_export_source).read_bytes()).hexdigest()==_export_hash
shutil.copy2(ROOT/_export_source,ROOT/'submission.csv')
assert hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()==_export_hash
(ROOT/'past_trial_export_receipt.json').write_text(json.dumps(dict(slot=_export_slot,source_file=_export_source,
    output='submission.csv',csv_sha256=_export_hash,scientific_prefix_sha256=PREFIX_SHA,
    parent_stage_and_both_retained_model_stages_required=True,training_performed=False),indent=2))
"""
        ending='EXPORT_SLOT='+repr(slot)+'\nEXPORT_SOURCE='+repr(file)+'\nPREFIX_SHA='+repr(old['scientific_sha256'])+'\n'+ending
        nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(ending,'past_trial_required_export').splitlines(keepends=True)))
        assert nb['cells'][:-1]==original['cells']
        folder=ROOT/'kaggle/forks'/slug;folder.mkdir(exist_ok=True)
        (folder/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
        meta=json.loads((JOINT/'kernel-metadata.json').read_text());meta.update(id='clarkkitchen/casmi26-'+slug,title='casmi26 '+slug.replace('-',' '))
        write(folder/'kernel-metadata.json',meta)
        exports[slot]=dict(kernel=meta['id'],notebook_path=str((folder/'notebook.ipynb').relative_to(ROOT)),
            notebook_sha256=sha(folder/'notebook.ipynb'),scientific_sha256=scientific_sha(nb),version=1,output='submission.csv',
            source_file=file,joint_prefix_exact=True,scientific_prefix_sha256=old['scientific_sha256'])
    write(OUT/'export-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),exports=exports,
        joint_notebook_sha256=old['notebook_sha256'],joint_scientific_sha256=old['scientific_sha256'],builder_sha256=sha(__file__),
        reason='Kaggle HTTP400 explicitly requires filename submission.csv; neither rejected request consumed an accepted scoring trial.',
        original_qualification_preserved=True,new_training=False,new_recipe=False,maximum_accepted_trials=2))
    print('BUILT TWO EXACT EXPORT ALIASES',json.dumps(exports),flush=True)

if __name__=='__main__':main()
