"""Audit a mandatory filename alias after an exact joint-prefix GPU rerun."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
from submit_c09_external_once import scientific_sha
import audit_two_past_trials as prefix_audit

OUT=ROOT/'outputs/local/two-past-trials-20261110'

def main():
    p=argparse.ArgumentParser();p.add_argument('--slot',choices=['P18','P19'],required=True);p.add_argument('folder',type=Path);a=p.parse_args();a.folder=a.folder.resolve()
    target=a.folder/'independent_export_audit.json';assert not target.exists()
    freeze=json.loads((OUT/'export-freeze.json').read_text());entry=freeze['exports'][a.slot]
    notebook=json.loads((ROOT/entry['notebook_path']).read_text(encoding='utf-8'))
    joint=json.loads((ROOT/'kaggle/forks/p18-p19-past-trained-trials/notebook.ipynb').read_text(encoding='utf-8'))
    assert notebook['cells'][:-1]==joint['cells'] and scientific_sha(joint)==entry['scientific_prefix_sha256']
    assert scientific_sha(notebook)==entry['scientific_sha256'] and sha(ROOT/entry['notebook_path'])==entry['notebook_sha256']
    receipt=json.loads((a.folder/'past_trial_export_receipt.json').read_text())
    assert receipt['slot']==a.slot and receipt['source_file']==entry['source_file'] and receipt['output']=='submission.csv'
    assert receipt['scientific_prefix_sha256']==entry['scientific_prefix_sha256']
    assert receipt['parent_stage_and_both_retained_model_stages_required'] and not receipt['training_performed']
    assert sha(a.folder/'submission.csv')==sha(a.folder/entry['source_file'])==receipt['csv_sha256']
    # This frozen auditor independently validates both newly executed model
    # paths, parent, identities, fallback, arithmetic and local/cloud forward
    # parity. Its source claim is scoped to the unchanged joint prefix.
    prefix_path=a.folder/'independent_past_trials_audit.json'
    if not prefix_path.exists():
        previous=sys.argv;sys.argv=[prefix_audit.__file__,str(a.folder)]
        try:prefix_audit.main()
        finally:sys.argv=previous
    report=json.loads(prefix_path.read_text())
    assert report['execution_qualified'] and report['independent_audit_source_sha256']==sha(prefix_audit.__file__)
    assert report['remote_scientific_sha256']==entry['scientific_prefix_sha256']
    assert report['recipes'][a.slot]['csv_sha256']==receipt['csv_sha256']
    result=json.loads(json.dumps(report));result['recipes'][a.slot].update(source_output=entry['source_file'],output='submission.csv')
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),export_slot=a.slot,joint_prefix_exact=True,
        scientific_prefix_sha256=entry['scientific_prefix_sha256'],prefix_audit_path=str(prefix_path.relative_to(ROOT)),
        prefix_audit_sha256=sha(prefix_path),prefix_audit_source_sha256=sha(prefix_audit.__file__),
        independent_audit_source_sha256=sha(__file__),notebook_sha256=entry['notebook_sha256'],
        remote_scientific_sha256=entry['scientific_sha256'],competition_mandated_filename='submission.csv',
        export_receipt_sha256=sha(a.folder/'past_trial_export_receipt.json'),export_alias_bytes_exact=True)
    write(target,result);print('DONE EXACT EXPORT AUDIT',a.slot,json.dumps(result['recipes'][a.slot]),flush=True)

if __name__=='__main__':main()
