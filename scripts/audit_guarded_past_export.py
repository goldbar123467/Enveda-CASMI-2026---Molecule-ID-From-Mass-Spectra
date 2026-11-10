"""Audit whole-stage cleanup and the byte-exact retained scoring body."""
import argparse,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from build_c09_controls import unwrap
from submit_c09_external_once import scientific_sha
import audit_two_past_trials as prefix_audit

OUT=ROOT/'outputs/local/two-past-trials-20261110'

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder.resolve()
    target=folder/'independent_guarded_export_audit.json';assert not target.exists()
    e=json.loads((OUT/'guarded-export-freeze.json').read_text());source=ROOT/e['notebook_path']
    nb=json.loads(source.read_text(encoding='utf-8'));old=json.loads((ROOT/'kaggle/forks/p19-c18-past-trained-trial/notebook.ipynb').read_text(encoding='utf-8'))
    index=e['joint_cell_index'];raw=''.join(old['cells'][index]['source']);actual=''.join(nb['cells'][index]['source'])
    assert len(nb['cells'])==len(old['cells'])+1
    assert actual==guarded_source(raw,'all_required_retained_model_stages') and unwrap(actual)==raw
    assert hashlib.sha256(raw.encode()).hexdigest()==e['raw_joint_body_sha256']
    for i in range(len(old['cells'])):
        if i!=index:assert nb['cells'][i]==old['cells'][i]
    assert sha(source)==e['notebook_sha256'] and scientific_sha(nb)==e['scientific_sha256']
    probe_path=folder/'past_trial_cleanup_probe.json';probe=json.loads(probe_path.read_text())
    assert probe['actual_scoring_file_unchanged'] and probe['synthetic_only'] and not probe['quality_labels_used']
    assert probe['results']['original']['default_file_left'] and not probe['results']['guarded']['default_file_left']
    assert all(v['protected_parent_retained'] for v in probe['results'].values())
    receipt=json.loads((folder/'past_trial_export_receipt.json').read_text())
    assert receipt['slot']=='P19' and receipt['source_file']=='p19_c18_submission.csv' and receipt['output']=='submission.csv'
    assert sha(folder/'submission.csv')==sha(folder/receipt['source_file'])==receipt['csv_sha256']
    prefix_path=folder/'independent_past_trials_audit.json'
    if not prefix_path.exists():
        previous=sys.argv;sys.argv=[prefix_audit.__file__,str(folder)]
        try:prefix_audit.main()
        finally:sys.argv=previous
    report=json.loads(prefix_path.read_text());assert report['execution_qualified']
    assert report['independent_audit_source_sha256']==sha(prefix_audit.__file__)
    assert report['recipes']['P19']['csv_sha256']==receipt['csv_sha256']
    previous=json.loads((ROOT/'outputs/forks/p19-c18-past-trained-trial-v1/independent_export_audit.json').read_text())
    assert previous['recipes']['P19']['csv_sha256']==receipt['csv_sha256'],'Normal output changed after cleanup-only guard'
    result=json.loads(json.dumps(report));result['recipes']['P19'].update(source_output='p19_c18_submission.csv',output='submission.csv')
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(),export_slot='P19',joint_prefix_exact=False,
        raw_joint_prefix_exact=True,parent_and_export_cells_exact=True,whole_stage_cleanup_guard=True,
        cleanup_probe_sha256=sha(probe_path),real_Kaggle_cleanup_failure_probe_passed=True,normal_output_exact_previous_export=True,
        prefix_audit_sha256=sha(prefix_path),prefix_audit_source_sha256=sha(prefix_audit.__file__),
        independent_audit_source_sha256=sha(__file__),notebook_sha256=e['notebook_sha256'],
        remote_scientific_sha256=e['scientific_sha256'],competition_mandated_filename='submission.csv',export_alias_bytes_exact=True)
    write(target,result);print('DONE GUARDED SAME-RECIPE EXPORT AUDIT',json.dumps(result['recipes']['P19']),flush=True)

if __name__=='__main__':main()
