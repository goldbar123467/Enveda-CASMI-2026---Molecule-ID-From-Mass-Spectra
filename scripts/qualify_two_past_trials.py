"""Seal execution qualification for exactly two explicit past-work trials."""
import json
from datetime import datetime,timezone
from pathlib import Path
from precision_cache import ROOT,sha,write

OUT=ROOT/'outputs/local/two-past-trials-20261110'
SMOKE=ROOT/'outputs/forks/p18-p19-past-trained-trials-v1'

def main():
    qpath=OUT/'submission-qualification.json';assert not qpath.exists(),'Qualification is immutable'
    authorization=ROOT/'.local/bounded-two-submissions-20261110.json'
    auth=json.loads(authorization.read_text());assert auth['maximum_new_submissions']==2 and auth['maximum_pending']==1
    audit_path=SMOKE/'independent_past_trials_audit.json';audit=json.loads(audit_path.read_text())
    pin=json.loads((OUT/'production-freeze.json').read_text());push=json.loads((OUT/'push-receipt.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    privacy=json.loads((OUT/'dataset-privacy.json').read_text())
    assert pin['builder_sha256']==sha(ROOT/'scripts/build_two_past_trials.py')
    assert pin['notebook_sha256']==sha(ROOT/'kaggle/forks/p18-p19-past-trained-trials/notebook.ipynb')
    assert {d['ref'] for d in privacy['datasets']}=={m['dataset'] for m in pin['models'].values()}
    assert all(d['isPrivate'] and d['currentVersionNumber']==1 and d['status']=='ready' for d in privacy['datasets'])
    assert selection['selected']==['C22-peak-dose_allmode','C18-peak-drop_allmode'] and not selection['reserved_labels_read']
    assert audit['execution_qualified'] and audit['source_parent_exact'] and not audit['statistical_promotion']
    assert not audit['training_performed'] and not audit['reserved_labels_read'] and audit['unchanged_P01_primary_error']<1e-6
    assert audit['notebook_sha256']==pin['notebook_sha256'] and audit['remote_scientific_sha256']==push['remote_scientific_sha256']
    assert audit['selection_sha256']==sha(OUT/'selection.json')
    for slot,r in audit['recipes'].items():
        assert r['changed_rows']>0 and r['real_queries']==12 and r['valid_rows']==400 and r['exact_fallback_rows']==388
        assert r['official_identity_duplicates']==r['rank_one_changes']==r['candidate_set_changes']==0
        assert r['independent_tail_arithmetic_exact'] and r['local_cloud_final_lists_exact']
        assert max(r['local_cloud_forward_max_abs_difference'].values())<1e-3 and r['independent_standardization_error']<1e-10
        assert r['manifest_sha256']==pin['models'][slot]['manifest_sha256']
        assert r['csv_sha256']==sha(SMOKE/r['output'])
    paths=[ROOT/'scripts'/name for name in ['select_two_past_trials.py','build_two_past_trials.py','audit_two_past_trials.py',
        'submit_two_past_trials.py','monitor_two_past_trials.py','qualify_two_past_trials.py','c09_submission_guard.py','compatible_tail.py']]
    paths += [ROOT/'kaggle/forks/p18-p19-past-trained-trials'/n for n in ['notebook.ipynb','kernel-metadata.json']]
    paths += [OUT/n for n in ['selection-protocol.json','selection.json','prediction-freeze.json','production-freeze.json','dataset-privacy.json','push-receipt.json']]
    paths += [audit_path,SMOKE/'past_trials_execution.json',SMOKE/'past_trials_parent.csv',SMOKE/'effective_query_spectra.parquet']
    for slot,entry in pin['models'].items():
        bundle=ROOT/entry['local_bundle'];paths+=[bundle/'c08-manifest.json']+[bundle/n for n in entry['files_sha256']]
        paths += [SMOKE/entry['output'],SMOKE/(slot+'_scores.parquet')]
    report=dict(qualified_utc=datetime.now(timezone.utc).isoformat(),decision='execution_qualified_explicit_two_trials',
        order=['P18','P19'],kernel=push['kernel'],kernel_version=push['version'],maximum_new_submissions=2,maximum_pending=1,
        champion_retained=.480,authorization_sha256=sha(authorization),audit_path=str(audit_path.relative_to(ROOT)),
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},statistical_promotion=False,reserved_labels_read=False,
        rationale='User explicitly requested two best past-work scoring trials. Select retained complete fitted recipes on common already-used240; both95%CIs cross zero, previous promotion gates remain failed. Real COMPLETE Kaggle execution, independent source/model/identity/fusion/fallback and local/cloud audits passed. No claim of hidden gain.',
        no_new_training=True,no_hourly_campaign=True,no_third_trial=True)
    write(qpath,report);print('QUALIFIED TWO EXPLICIT EXECUTION TRIALS',sha(qpath),flush=True)

if __name__=='__main__':main()
