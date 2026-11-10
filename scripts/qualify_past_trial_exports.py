"""Bind each separately smoked submission.csv alias to the same two recipes."""
import json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write

OUT=ROOT/'outputs/local/two-past-trials-20261110'

def main():
    target=OUT/'submission-qualification-exports.json';assert not target.exists()
    auth=ROOT/'.local/bounded-two-submissions-20261110.json';authorization=json.loads(auth.read_text())
    assert authorization['maximum_new_submissions']==2 and authorization['maximum_pending']==1
    freeze=json.loads((OUT/'export-freeze.json').read_text());production=json.loads((OUT/'production-freeze.json').read_text())
    assert freeze['builder_sha256']==sha(ROOT/'scripts/build_past_trial_exports.py')
    privacy=json.loads((OUT/'dataset-privacy.json').read_text())
    assert {d['ref'] for d in privacy['datasets']}=={e['dataset'] for e in production['models'].values()}
    assert all(d['isPrivate'] and d['currentVersionNumber']==1 and d['status']=='ready' for d in privacy['datasets'])
    exports={};paths=[OUT/n for n in ['export-freeze.json','production-freeze.json','selection.json','selection-protocol.json',
        'prediction-freeze.json','dataset-privacy.json','rejected-export-reconciliation.json','submission-http-error-detail.json']]
    paths += [ROOT/'scripts'/n for n in ['build_past_trial_exports.py','audit_past_trial_export.py','audit_two_past_trials.py',
        'submit_past_trial_exports.py','monitor_past_trial_exports.py','qualify_past_trial_exports.py','c09_submission_guard.py','compatible_tail.py']]
    for slot,entry in freeze['exports'].items():
        source=ROOT/entry['notebook_path'];folder=ROOT/'outputs/forks'/source.parent.name
        folder=folder.with_name(folder.name+'-v1');audit_path=folder/'independent_export_audit.json'
        audit=json.loads(audit_path.read_text());push=OUT/('export-push-receipt-'+slot+'.json')
        assert audit['execution_qualified'] and audit['export_slot']==slot and audit['joint_prefix_exact'] and audit['export_alias_bytes_exact']
        assert not audit['training_performed'] and not audit['reserved_labels_read'] and not audit['statistical_promotion']
        assert audit['notebook_sha256']==entry['notebook_sha256']==sha(source)
        assert audit['remote_scientific_sha256']==entry['scientific_sha256']==json.loads(push.read_text())['scientific_sha256']
        assert audit['independent_audit_source_sha256']==sha(ROOT/'scripts/audit_past_trial_export.py')
        assert audit['prefix_audit_source_sha256']==sha(ROOT/'scripts/audit_two_past_trials.py')
        assert audit['selection_sha256']==sha(OUT/'selection.json')
        r=audit['recipes'][slot];assert r['output']=='submission.csv' and r['csv_sha256']==sha(folder/'submission.csv')
        assert r['real_queries']==12 and r['valid_rows']==400 and r['changed_rows']>0 and r['exact_fallback_rows']==388
        assert r['official_identity_duplicates']==r['rank_one_changes']==r['candidate_set_changes']==0
        assert r['local_cloud_final_lists_exact'] and r['independent_tail_arithmetic_exact']
        assert max(r['local_cloud_forward_max_abs_difference'].values())<1e-3 and audit['unchanged_P01_primary_error']<1e-6
        exports[slot]=dict(entry,audit_path=str(audit_path.relative_to(ROOT)))
        paths += [source,source.with_name('kernel-metadata.json'),push,audit_path,folder/'past_trial_export_receipt.json',
            folder/'independent_past_trials_audit.json',folder/'past_trials_execution.json',folder/'past_trials_parent.csv',
            folder/'effective_query_spectra.parquet',folder/'submission.csv',folder/entry['source_file']]
        paths += [folder/(name+'_scores.parquet') for name in ['P18','P19']]
    for entry in production['models'].values():
        bundle=ROOT/entry['local_bundle'];assert sha(bundle/'c08-manifest.json')==entry['manifest_sha256']
        for name,pin in entry['files_sha256'].items():assert sha(bundle/name)==pin,name
        paths += [bundle/'c08-manifest.json']+[bundle/name for name in entry['files_sha256']]
    report=dict(qualified_utc=datetime.now(timezone.utc).isoformat(),decision='execution_qualified_explicit_two_trials',
        order=['P18','P19'],maximum_new_submissions=2,maximum_pending=1,champion_retained=.480,
        authorization_sha256=sha(auth),exports=exports,artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        statistical_promotion=False,reserved_labels_read=False,no_new_training=True,no_hourly_campaign=True,no_third_trial=True,
        rationale='Same frozen two retained fits and joint scientific prefix; only mandatory submission.csv export aliases added. Both exact export notebooks completed real Kaggle GPU execution and independent source/model/arithmetic/identity/fallback/local-cloud audits. Prior HTTP400 requests were rejected, reconciled and consumed no accepted trial. Explicit user trials; development confidence intervals still cross zero.')
    write(target,report);print('QUALIFIED TWO MANDATORY-FILENAME EXPORTS',sha(target),flush=True)

if __name__=='__main__':main()
