"""Qualify only the remaining C18 entry after a cleanup-only source amendment."""
import json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write

OUT=ROOT/'outputs/local/two-past-trials-20261110'
FOLDER=ROOT/'outputs/forks/p19-c18-guarded-export-v1'

def main():
    target=OUT/'submission-qualification-guarded.json';assert not target.exists()
    auth=ROOT/'.local/bounded-two-submissions-20261110.json';a=json.loads(auth.read_text())
    assert a['maximum_new_submissions']==2 and a['maximum_pending']==1
    ledger=json.loads((ROOT/'.local/two-past-trials-ledger-20261110.json').read_text())
    assert len(ledger['submissions'])==1 and str(ledger['submissions'][0]['reference'])=='58212322'
    e=json.loads((OUT/'guarded-export-freeze.json').read_text());audit_path=FOLDER/'independent_guarded_export_audit.json';audit=json.loads(audit_path.read_text())
    assert e['builder_sha256']==sha(ROOT/'scripts/build_guarded_past_export.py')
    assert e['raw_joint_prefix_exact'] and e['parent_and_export_cells_exact'] and not e['recipe_changed'] and not e['model_scoring_code_changed']
    assert audit['execution_qualified'] and audit['export_slot']=='P19' and audit['raw_joint_prefix_exact']
    assert audit['whole_stage_cleanup_guard'] and audit['real_Kaggle_cleanup_failure_probe_passed'] and audit['normal_output_exact_previous_export']
    assert audit['notebook_sha256']==e['notebook_sha256']==sha(ROOT/e['notebook_path'])
    assert audit['remote_scientific_sha256']==e['scientific_sha256']
    assert audit['independent_audit_source_sha256']==sha(ROOT/'scripts/audit_guarded_past_export.py')
    assert audit['prefix_audit_source_sha256']==sha(ROOT/'scripts/audit_two_past_trials.py')
    r=audit['recipes']['P19'];assert r['output']=='submission.csv' and r['csv_sha256']==sha(FOLDER/'submission.csv')
    assert r['real_queries']==12 and r['valid_rows']==400 and r['changed_rows']>0 and r['exact_fallback_rows']==388
    assert r['official_identity_duplicates']==r['rank_one_changes']==r['candidate_set_changes']==0
    assert r['local_cloud_final_lists_exact'] and r['independent_tail_arithmetic_exact']
    assert max(r['local_cloud_forward_max_abs_difference'].values())<1e-3
    push=json.loads((OUT/'guarded-export-push-receipt.json').read_text());assert push['scientific_sha256']==e['scientific_sha256'] and push['private'] and not push['internet']
    paths=[OUT/n for n in ['guarded-export-freeze.json','guarded-export-push-receipt.json','production-freeze.json','selection.json',
        'selection-protocol.json','prediction-freeze.json','dataset-privacy.json','submission-qualification-exports.json','preflight-cleanup-probe.json']]
    paths += [ROOT/'scripts'/n for n in ['build_guarded_past_export.py','audit_guarded_past_export.py','audit_two_past_trials.py',
        'qualify_guarded_past_export.py','submit_guarded_past_trial.py','monitor_reviewed_past_trials.py','c09_submission_guard.py','compatible_tail.py',
        'build_general_anchor.py','submit_c09_external_once.py']]
    paths += [ROOT/e['notebook_path'],(ROOT/e['notebook_path']).with_name('kernel-metadata.json'),audit_path]
    paths += [FOLDER/n for n in ['past_trial_cleanup_probe.json','past_trial_export_receipt.json','independent_past_trials_audit.json',
        'past_trials_execution.json','past_trials_parent.csv','effective_query_spectra.parquet','submission.csv','p19_c18_submission.csv','P18_scores.parquet','P19_scores.parquet']]
    production=json.loads((OUT/'production-freeze.json').read_text())
    for m in production['models'].values():
        bundle=ROOT/m['local_bundle'];assert sha(bundle/'c08-manifest.json')==m['manifest_sha256']
        for n,pin in m['files_sha256'].items():assert sha(bundle/n)==pin,n
        paths += [bundle/'c08-manifest.json']+[bundle/n for n in m['files_sha256']]
    export=dict(e,audit_path=str(audit_path.relative_to(ROOT)))
    report=dict(qualified_utc=datetime.now(timezone.utc).isoformat(),decision='execution_qualified_explicit_two_trials',order=['P18','P19'],
        maximum_new_submissions=2,maximum_pending=1,champion_retained=.480,authorization_sha256=sha(auth),prior_accepted_reference=58212322,
        exports={'P19':export},artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},guard_only_amendment=True,
        statistical_promotion=False,reserved_labels_read=False,no_new_training=True,no_hourly_campaign=True,no_third_trial=True,
        rationale='One approved entry remains. The C18 mathematical scoring body, checkpoints, parent and export are exact; only whole-stage failure cleanup and a synthetic failure probe were added. Real GPU normal output remains byte-identical; real Kaggle forced-preflight probe confirms baseline cleanup. First P18 source/submission remains immutable. Explicit scoring trial; prior statistical gate still failed.')
    write(target,report);print('QUALIFIED REMAINING GUARDED C18 ENTRY',sha(target),flush=True)

if __name__=='__main__':main()
