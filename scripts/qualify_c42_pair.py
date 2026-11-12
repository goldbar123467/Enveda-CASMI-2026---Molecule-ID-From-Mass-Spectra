"""Seal two exploratory scoring entries only after all actual evidence passes."""
import json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write


def main():
    active=json.loads((ROOT/'.local/c38-active-goal.json').read_text())
    assert active['active'] and active['submissions_approved']==2 and active['submissions_used']==0
    out=ROOT/'outputs/local/c42-two-new-submissions-20261111';target=out/'submission-qualification.json';assert not target.exists()
    c39=ROOT/'outputs/local/c42-fixed-c39-validation-20261111';c43=ROOT/'outputs/local/c43-paired-confirmation-20261111'
    quality=json.loads((c39/'quality-results.json').read_text());confirmation=json.loads((c43/'confirmation-results.json').read_text())
    from confirm_c43_pair import verify as verify_confirmation_preparation
    preparation=verify_confirmation_preparation()
    assert all(quality['results'][t]['admission']['passed'] and confirmation['results'][t]['admission']['passed'] for t in ['cpu','gpu'])
    assert quality['queries']==352 and quality['all_queries_retained'] and quality['missing_truths_score_zero']
    assert confirmation['queries']==120 and confirmation['one_joint_reserved_evaluation'] and confirmation['no_recipe_or_cohort_selection']
    assert confirmation['fitting_exposure']['new_head_identity_overlaps']==0 and confirmation['all_absent_truths_zero']
    consumed=ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json';consumption=json.loads(consumed.read_text())
    assert consumption['state']=='completed' and consumption['cycle']=='C43_C42R1_PAIRED' and consumption['results_sha256']==sha(c43/'confirmation-results.json')
    assert consumption['source_sha256']==sha(ROOT/'scripts/confirm_c43_pair.py') and consumption['policy_sha256']==sha(ROOT/'configs/c43-paired-confirmation-20261111.json')
    assert all(confirmation['commitment'][k]==consumption[k] for k in confirmation['commitment'])
    evaluation=json.loads((c39/'evaluation-freeze.json').read_text())
    assert quality['evaluation_freeze_sha256']==sha(c39/'evaluation-freeze.json') and evaluation['source_sha256']==sha(ROOT/'scripts/evaluate_c42_fixed.py')
    for folder in [c39,c43/'predictions']:
        receipt=json.loads((folder/'score-receipt.json').read_text())
        for n,h in receipt['files_sha256'].items():assert sha(folder/n)==h,n
    assert preparation['head_manifest_sha256']==sha(ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111/head-manifest.json')
    head=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111';manifest=json.loads((head/'head-manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha(head/n)==h,n
    full=ROOT/'outputs/forks/c41r3-full-masked-validation-v1'
    collection=json.loads((full/'collection-audit.json').read_text());assert collection['queries']==352 and collection['forbidden_reference_rows']==0
    for n,h in collection['files_sha256'].items():assert sha(full/n)==h,n
    fidelity=json.loads((full/'acquisition-fidelity-audit.json').read_text());assert fidelity['all_original_acquisitions_retained'] and fidelity['acquisitions']==1292
    ranking=json.loads((c39/'independent-ranking-audit.json').read_text())
    assert ranking['score_receipt_sha256']==sha(c39/'score-receipt.json')
    assert all(ranking['trials'][t]['independent_RRF_exact'] for t in ['cpu','gpu'])
    trials={};paths=[c39/n for n in ['quality-results.json','evaluation-freeze.json','score-receipt.json','independent-ranking-audit.json']]
    paths += [c43/n for n in ['confirmation-results.json','preparation.json','prediction-freeze.json','fitting-exposure.csv']]
    paths += [consumed,full/'collection-audit.json',full/'acquisition-fidelity-audit.json',ROOT/'configs/c43-paired-confirmation-20261111.json',
        ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111/head-manifest.json',ROOT/'outputs/local/c42-crossmodal-engineering-20261111/production-source-audit.json']
    for t,slot in [('cpu','p20-c42-cpu-alignment'),('gpu','p21-c42-gpu-alignment')]:
        folder=ROOT/'kaggle/forks'/slot;cloud=ROOT/'outputs/forks'/(slot+'-v1');ap=cloud/'independent-production-audit.json'
        audit=json.loads(ap.read_text());assert audit['execution_qualified'] and audit['trial']==t and audit['remote_source_exact']
        assert audit['local_cloud_final_lists_exact'] and audit['independent_tail_arithmetic_exact'] and audit['changed_real_preview_lists']>0
        assert audit['rank1_changes']==audit['candidate_membership_changes']==audit['formula_slot_changes']==audit['official_identity_duplicates']==0
        assert audit['valid_rows']==400 and audit['real_preview_queries']==12 and not audit['quality_labels_read']
        assert audit['source_sha256']==sha(ROOT/'scripts/audit_c42_production.py') and sha(cloud/'submission.csv')==audit['csv_sha256']
        pin=json.loads((folder/'production-pin.json').read_text());assert pin['notebook_sha256']==sha(folder/'notebook.ipynb')==audit['notebook_sha256']
        assert pin['scientific_sha256']==audit['remote_scientific_sha256']
        paths += [folder/'notebook.ipynb',folder/'kernel-metadata.json',folder/'production-pin.json',ap,cloud/'submission.csv']
        for name,h in audit['artifact_sha256'].items():assert sha(ROOT/name)==h,name;paths.append(ROOT/name)
        trials[t]=dict(kernel='clarkkitchen/casmi26-'+slot,version=1,notebook_path=str((folder/'notebook.ipynb').relative_to(ROOT)),
            audit_path=str(ap.relative_to(ROOT)),smoke_output=str((cloud/'submission.csv').relative_to(ROOT)))
    for name in ['qualify_c42_pair.py','submit_c42_pair.py','confirm_c43_pair.py','audit_c42_production.py','audit_c42_ranking.py','evaluate_c42_fixed.py']:paths.append(ROOT/'scripts'/name)
    authorization=ROOT/'outputs/local/c38-source-support-20261111/authorization.json'
    result=dict(qualified_utc=datetime.now(timezone.utc).isoformat(),decision='execution_qualified_fixed_exploratory_pair',order=['cpu','gpu'],
        maximum_new_accepted_submissions=2,maximum_pending=1,authorization_sha256=sha(authorization),champion_retained=.480,
        trials=trials,artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},claimed_public_improvement=False,
        statistical_promotion=False,C39_directional_gates_passed=True,paired_C13_directional_gates_passed=True,
        confidence_intervals_are_uncertainty_not_proof=True,leaderboard_calibration_established=False,
        limitations=[quality['limitation'],confirmation['limitation']],
        rationale='Two predeclared new fitted heads passed actual source/acquisition, fixed directional and single paired reserved checks. Each production path passed real changed-list GPU execution, independent arithmetic, preserved-parent and local/cloud agreement checks. These remain exploratory transfer trials; only their actual public scores establish improvement.')
    write(target,result);print('C42 QUALIFIED TWO EXPLORATORY ENTRIES',sha(target),flush=True)


if __name__=='__main__':main()
