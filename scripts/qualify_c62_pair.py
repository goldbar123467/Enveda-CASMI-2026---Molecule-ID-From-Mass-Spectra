"""Seal only the actual C56 pair after complete fresh and reserved admission.

Pure evidence verification: no labels, fitting, account calls or submissions.
Retired controllers remain held; only this qualification's hash is authorized.
"""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c62-two-new-submissions-20261112'
MODEL=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
SOURCE=ROOT/'outputs/local/c58-complete-target-20261112'
CONFIRM=ROOT/'outputs/local/c61-paired-confirmation-20261112'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):digest.update(chunk)
    return digest.hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def main():
    active_path=ROOT/'.local/c38-active-goal.json';active=read(active_path)
    assert active['active'] and active['submissions_approved']==2 and active['submissions_used']==0
    assert active['preserve_champion']==.480 and active['old_campaigns_stay_inactive'] and not active['hourly_automation']
    assert active['free_local_and_kaggle_compute_authorized'] and not active['new_paid_compute_authorized']
    target=OUT/'submission-qualification.json';assert not target.exists()
    ledger=ROOT/'.local/c42-two-new-submissions-20261111.json'
    assert not ledger.exists() or not read(ledger).get('submissions') and not read(ledger).get('intents')
    files={}
    def include(path,pin=None):
        path=Path(path);value=sha(path)
        if pin is not None:assert value==pin,str(path)
        key=str(path.relative_to(ROOT));assert key not in files or files[key]==value
        files[key]=value;return read(path) if path.suffix=='.json' else None
    def seal(path,source_script):
        obj=include(path);include(ROOT/'scripts'/source_script,obj['source_sha256'])
        for name,pin in obj['files_sha256'].items():include(ROOT/name,pin)
        return obj

    family=include(MODEL/'complete-candidate-freeze.json')
    policy=ROOT/'configs/c56-bounded-semantic-pair-20261112.json';include(policy,family['policy_sha256'])
    used=include(MODEL/'independent-used-quality-audit.json');assert used['passed_pair'] and used['used_queries']==387 and used['used_acquisitions']==1327
    include(ROOT/'scripts/audit_c56_used_quality.py',used['source_sha256'])
    assert used['complete_candidate_freeze_sha256']==sha(MODEL/'complete-candidate-freeze.json')
    include(MODEL/'used-development-results.json',used['old352_result_sha256'])
    include(MODEL/'used-auxiliary-results.json',used['used35_result_sha256'])
    include(MODEL/'independent-fitted-model-audit.json')
    for name in ['used-development-freeze.json','used-auxiliary-freeze.json']:
        frozen=include(MODEL/name)
        for path,pin in frozen['files_sha256'].items():include(ROOT/path,pin)

    fresh=include(SOURCE/'source-results.json');fa=include(SOURCE/'independent-quality-audit.json')
    sf=seal(SOURCE/'evaluation-freeze.json','evaluate_c58_target.py')
    include(ROOT/'scripts/audit_c58_quality.py',fa['source_sha256'])
    assert fresh['passed_pair'] and fa['passed_pair'] and fresh['queries']==fa['queries']==482 and fresh['acquisitions']==fa['acquisitions']==2125
    assert fresh['chemistry_groups']==fa['chemistry_groups']==447 and fresh['all_queries_retained'] and fresh['absent_truths_zero'] and not fresh['C13_read']
    assert fa['quality_results_sha256']==sha(SOURCE/'source-results.json') and fresh['evaluation_freeze_sha256']==fa['evaluation_freeze_sha256']==sha(SOURCE/'evaluation-freeze.json')
    include(ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json',fresh['consumption_sha256'])

    result=include(CONFIRM/'confirmation-results.json');audit=include(CONFIRM/'independent-confirmation-audit.json')
    cf=seal(CONFIRM/'prediction-freeze.json','confirm_c61_pair.py')
    include(ROOT/'scripts/audit_c61_confirmation.py',audit['source_sha256'])
    assert result['passed_pair'] and audit['passed_pair'] and result['queries']==audit['queries']==120 and result['acquisitions']==audit['acquisitions']==120
    assert result['one_joint_reserved_evaluation'] and result['all_absent_truths_zero'] and result['no_recipe_or_cohort_selection']
    assert result['fitting_exposure']['new_head_identity_overlaps']==result['fitting_exposure']['prior_used_identity_overlaps']==0
    assert audit['confirmation_results_sha256']==sha(CONFIRM/'confirmation-results.json') and audit['prediction_freeze_sha256']==sha(CONFIRM/'prediction-freeze.json')
    marker_path=ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json';marker=include(marker_path)
    assert marker['state']=='completed' and marker['cycle']=='C61_C56_COMPLETE_PAIRED' and marker['results_sha256']==sha(CONFIRM/'confirmation-results.json')
    assert audit['global_consumption_sha256']==sha(marker_path)
    assert all(marker[k]==v for k,v in result['commitment'].items())
    assert marker['source_sha256']==sha(ROOT/'scripts/confirm_c61_pair.py') and marker['prediction_freeze_sha256']==sha(CONFIRM/'prediction-freeze.json') and marker['policy_sha256']==sha(policy)
    for entry in [fresh,audit,result]:
        assert entry['passed_pair']
    for t in ['cpu','gpu']:
        assert fresh['results'][t]['admission']['passed'] and result['results'][t]['admission']['passed']

    trials={}
    for trial,entry in family['candidates'].items():
        folder=ROOT/entry['path'];cloud=ROOT/'outputs/forks'/(folder.name+'-v1')
        ap=cloud/'independent-production-audit.json';a=include(ap)
        assert a['execution_qualified'] and a['trial']==trial and a['remote_source_exact'] and a['local_cloud_final_lists_exact'] and a['independent_tail_arithmetic_exact']
        assert a['rank1_changes']==a['candidate_membership_changes']==a['formula_slot_changes']==a['official_identity_duplicates']==0
        assert a['changed_real_preview_lists']>0 and a['valid_rows']==400 and a['real_preview_queries']==12 and not a['quality_labels_read']
        assert a['notebook_sha256']==entry['notebook_sha256'] and a['remote_scientific_sha256']==entry['scientific_sha256'] and a['head_manifest_sha256']==entry['head_manifest_sha256']
        include(ROOT/'scripts/audit_c56_production.py',a['source_sha256'])
        include(folder/'notebook.ipynb',entry['notebook_sha256']);include(folder/'production-pin.json',entry['pin_sha256']);include(folder/'kernel-metadata.json')
        include(cloud/'submission.csv',a['csv_sha256'])
        for name,pin in a['artifact_sha256'].items():include(ROOT/name,pin)
        bundle=ROOT/entry['bundle'];manifest=include(bundle/'head-manifest.json',entry['head_manifest_sha256'])
        for name,pin in manifest['files'].items():include(bundle/name,pin)
        assert entry['version']==1
        trials[trial]=dict(kernel=entry['kernel'],version=1,notebook_path=str((folder/'notebook.ipynb').relative_to(ROOT)),audit_path=str(ap.relative_to(ROOT)),smoke_output=str((cloud/'submission.csv').relative_to(ROOT)))
    assert set(trials)=={'cpu','gpu'}
    for slot,collector,auditor,pin_name in [
        ('c57-full-masked-validation','collect_c57_validation.py','audit_c57_acquisitions.py','validation-pin.json'),
        ('c59-c13-main-compatibility','collect_c59_inference.py','audit_c59_acquisitions.py','compatibility-pin.json')]:
        cloud=ROOT/'outputs/forks'/(slot+'-v1');collection=include(cloud/'collection-audit.json')
        include(ROOT/'scripts'/collector,collection['source_sha256'])
        for name,pin in collection['files_sha256'].items():include(cloud/name,pin)
        fidelity=include(cloud/'acquisition-fidelity-audit.json');include(ROOT/'scripts'/auditor,fidelity['source_sha256'])
        include(ROOT/'kaggle/forks'/slot/pin_name);include(ROOT/'kaggle/forks'/slot/'notebook.ipynb')
    for name in ['c58_complete_target.py','audit_c58_reference_mask.py','c60_complete_c13.py','audit_c60_reference_mask.py','c61_exposure.py','release_c59_c13.py','build_c59_c13_compatibility.py']:
        include(ROOT/'scripts'/name)
    for name in ['qualify_c62_pair.py','submit_c62_pair.py','watch_c62_scores.py','c09_submission_guard.py','submit_c09_external_once.py']:
        include(ROOT/'scripts'/name)
    authorization=ROOT/'outputs/local/c38-source-support-20261111/authorization.json';include(authorization)
    write(target,dict(qualified_utc=datetime.now(timezone.utc).isoformat(),controller='C62',decision='execution_qualified_fixed_exploratory_pair',
        order=['cpu','gpu'],maximum_new_accepted_submissions=2,maximum_pending=1,authorization_sha256=sha(authorization),champion_retained=.480,
        trials=trials,artifact_sha256=files,claimed_public_improvement=False,statistical_promotion=False,
        all387_used_guards_passed=True,all482_fresh_complete_incremental_guards_passed=True,one_joint_C13_complete_incremental_guards_passed=True,
        leaderboard_calibration_established=False,limitations=[fresh['limitation'],result['limitation']],
        rationale='Actual clean CPU/GPU pair, complete P16 and production paths passed fixed checks. Only actual scores establish public improvement.'))
    # Never clear the hold that disables retired C42/C43 controllers.
    assert active.get('scientific_gate_hold')
    active.update(submission_controller='C62',current_pair_qualification_sha256=sha(target));write(active_path,active)
    print('C62 QUALIFIED EXACT TWO NEW EXPLORATORY ENTRIES',sha(target),flush=True)


if __name__=='__main__':main()
