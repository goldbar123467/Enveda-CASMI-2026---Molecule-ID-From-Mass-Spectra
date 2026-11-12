"""Current-family once-only C13 inference plan and collected-core commitments."""
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_confirmation_preparation import (ARMS, CODE_PLAN, ORIGINAL, PACKAGE,
    PREPARATION, SLOTS, prior)
from c76_submission_contract import (BASE, BINDING, CONFIRM, FAMILY, OPS_PLAN,
    POLICY, verify_files)

PLAN = CONFIRM / 'inference-plan.json'
EXECUTION_CODE_PLAN = BASE / 'confirmation-execution-code-plan-r1.json'
RELEASE = CONFIRM / 'release'
CLOUDS = {arm: ROOT / 'outputs/forks' / (SLOTS[arm] + '-v1') for arm in ARMS}
BUNDLE = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'


def verify_plan():
    prior(); plan = read(PLAN)
    assert plan['family'] == FAMILY and plan['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert plan['policy_sha256'] == sha(POLICY)
    assert plan['source_quality_audit_sha256'] == sha(BASE / 'independent-source-quality-audit.json')
    assert plan['production_pair_audit_sha256'] == sha(BASE / 'actual-T4-production-parent-pair-audit.json')
    assert plan['frozen_before_C13_inference'] and not plan['C13_outcomes_read']
    verify_files(plan['files_sha256'])
    verify_files(read(EXECUTION_CODE_PLAN)['files_sha256'])
    return plan


def collected(arm):
    verify_plan(); assert arm in ARMS
    cloud = CLOUDS[arm]; path = cloud / 'collection-audit.json'; audit = read(path)
    assert audit['source_sha256'] == sha(ROOT / 'scripts/collect_c76_confirmation.py')
    assert audit['arm'] == arm and audit['queries'] == audit['acquisitions'] == 120
    assert audit['actual_two_T4s'] and audit['remote_source_exact'] and audit['actual_original_image_exact']
    assert audit['baseline_replay_exact'] and audit['forbidden_reference_rows'] == 0
    assert audit['all_original_metadata_peaks_and_unknown_CE_exact']
    assert audit['original_trailing_space_token_preserved']
    assert audit['exact_PC_supported_IDs'] == 119 and audit['exact_Engine2_IDs'] == 120
    assert audit['inference_plan_sha256'] == sha(PLAN)
    verify_files(audit['files_sha256'], cloud)
    assert read(cloud / 'independent-core-health-audit.json')['passed']
    return cloud, audit


def prepare_plan():
    prior(); assert not PLAN.exists()
    code_plan = read(EXECUTION_CODE_PLAN); verify_files(code_plan['files_sha256'])
    for arm in ARMS:
        assert not (RELEASE / (arm + '-push-intent.json')).exists()
        assert not (CLOUDS[arm] / 'fetch-complete.json').exists()
    reference = read(PREPARATION / 'independent-reference-audit.json')
    notebook = read(PREPARATION / 'independent-notebook-audit.json')
    assert reference['passed'] and notebook['passed']
    assert reference['source_sha256'] == sha(ROOT / 'scripts/audit_c76_confirmation_references.py')
    assert notebook['source_sha256'] == sha(ROOT / 'scripts/audit_c76_confirmation_notebooks.py')
    verify_files(reference['artifact_sha256']); verify_files(notebook['artifact_sha256'])
    files = [BINDING, OPS_PLAN, CODE_PLAN, EXECUTION_CODE_PLAN, POLICY,
        BASE / 'confirmation-protocol-preparation-r1.json',
        BASE / 'independent-source-quality-audit.json', BASE / 'actual-T4-production-parent-pair-audit.json',
        PREPARATION / 'independent-reference-audit.json', PREPARATION / 'independent-notebook-audit.json',
        PREPARATION / 'notebook-builds.json', PACKAGE / 'validation-contract.json', PACKAGE / 'dataset-metadata.json']
    files += [ROOT / name for name in code_plan['files_sha256']]
    files += [ROOT / name for name in read(CODE_PLAN)['files_sha256']]
    files += [ROOT / name for name in reference['artifact_sha256']]
    files += [ROOT / name for name in notebook['artifact_sha256']]
    contract = read(PACKAGE / 'validation-contract.json')
    assert contract['queries'] == contract['spectra'] == 120 and not contract['truth_injection']
    for name, pin in contract['files'].items(): assert sha(PACKAGE / name) == pin; files.append(PACKAGE / name)
    manifest = read(BUNDLE / 'pair-manifest.json'); files.append(BUNDLE / 'pair-manifest.json')
    for name, pin in manifest['files_sha256'].items(): assert sha(BUNDLE / name) == pin; files.append(BUNDLE / name)
    evaluator, auditor, execution = ['scripts/' + name for name in [
        'evaluate_c76_confirmation.py', 'audit_c76_confirmation.py', 'audit_c76_confirmation_execution.py']]
    files += [ROOT / name for name in [evaluator, auditor, execution, 'scripts/c76_confirmation_inputs.py']]
    files = list(dict.fromkeys(files))
    CONFIRM.mkdir(exist_ok=True)
    write(PLAN, dict(frozen_utc=stamp(), source_sha256=sha(__file__), family=FAMILY,
        complete_pipeline_binding_sha256=sha(BINDING), policy_sha256=sha(POLICY),
        source_quality_audit_sha256=sha(BASE / 'independent-source-quality-audit.json'),
        production_pair_audit_sha256=sha(BASE / 'actual-T4-production-parent-pair-audit.json'),
        evaluator_path=evaluator, evaluator_source_sha256=sha(ROOT / evaluator),
        auditor_path=auditor, auditor_source_sha256=sha(ROOT / auditor), execution_auditor_path=execution,
        files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        queries=120, acquisitions=120, frozen_before_C13_inference=True, C13_outcomes_read=False,
        quality_labels_mounted=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 ACTUAL RESERVED INFERENCE PLAN FROZEN BEFORE EITHER CORE RUN', sha(PLAN), flush=True)


if __name__ == '__main__': prepare_plan()
