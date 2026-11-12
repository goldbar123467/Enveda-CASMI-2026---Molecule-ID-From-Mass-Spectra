"""Verify the reserved prediction freeze without opening query truth."""
from c75_port_utils import ROOT, read, sha
from c76_submission_contract import (ACTIVE, BASE, BINDING, CONFIRM, CONSUMPTION, FAMILY,
    OPS_PLAN, POLICY, rooted, source_and_production, verify_files)

ORIGINAL = ROOT / 'outputs/local/c13-confirmation-20261110'
QUERY = ROOT / 'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet'
LABELS = ORIGINAL / 'labels.csv'


def verify_predictions(unconsumed=True):
    source_and_production()
    assert read(ACTIVE)['submissions_used'] == 0, 'Confirmation must precede either approved scoring trial'
    verify_files(read(OPS_PLAN)['files_sha256'])
    if unconsumed: assert not CONSUMPTION.exists(), 'Reserved evaluation was already started; reconcile without rerunning'
    plan_path = CONFIRM / 'inference-plan.json'; plan = read(plan_path)
    assert plan['family'] == FAMILY and plan['policy_sha256'] == sha(POLICY)
    assert plan['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert plan['frozen_before_C13_inference'] and not plan['C13_outcomes_read']
    assert plan['source_quality_audit_sha256'] == sha(BASE / 'independent-source-quality-audit.json')
    assert plan['production_pair_audit_sha256'] == sha(BASE / 'actual-T4-production-parent-pair-audit.json')
    verify_files(plan['files_sha256'])
    for field, name in [('evaluator_path', 'evaluate_c76_confirmation.py'),
        ('auditor_path', 'audit_c76_confirmation.py'), ('execution_auditor_path', 'audit_c76_confirmation_execution.py')]:
        assert plan[field] == 'scripts/' + name
        assert plan['files_sha256'][plan[field]] == sha(rooted(plan[field]))
    assert plan['files_sha256']['scripts/c76_confirmation_inputs.py'] == sha(__file__)
    assert plan['evaluator_source_sha256'] == sha(rooted(plan['evaluator_path']))
    assert plan['auditor_source_sha256'] == sha(rooted(plan['auditor_path']))
    execution_path = CONFIRM / 'execution-qualification.json'; execution = read(execution_path)
    assert execution['family'] == FAMILY and execution['passed']
    assert execution['source_sha256'] == sha(rooted(plan['execution_auditor_path']))
    assert execution['inference_plan_sha256'] == sha(plan_path)
    assert execution['queries'] == execution['acquisitions'] == 120
    for name in ['actual_two_T4s_both_controls', 'original_C76_R3_core_and_adapter_exact',
        'all_original_metadata_peaks_and_unknown_CE_exact', 'original_trailing_space_token_preserved',
        'all_prior_and_global231_chemistry_and_alias_masks_preserved', 'both_frozen_heads_independently_rebuilt']:
        assert execution[name], name
    assert execution['exact_PC_supported_IDs'] == 119 and execution['exact_Engine2_IDs'] == 120
    assert execution['forbidden_reference_rows'] == 0 and not execution['quality_labels_read']
    verify_files(execution['artifact_sha256'])
    frozen_path = CONFIRM / 'joint-prediction-freeze.json'; frozen = read(frozen_path)
    assert frozen['family'] == FAMILY and frozen['inference_plan_sha256'] == sha(plan_path)
    assert frozen['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert frozen['execution_qualification_sha256'] == sha(execution_path)
    assert frozen['queries'] == frozen['acquisitions'] == 120
    assert frozen['all_six_controls_and_both_heads_frozen_before_quality'] and not frozen['quality_labels_read']
    verify_files(frozen['files_sha256'])
    assert set(frozen['controls']) == {'legacy', 'c76'}
    paths = {}
    for arm, entries in frozen['controls'].items():
        assert set(entries) == {'P01.csv', 'P16.csv', 'physical-parent.csv'}
        for name, path in entries.items():
            paths[arm + '_' + {'P01.csv': 'P01', 'P16.csv': 'P16', 'physical-parent.csv': 'physical'}[name]] = path
    assert set(frozen['candidates']) == {'cpu', 'gpu'}; paths.update(frozen['candidates'])
    policy = read(POLICY)
    assert set(paths) == set(policy['required_controls'] + policy['additional_reported_controls'] + ['cpu', 'gpu'])
    for path in paths.values(): assert frozen['files_sha256'][path] == sha(rooted(path))
    contract = read(QUERY.with_name('validation-contract.json'))
    assert sha(QUERY) == contract['files']['validation-queries.parquet']
    assert frozen['original_query_sha256'] == sha(QUERY)
    return plan, frozen, policy, paths, contract
