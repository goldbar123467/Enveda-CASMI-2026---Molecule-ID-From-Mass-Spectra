"""Current-family source/confirmation/production prerequisites; stdlib only.

No function matches predictions to labels. These checks cannot manufacture
missing inference, confirmation or scoring evidence.
"""
import math
from pathlib import Path
from c75_port_utils import ROOT, read, sha

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
BINDING = BASE / 'complete-pipeline-binding-r1.json'
CONFIRM = BASE / 'reserved-confirmation-r1'
POLICY = ROOT / 'configs/c76-c13-confirmation-v1.json'
OPS_PLAN = BASE / 'final-operations-plan-r1a.json'
OUT = BASE / 'two-new-scoring-trials-r1'
QUALIFICATION = OUT / 'submission-qualification.json'
LEDGER = ROOT / '.local/c42-two-new-submissions-20261111.json'
ACTIVE = ROOT / '.local/c38-active-goal.json'
CONSUMPTION = ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json'
COMP = 'enveda-CASMI26-molecule-id-mass-spectra'
FAMILY = 'C76_R1_COMPLETE_C71_PAIR'
TRIALS = ('cpu', 'gpu')


def rooted(name):
    path = (ROOT / str(name).replace('\\', '/')).resolve()
    assert path.is_relative_to(ROOT.resolve()), ('Artifact outside workspace', name)
    return path


def verify_files(pins, base=ROOT):
    assert pins, 'Empty artifact commitment'
    for name, pin in pins.items():
        path = (base / str(name).replace('\\', '/')).resolve()
        assert path.is_relative_to(base.resolve()), name
        assert sha(path) == pin, name


def authorization():
    active = read(ACTIVE)
    assert active['active'] and active['submissions_approved'] == 2
    assert 0 <= active['submissions_used'] <= 2
    assert active['free_local_and_kaggle_compute_authorized']
    assert not active['new_paid_compute_authorized'] and not active['hourly_automation']
    assert active['preserve_champion'] == .480
    return active


def family():
    binding = read(BINDING); verify_files(binding['files_sha256'])
    assert binding['revision'] == 'C76_r1_processed_ID_interface'
    assert binding['no_new_fitting_or_score_arithmetic'] and binding['source_core_notebooks_unchanged']
    return binding


def source_and_production():
    authorization(); binding = family()
    quality_path = BASE / 'source-results.json'; quality = read(quality_path)
    audit_path = BASE / 'independent-source-quality-audit.json'; audit = read(audit_path)
    assert quality['both_source_gates_passed'] and audit['passed'] and audit['both_source_gates_passed']
    assert quality['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert audit['source_results_sha256'] == sha(quality_path)
    assert audit['source_sha256'] == sha(ROOT / 'scripts/audit_c76_source_quality.py')
    assert quality['source_sha256'] == sha(ROOT / 'scripts/evaluate_c76_source.py')
    verify_files(quality['files_sha256'], BASE)
    consumed_path = ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json'
    consumed = read(consumed_path)
    assert consumed['state'] == 'complete' and consumed['source_results_sha256'] == sha(quality_path)
    frozen_path = BASE / 'joint-source-prediction-freeze.json'; frozen = read(frozen_path)
    assert consumed['joint_prediction_freeze_sha256'] == sha(frozen_path)
    assert frozen['all_six_controls_and_both_heads_frozen_before_quality']
    verify_files(frozen['files_sha256'])
    plan = read(BASE / 'source-quality-plan-r1.json')
    assert plan['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert plan['evaluator_source_sha256'] == quality['source_sha256']
    assert plan['auditor_source_sha256'] == audit['source_sha256']
    assert plan['policy_sha256'] == sha(ROOT / 'configs/c76-source-evaluation-v1.json')
    assert quality['quality_plan_sha256'] == sha(BASE / 'source-quality-plan-r1.json')
    local_path = BASE / 'interface-repair-r1/independent-run-audit.json'
    local = read(local_path); assert local['passed'] and local['inactive_rows_preserved_exact'] == 388
    cloud_path = BASE / 'cpu-head-parity-r1/independent-comparison.json'
    assert read(cloud_path)['passed']
    production_plan = BASE / 'production-evidence-plan-r1.json'
    verify_files(read(production_plan)['files_sha256'])
    pair_path = BASE / 'actual-T4-production-parent-pair-audit.json'; pair = read(pair_path)
    assert pair['passed'] and pair['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert pair['source_sha256'] == sha(ROOT / 'scripts/audit_c76_cloud_parent_pair.py')
    assert pair['both_actual_two_T4_runs'] and pair['both_same_cloud_input_local_head_replays_passed']
    assert pair['original_P01_P16_and_physical_parents_exact_across_both_actual_T4_runs']
    evidence = [BINDING, quality_path, audit_path, consumed_path, frozen_path,
        local_path, cloud_path, pair_path, production_plan, BASE / 'source-quality-plan-r1.json']
    trials = {}
    for trial in TRIALS:
        folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
        cloud = ROOT / 'outputs/forks' / ('c76-' + trial + '-production-v1')
        ap = cloud / 'independent-production-audit.json'; a = read(ap)
        assert pair['individual_actual_production_audits_sha256'][trial] == sha(ap)
        assert a['execution_qualified'] and a['trial'] == trial
        assert a['remote_source_exact'] and a['actual_original_image_exact'] and a['actual_two_T4s']
        assert a['same_cloud_input_local_head_replay_exact'] and a['all_features_bitexact'] and a['final_lists_exact']
        assert a['independent_head_arithmetic_checked'] and a['membership_and_formula_slots_preserved']
        assert a['actual_query_count'] == 12 and a['acquisitions'] == 41 and a['valid_rows'] == 400
        assert a['exact_inactive_fallback_rows'] == 388 and a['changed_real_preview_lists'] > 0
        assert a['complete_pipeline_binding_sha256'] == sha(BINDING)
        assert a['source_sha256'] == sha(ROOT / 'scripts/collect_c76_production_smokes.py')
        verify_files(a['artifact_sha256'])
        pin = read(folder / 'production-pin.json'); metadata = read(folder / 'kernel-metadata.json')
        expected = binding['candidates'][trial]
        assert sha(folder / 'notebook.ipynb') == expected['notebook_sha256'] == pin['notebook_sha256']
        assert sha(folder / 'production-pin.json') == expected['production_pin_sha256']
        assert metadata['id'] == expected['kernel']
        pushed = read(BASE / 'actual-T4-production-release' / (trial + '-push-result.json'))
        assert pushed['version'] == 1 and not pushed['error']
        assert pushed['actual_kernel'] == pushed['requested_kernel'] == expected['kernel']
        trials[trial] = dict(kernel=pushed['actual_kernel'], version=1,
            notebook_path=(folder / 'notebook.ipynb').relative_to(ROOT).as_posix(),
            metadata_path=(folder / 'kernel-metadata.json').relative_to(ROOT).as_posix(),
            audit_path=ap.relative_to(ROOT).as_posix(),
            smoke_output=(cloud / 'submission.csv').relative_to(ROOT).as_posix(),
            csv_sha256=sha(cloud / 'submission.csv'))
        evidence += [ap, folder / 'notebook.ipynb', folder / 'kernel-metadata.json',
            folder / 'production-pin.json', cloud / 'submission.csv']
        evidence += [rooted(name) for name in a['artifact_sha256']]
    return trials, evidence


def confirmation():
    """Require the new once-only execution/evaluation audit; never old C43/C59."""
    policy = read(POLICY)
    plan_path = CONFIRM / 'inference-plan.json'; plan = read(plan_path)
    assert plan['family'] == FAMILY and plan['policy_sha256'] == sha(POLICY)
    assert plan['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert plan['frozen_before_C13_inference'] and not plan['C13_outcomes_read']
    assert plan['source_quality_audit_sha256'] == sha(BASE / 'independent-source-quality-audit.json')
    assert plan['production_pair_audit_sha256'] == sha(BASE / 'actual-T4-production-parent-pair-audit.json')
    verify_files(plan['files_sha256'])
    for field, expected in [('evaluator_path', 'scripts/evaluate_c76_confirmation.py'),
        ('auditor_path', 'scripts/audit_c76_confirmation.py'),
        ('execution_auditor_path', 'scripts/audit_c76_confirmation_execution.py')]:
        assert plan[field] == expected
        assert plan['files_sha256'][expected] == sha(rooted(expected))
    execution_path = CONFIRM / 'execution-qualification.json'; execution = read(execution_path)
    assert execution['family'] == FAMILY and execution['passed']
    assert execution['source_sha256'] == sha(rooted(plan['execution_auditor_path']))
    assert execution['inference_plan_sha256'] == sha(plan_path)
    assert execution['queries'] == execution['acquisitions'] == 120
    assert execution['actual_two_T4s_both_controls'] and execution['original_C76_R3_core_and_adapter_exact']
    assert execution['all_original_metadata_peaks_and_unknown_CE_exact']
    assert execution['original_trailing_space_token_preserved']
    assert execution['exact_PC_supported_IDs'] == 119 and execution['exact_Engine2_IDs'] == 120
    assert execution['forbidden_reference_rows'] == 0
    assert execution['all_prior_and_global231_chemistry_and_alias_masks_preserved']
    assert execution['both_frozen_heads_independently_rebuilt'] and not execution['quality_labels_read']
    verify_files(execution['artifact_sha256'])
    frozen_path = CONFIRM / 'joint-prediction-freeze.json'; frozen = read(frozen_path)
    assert frozen['family'] == FAMILY and frozen['inference_plan_sha256'] == sha(plan_path)
    assert frozen['complete_pipeline_binding_sha256'] == sha(BINDING)
    assert frozen['queries'] == frozen['acquisitions'] == 120
    assert frozen['all_six_controls_and_both_heads_frozen_before_quality']
    assert frozen['execution_qualification_sha256'] == sha(execution_path)
    assert not frozen['quality_labels_read']
    verify_files(frozen['files_sha256'])
    result_path = CONFIRM / 'confirmation-results.json'; result = read(result_path)
    audit_path = CONFIRM / 'independent-confirmation-audit.json'; audit = read(audit_path)
    consumed = read(CONSUMPTION)
    assert consumed['state'] == 'complete' and consumed['family'] == FAMILY
    assert consumed['results_sha256'] == sha(result_path)
    assert consumed['joint_prediction_freeze_sha256'] == sha(frozen_path)
    assert consumed['policy_sha256'] == result['policy_sha256'] == sha(POLICY)
    assert result['inference_plan_sha256'] == sha(plan_path)
    assert result['joint_prediction_freeze_sha256'] == sha(frozen_path)
    assert result['family'] == FAMILY and result['queries'] == result['acquisitions'] == 120
    assert result['actual_mode_counts'] == policy['actual_mode_counts']
    assert result['both_confirmation_gates_passed'] and result['all_absent_truths_zero']
    assert result['individual_confirmation_gates'] == {'cpu': True, 'gpu': True}
    assert result['all_original_queries_retained'] and result['no_recipe_or_cohort_selection']
    assert audit['passed'] and audit['both_confirmation_gates_passed']
    assert audit['confirmation_results_sha256'] == sha(result_path)
    assert audit['every_rank_and_exact_gate_independently_rebuilt']
    assert audit['all_query_accounting_independently_rebuilt']
    assert result['source_sha256'] == plan['evaluator_source_sha256']
    assert audit['source_sha256'] == plan['auditor_source_sha256']
    assert sha(rooted(plan['evaluator_path'])) == result['source_sha256']
    assert sha(rooted(plan['auditor_path'])) == audit['source_sha256']
    verify_files(result['files_sha256'], CONFIRM)
    return [POLICY, plan_path, execution_path, frozen_path, result_path, audit_path, CONSUMPTION,
        *[rooted(name) for name in plan['files_sha256']],
        *[rooted(name) for name in frozen['files_sha256']],
        *[rooted(name) for name in execution['artifact_sha256']]]


def qualified():
    authorization(); q = read(QUALIFICATION)
    assert q['family'] == FAMILY and q['decision'] == 'qualified_fixed_exploratory_pair'
    assert q['maximum_new_accepted_submissions'] == 2 and q['maximum_pending'] == 1
    assert q['order'] == list(TRIALS) and q['champion_retained'] == .480
    assert not q['claimed_public_improvement']
    assert q['complete_pipeline_binding_sha256'] == sha(BINDING)
    verify_files(q['artifact_sha256'])
    assert set(q['trials']) == set(TRIALS)
    return q


def validate_ledger(ledger, qualification_sha):
    assert ledger['maximum_new_accepted_submissions'] == 2 and ledger['maximum_pending'] == 1
    assert ledger['family'] == FAMILY, 'A different campaign already used the shared allowance'
    records, intents = ledger['submissions'], ledger['intents']
    assert len(records) <= 2 and len(intents) <= 2
    assert [r['trial'] for r in records] in [[], ['cpu'], ['cpu', 'gpu']]
    assert [i['trial'] for i in intents] in [[], ['cpu'], ['cpu', 'gpu']]
    assert len({str(r['reference']) for r in records}) == len(records)
    assert len({i['trial'] for i in intents}) == len(intents)
    for item in [*records, *intents]:
        assert item['trial'] in TRIALS and item['qualification_sha256'] == qualification_sha
        assert item['marker'] == marker(item['trial'])
    for item in intents:
        if item['resolved']:
            assert len([r for r in records if r['trial'] == item['trial']]) == 1, 'Resolved intent without its accepted record'
    for record in records:
        matching = [i for i in intents if i['trial'] == record['trial']]
        assert len(matching) == 1 and matching[0]['resolved']
        assert str(matching[0]['reference']) == str(record['reference'])
        for key in ['kernel', 'version', 'csv_sha256', 'remote_scientific_sha256']:
            assert matching[0][key] == record[key]
    return ledger


def marker(trial):
    assert trial in TRIALS
    return 'C76R1-' + trial.upper() + '-20261112'


def new_release_allowed(trial, rows, ledger):
    from c09_submission_guard import TERMINAL
    assert trial in TRIALS
    assert not any(r['status'] not in TERMINAL for r in rows), 'An account scoring entry is pending'
    assert not any(not i['resolved'] for i in ledger['intents']), 'Reconcile the unresolved durable intent first'
    assert len(ledger['submissions']) < 2
    assert not any(r['trial'] == trial for r in ledger['submissions']), 'This approved trial was already accepted'
    assert not any(marker(t) in r['description'] for r in rows for t in TRIALS
        if not any(x['trial'] == t for x in ledger['submissions'])), 'Reconcile existing account marker first'
    if trial == 'cpu':
        assert not ledger['submissions'], 'CPU must be first'
    else:
        assert len(ledger['submissions']) == 1 and ledger['submissions'][0]['trial'] == 'cpu'
        first = ledger['submissions'][0]
        actual = [r for r in rows if r['reference'] == str(first['reference'])]
        assert len(actual) == 1, 'First accepted reference is absent from account history'
        a = actual[0]; score = a['public_score']
        assert a['success'] and a['status'] == 'COMPLETE' and not a['error']
        assert score is not None and math.isfinite(score), 'Actual terminal numeric CPU score required'
        assert marker('cpu') in a['description']
        first.update(a)
