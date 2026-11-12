"""Freeze complete reserved inference plumbing before actual source outcomes."""
import ast
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import (BASE, BINDING, CONFIRM, CONSUMPTION, LEDGER,
    OPS_PLAN, POLICY, authorization, family, verify_files)
from c76_confirmation_preparation import CODE_PLAN
from c76_confirmation_inference import CLOUDS, EXECUTION_CODE_PLAN, PLAN, RELEASE

SOURCES = ['c76_confirmation_inference.py', 'release_c76_confirmation.py',
    'collect_c76_confirmation.py', 'complete_c76_confirmation.py',
    'audit_c76_confirmation_execution.py', 'probe_c76_confirmation_execution.py']


def main():
    assert authorization()['submissions_used'] == 0; family()
    assert not EXECUTION_CODE_PLAN.exists()
    for path in [CONSUMPTION, LEDGER, PLAN, RELEASE, BASE / 'source-results.json',
        BASE / 'joint-source-prediction-freeze.json', CONFIRM / 'execution-qualification.json',
        ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json']:
        assert not path.exists(), path
    for cloud in CLOUDS.values(): assert not cloud.exists()
    for arm in ['legacy', 'c76']:
        assert not (BASE / 'actual-T4-source-release' / (arm + '-push-result.json')).exists()
    verify_files(read(CODE_PLAN)['files_sha256'])
    verify_files(read(OPS_PLAN)['files_sha256'])
    log = BASE / 'confirmation-execution-probe-r1-full.log'; text = log.read_text(encoding='utf8')
    assert 'Ran 8 tests' in text and text.rstrip().endswith('OK')
    for line in ['C76 COMPLETE RESERVED P16/PHYSICAL PARENT REPLAYED legacy',
        'C76 COMPLETE RESERVED P16/PHYSICAL PARENT REPLAYED c76',
        'C76 BOTH UNCHANGED RESERVED HEADS SCORED WITHOUT LABELS',
        'C76 COMPLETE RESERVED EXECUTION AND BOTH HEAD RANKINGS INDEPENDENTLY AUDITED',
        'C76 BOTH RESERVED PREDICTIONS JOINTLY SEALED']:
        assert line in text, line
    for name in SOURCES:
        path = ROOT / 'scripts' / name
        ast.parse(path.read_text(encoding='utf8'), filename=name)
        assert log.stat().st_mtime >= path.stat().st_mtime, ('Probe predates source', name)
    dependencies = ['freeze_c76_confirmation_execution.py', 'c75_port_utils.py', 'c74_complete_target.py',
        'c67_ranker.py', 'legacy_formate_compat.py', 'audit_c59_core_health.py',
        'c76_confirmation_inputs.py', 'evaluate_c76_confirmation.py', 'audit_c76_confirmation.py',
        'c76_submission_contract.py', 'submit_c76_pair.py', 'submit_c09_external_once.py']
    files = [ROOT / 'scripts' / name for name in SOURCES + dependencies]
    files += [BINDING, OPS_PLAN, POLICY, CODE_PLAN, BASE / 'confirmation-protocol-preparation-r1.json', log]
    source = ROOT / 'outputs/local/c37-selection-trace-20261111/pinned-public-source'
    files += [source / 'chem.py', source / 'frag.py']
    files = list(dict.fromkeys(files))
    write(EXECUTION_CODE_PLAN, dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        complete_pipeline_binding_sha256=sha(BINDING), final_operations_plan_sha256=sha(OPS_PLAN),
        policy_sha256=sha(POLICY), preparation_code_plan_sha256=sha(CODE_PLAN),
        files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        fixed_before_C72_predictions_or_outcomes=True, C13_query_mapping_read=False, C13_labels_read=False,
        actual_reserved_dataset_published=False, actual_reserved_core_runs=0,
        all_test_account_calls_mocked=True, all_test_core_receipts_synthetic=True,
        synthetic_tests_passed=8, synthetic_postprocessing_queries=120,
        original_P16_runtime_and_both_unchanged_real_models_executed_on_synthetic_inputs=True,
        synthetic_full_parent_head_audit_and_joint_seal_passed=True,
        known_good_no_external_promotion_path_in_integration_test=True,
        changed_same_formula_leader_and_missing_group_fallback_separately_tested=True,
        valid_mass_and_inconsistent_or_unknown_mass_promotions_separately_tested=True,
        private_upload_and_durable_lost_response_no_retry_tested=True,
        original_actual_T4_source_production_and_C13_gates_unchanged=True,
        all_actual_C13_implementation_must_be_bound_in_inference_plan_before_API_release=True,
        scoring_entries=0, qualified_for_scoring=False))
    print('C76 COMPLETE RESERVED EXECUTION CODE FROZEN; NO REAL C13 RUN OR SCORE', sha(EXECUTION_CODE_PLAN), flush=True)


if __name__ == '__main__': main()
