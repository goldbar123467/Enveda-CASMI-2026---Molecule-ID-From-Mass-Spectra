"""Freeze confirmation policy and new scoring guards before fresh outcomes.

This is not C13 execution preparation, qualification, inference or submission.
The dedicated C13 implementation must still be pinned before any new C13 run.
"""
import argparse
import ast
import subprocess
import sys
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import (BASE, BINDING, CONFIRM, CONSUMPTION, LEDGER,
    OPS_PLAN, POLICY, authorization, family)

SOURCES = ['c76_submission_contract.py', 'qualify_c76_pair.py', 'submit_c76_pair.py',
    'release_c76_production_r1.py', 'probe_c76_submission_guard.py', 'prepare_c76_final_operations.py',
    'c76_account_history.py', 'c75_port_utils.py', 'c09_submission_guard.py', 'submit_c09_external_once.py']


def main(check=False):
    authorization(); family()
    if check:
        from c76_submission_contract import verify_files
        plan = read(OPS_PLAN); verify_files(plan['files_sha256'])
        assert plan['complete_pipeline_binding_sha256'] == sha(BINDING)
        print('C76 FINAL OPERATIONS PINS EXACT; NO EXECUTION OR QUALITY CLAIM', flush=True); return
    assert not OPS_PLAN.exists()
    old_plan = BASE / 'final-operations-plan-r1.json'
    archive = BASE / 'final-operations-history-amendment-r1a'
    old = read(archive / 'previous-sources-freeze.json')
    assert old['previous_operations_plan_sha256'] == sha(old_plan)
    for name, pin in old['files_sha256'].items(): assert sha(archive / 'previous-sources' / name) == pin, name
    for path in [CONSUMPTION, LEDGER, BASE / 'source-results.json',
        BASE / 'joint-source-prediction-freeze.json', CONFIRM / 'inference-plan.json',
        ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json']:
        assert not path.exists(), path
    for arm in ['legacy', 'c76']:
        assert not (BASE / 'actual-T4-source-release' / (arm + '-push-result.json')).exists()
    for name in SOURCES: ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf8'), filename=name)
    # These are offline adversarial tests; never import/create a Kaggle client.
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/probe_c76_submission_guard.py')],
        check=True, cwd=ROOT)
    policy = read(POLICY)
    assert policy['queries'] == policy['acquisitions'] == 120
    assert policy['actual_mode_counts'] == {'positive': 60, 'negative': 60}
    assert policy['maximum_joint_consumptions'] == 1 and policy['maximum_new_scoring_submissions'] == 2
    files = [ROOT / 'scripts' / name for name in SOURCES]
    files += [POLICY, BINDING, BASE / 'source-quality-plan-r1.json', BASE / 'production-evidence-plan-r1.json',
        ROOT / 'outputs/publish/c13-mixed-casmi2022-20261110/validation-contract.json',
        ROOT / 'outputs/local/c59-c13-compatibility-20261112/channel-support-freeze.json',
        old_plan, archive / 'previous-sources-freeze.json']
    write(OPS_PLAN, dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        operations_revision='r1a_full_account_history', previous_operations_plan_sha256=sha(old_plan),
        previous_sources_archive=archive.relative_to(ROOT).as_posix() + '/previous-sources',
        complete_pipeline_binding_sha256=sha(BINDING),
        files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        confirmation_policy_frozen_before_C72_predictions_and_outcomes=True,
        C13_query_mapping_or_outcomes_read=False, C72_outcomes_read=False,
        dedicated_C13_execution_preparation_complete=False,
        dedicated_C13_execution_and_evaluation_implementation_still_required=True,
        dedicated_C13_implementation_must_be_pinned_before_C13_inference=True,
        original_R1_family_source_and_production_gates_unchanged=True,
        old_R0_dataset_readiness_check_preserved_in_bound_launcher=True,
        added_preflight_verifies_all_actual_R1_mounted_datasets=True,
        shared_existing_C42_ledger_and_lock=True,
        adversarial_offline_guard_tests_passed=True,
        all_account_history_pages_required_before_release_or_observation=True,
        durable_intent_precedes_API_and_ambiguous_response_never_auto_retries=True,
        accepted_errors_consume_entry=True, maximum_new_accepted_submissions=2,
        maximum_pending=1, actual_terminal_numeric_CPU_required_before_GPU=True,
        qualified_for_scoring=False, scoring_entries=0, no_repeat_user_approval_required=True))
    print('C76 CONFIRMATION POLICY AND SCORING GUARDS FROZEN BEFORE FRESH OUTCOMES', sha(OPS_PLAN), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); main(args.check)
