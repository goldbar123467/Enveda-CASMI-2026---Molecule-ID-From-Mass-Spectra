"""Seal the existing two-trial allowance after all current-family evidence."""
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import (BASE, BINDING, CONFIRM, FAMILY, OPS_PLAN,
    OUT, QUALIFICATION, LEDGER, authorization, source_and_production, confirmation, verify_files)


def main():
    active = authorization()
    assert active['submissions_used'] == 0 and not LEDGER.exists()
    assert not QUALIFICATION.exists(), 'Never overwrite an existing qualification'
    plan = read(OPS_PLAN); verify_files(plan['files_sha256'])
    assert plan['complete_pipeline_binding_sha256'] == sha(BINDING)
    trials, evidence = source_and_production()
    evidence += confirmation()
    evidence += [OPS_PLAN, ROOT / 'outputs/local/c38-source-support-20261111/authorization.json']
    evidence += [ROOT / name for name in plan['files_sha256']]
    OUT.mkdir(exist_ok=True)
    paths = list(dict.fromkeys(evidence))
    write(QUALIFICATION, dict(qualified_utc=stamp(), family=FAMILY,
        decision='qualified_fixed_exploratory_pair', order=['cpu', 'gpu'],
        maximum_new_accepted_submissions=2, maximum_pending=1,
        complete_pipeline_binding_sha256=sha(BINDING), final_operations_plan_sha256=sha(OPS_PLAN),
        authorization_sha256=sha(ROOT / 'outputs/local/c38-source-support-20261111/authorization.json'),
        champion_retained=.480, trials=trials,
        source_gates_and_independent_audit_passed=True,
        actual_T4_production_and_same_cloud_input_local_replay_passed=True,
        once_only_paired_C13_confirmation_and_independent_audit_passed=True,
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        claimed_public_improvement=False, statistical_promotion=False,
        leaderboard_calibration_established=False, original_C75_gate_passed=False,
        limitations=[read(BASE / 'source-results.json')['limitation'],
            read(CONFIRM / 'confirmation-results.json')['limitation']],
        rationale='Unchanged complete CPU/GPU candidates passed the predeclared source and reserved stress-confirmation checks and actual production execution. Only actual public scores establish public improvement.'))
    print('C76 QUALIFIED THE TWO EXISTING APPROVED TRIALS', sha(QUALIFICATION), flush=True)


if __name__ == '__main__': main()
