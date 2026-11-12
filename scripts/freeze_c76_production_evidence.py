"""Seal actual production verification rules before source outcomes exist."""
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    target = base / 'production-evidence-plan-r1.json'; assert not target.exists()
    binding = base / 'complete-pipeline-binding-r1.json'
    for name, pin in read(binding)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    assert not (base / 'source-results.json').exists()
    assert not (base / 'joint-source-prediction-freeze.json').exists()
    assert not (ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    files = [ROOT / 'configs/c76-production-evidence-v1.json',
        *[ROOT / 'scripts' / name for name in ['freeze_c76_production_evidence.py',
            'release_c76_production_smokes.py', 'collect_c76_production_smokes.py', 'audit_c76_cloud_parent_pair.py']],
        binding, base / 'source-quality-plan-r1.json', base / 'cpu-head-parity-r1/independent-comparison.json']
    write(target, dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        complete_pipeline_binding_sha256=sha(binding),
        files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        interpretation='Complete original C75 cross-device core gate remains failed. New frozen C76 uses actual T4 source controls and actual T4 production evidence, plus independent head/list replay on the same cloud inputs. The separately disclosed local batch8 engineering run is not T4 evidence.',
        no_scientific_or_source_quality_gate_changed=True,
        original_query_selection_and_all_acquisitions_required=True,
        local_complete_production_runtime_audit_still_required=True,
        source_and_production_cloud_jobs_not_released=True,
        C72_outcomes_read=False, C13_read=False, scoring_entries=0))
    print('C76 ACTUAL T4 PRODUCTION EVIDENCE PLAN FROZEN BEFORE FRESH SOURCE OUTCOMES', flush=True)


if __name__ == '__main__': main()
