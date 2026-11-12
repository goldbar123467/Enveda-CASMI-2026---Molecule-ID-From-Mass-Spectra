"""Pin reserved mask/notebook preparation code before any fresh outcomes."""
import ast
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import (BASE, BINDING, CONFIRM, CONSUMPTION, LEDGER,
    OPS_PLAN, POLICY, authorization, family)
from c76_confirmation_preparation import CODE_PLAN, C73, ORIGINAL, TEMPLATES

SOURCES = ['c76_confirmation_preparation.py', 'build_c76_confirmation.py',
    'audit_c76_confirmation_notebooks.py', 'prepare_c76_confirmation_references.py',
    'audit_c76_confirmation_references.py', 'probe_c76_confirmation_preparation.py']


def main():
    assert authorization()['submissions_used'] == 0; family()
    assert not CODE_PLAN.exists(), 'Never overwrite a frozen preparation plan'
    for path in [CONSUMPTION, LEDGER, BASE / 'source-results.json', BASE / 'joint-source-prediction-freeze.json',
        CONFIRM / 'inference-plan.json', ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json']:
        assert not path.exists(), path
    for arm in ['legacy', 'c76']:
        assert not (BASE / 'actual-T4-source-release' / (arm + '-push-result.json')).exists()
    log = BASE / 'confirmation-preparation-probe-r1.log'
    text = log.read_text(encoding='utf8'); assert 'Ran 6 tests' in text and text.rstrip().endswith('OK')
    assert 'C76 RESERVED REFERENCE AND ALL120 INPUT ALIAS AUDIT PASSED' in text
    for name in SOURCES:
        path = ROOT / 'scripts' / name
        ast.parse(path.read_text(encoding='utf8'), filename=name)
        assert log.stat().st_mtime >= path.stat().st_mtime, ('Probe predates source', name)
    code = ROOT / 'outputs/local/c45-complete-parent-validation-20261112/frozen-p16-code'
    deployment = read(code / 'deployment.json')
    files = [ROOT / 'scripts' / name for name in SOURCES]
    files += [ROOT / 'scripts/freeze_c76_confirmation_preparation.py',
        ROOT / 'scripts/build_general_anchor.py', ROOT / 'scripts/c75_port_utils.py',
        ROOT / 'scripts/c74_complete_target.py', code / 'deployment.json', code / 'c09e_policy.json',
        *[code / name for name in deployment['source_files']],
        BINDING, OPS_PLAN, POLICY, BASE / 'confirmation-protocol-preparation-r1.json', log,
        ORIGINAL / 'validation-contract.json',
        ROOT / 'outputs/local/c59-c13-compatibility-20261112/channel-support-freeze.json']
    files += [C73 / name for name in ['reference-exclusion-freeze.json', 'independent-reference-mask-audit.json',
        'chemistry-classification-freeze.json', 'global-exclusion-chemistry.parquet',
        'main-reference-sid-exclusion.npy', 'engine2-raw-key-exclusion.json', 'original-structure-identity-witnesses.parquet']]
    for slot in TEMPLATES.values():
        folder = ROOT / 'kaggle/forks' / slot
        files += [folder / name for name in ['notebook.ipynb', 'kernel-metadata.json', 'validation-pin.json']]
    files = list(dict.fromkeys(files))
    write(CODE_PLAN, dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        complete_pipeline_binding_sha256=sha(BINDING), final_operations_plan_sha256=sha(OPS_PLAN), policy_sha256=sha(POLICY),
        files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        frozen_before_C13_preparation=True, query_mapping_read=False, query_truth_read=False,
        C72_source_predictions_or_outcomes_available=False, original_R1_scientific_core_and_models_unchanged=True,
        synthetic_tests_passed=6, synthetic_main_reference_rows=275810, synthetic_P16_reference_rows=200724,
        synthetic_full_mask_preparer_and_independent_auditor_passed=True,
        all8_preparation_entrypoints_reject_missing_prior_checks=True,
        original_R3_and_explicit_C76_inner_payloads_byteexact=True,
        only_outer_query_count_contract_and_after_core_channel_instrumentation_change=True,
        source_and_actual_production_passes_required_before_query_mapping_or_masks=True,
        actual_reserved_reference_masks_prepared=False, actual_reserved_notebooks_built=False,
        dedicated_collection_replay_sealer_and_execution_auditor_still_required=True,
        all_remaining_sources_must_be_pinned_in_inference_plan_before_new_C13_inference=True,
        scoring_entries=0, qualified_for_scoring=False))
    print('C76 RESERVED MASK/NOTEBOOK PREPARATION CODE FROZEN; REAL QUERIES STILL SEALED', sha(CODE_PLAN), flush=True)


if __name__ == '__main__': main()
