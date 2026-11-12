"""Bind the entire repaired pair before fresh source predictions or outcomes."""
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    target = base / 'complete-pipeline-binding.json'; assert not target.exists()
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0 and auth['submissions_approved'] == 2
    panel = ROOT / 'outputs/local/c72-clean-target-holdout-20261112'
    pair = ROOT / 'outputs/local/c71-final-clean-direct-pair-20261112'
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-20261112'
    code = ROOT / 'outputs/local/c45-complete-parent-validation-20261112/frozen-p16-code'
    mask = ROOT / 'outputs/local/c74-complete-clean-target-20261112/target-masked-reference'
    assert not (panel / 'evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert not (ROOT / '.local/c42-two-new-submissions-20261111.json').exists()
    assert not (base / 'joint-source-prediction-freeze.json').exists()
    for slot in ['c73-full-masked-validation', 'c76-full-masked-validation']:
        assert not (ROOT / 'outputs/forks' / (slot + '-v1') / 'collection-audit.json').exists()
    assert read(base / 'fixture-resource-port/independent-run-audit.json')['C76_engineering_fixture_passed']
    assert read(base / 'cloud-validation-preparation/independent-source-audit.json')['passed']
    assert read(base / 'production-preparation/independent-source-audit.json')['passed']
    assert read(base / 'complete-pair-used-probe/results.json')['passed']
    for name in ['complete-candidate-freeze.json', 'complete-pipeline-binding.json']:
        for file, pin in read(pair / name)['files_sha256'].items(): assert sha(ROOT / file.replace('\\', '/')) == pin, file
    manifest = read(bundle / 'pair-manifest.json')
    for name, pin in manifest['files_sha256'].items(): assert sha(bundle / name) == pin, name
    contract = read(code / 'deployment.json')
    for name, pin in contract['source_files'].items(): assert sha(code / name) == pin, name
    for name, pin in contract['reference_files'].items(): assert sha(ROOT / 'outputs/research/2026-11-09-silver/extlib' / name) == pin, name
    mask_manifest = read(mask / 'mask-manifest.json')
    for name, pin in {**mask_manifest['reference_files'], **mask_manifest['audit_files']}.items(): assert sha(mask / name) == pin, name
    files = [Path(__file__),
        *[ROOT / 'scripts' / name for name in ['score_c76_complete_pair.py', 'c67_ranker.py',
            'legacy_formate_compat.py', 'c74_complete_target.py', 'c76_gl_portability.py',
            'prepare_c76_cloud_validation.py', 'audit_c76_cloud_source.py',
            'prepare_c76_complete_candidates.py', 'audit_c76_production_source.py',
            'complete_c76_target.py', 'collect_c76_source_controls.py', 'audit_complete_core_health.py']],
        ROOT / 'configs/c76-numerical-portability-v1.json',
        ROOT / 'configs/c76-source-controls-addendum-v1.json',
        ROOT / 'configs/c74-complete-clean-parent-20261112.json',
        pair / 'complete-pipeline-binding.json', pair / 'complete-candidate-freeze.json',
        pair / 'independent-fitted-pair-audit.json', pair / 'cpu-model.npz', pair / 'gpu-model.npz',
        panel / 'preparation-protocol.json', panel / 'validation-freeze.json',
        panel / 'independent-acquisition-audit.json', panel / 'fitted-query-alias-audit.json',
        ROOT / 'outputs/local/c73-clean-target-validation-20261112/reference-exclusion-freeze.json',
        ROOT / 'outputs/local/c73-clean-target-validation-20261112/independent-reference-mask-audit.json',
        mask / 'mask-manifest.json', mask / 'independent-mask-audit.json',
        code / 'deployment.json', code / 'c09e_policy.json',
        base / 'candidate-freeze-r1.json', base / 'extended-used-nine-r1/results.json',
        base / 'cpu-feature-parity/independent-comparison.json',
        base / 'fixture-resource-port/independent-run-audit.json',
        base / 'n1-resource-path-probe/results.json', base / 'complete-pair-used-probe/results.json',
        base / 'complete-candidates-preparation.json',
        base / 'cloud-validation-preparation/independent-source-audit.json',
        base / 'production-preparation/independent-source-audit.json',
        bundle / 'pair-manifest.json']
    files += [code / name for name in contract['source_files']]
    files += [bundle / name for name in manifest['files_sha256']]
    candidates = {}
    for trial in ['cpu', 'gpu']:
        folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
        files += [folder / name for name in ['notebook.ipynb', 'kernel-metadata.json', 'production-pin.json']]
        candidates[trial] = dict(kernel=read(folder / 'kernel-metadata.json')['id'],
            notebook_sha256=sha(folder / 'notebook.ipynb'), production_pin_sha256=sha(folder / 'production-pin.json'),
            model_sha256=sha(pair / (trial + '-model.npz')))
    for slot in ['c73-full-masked-validation', 'c76-full-masked-validation']:
        files += [ROOT / 'kaggle/forks' / slot / name for name in ['notebook.ipynb', 'kernel-metadata.json', 'validation-pin.json']]
    files = list(dict.fromkeys(files))
    write(target, dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        family='One fixed C76 numerical parent plus unchanged C71 CPU/GPU heads',
        prior_C71_complete_binding_sha256=sha(pair / 'complete-pipeline-binding.json'),
        candidates=candidates, files_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        P16_production_deployment_sha256=sha(code / 'deployment.json'),
        original_P01_scientific_payload_sha256='3f973abbb1593e356db4f6dec149c8ad99f5afa5ded8da8bab59152bd49e60e7',
        repaired_source_payload_sha256=read(ROOT / 'kaggle/forks/c76-full-masked-validation/validation-pin.json')['candidate_payload_sha256'],
        repaired_production_payload_sha256=read(ROOT / 'kaggle/forks/c76-cpu-production/production-pin.json')['production_payload_sha256'],
        numerical_rule='Original GL CPU2thread decisions; requested-device intensity; float64 atom-mass reduction. Exact helper/runner bytes pinned.',
        ranking_rule='Unchanged C71 101features and weights; same-formula slots/membership preserved; absent evidence keeps complete group.',
        resource_rule='Actual T4 uses original ICE default. Local2070 batch8 is explicitly separate engineering adaptation; no cross-batch exact-parity claim.',
        source_panel='Original fixed C72:512queries/all2215acquisitions,353positive/124mixed/35negative; chemistry/acquisition quarantine unchanged.',
        timing='C71 weights frozen before panel selection; C76 numerical engineering repair declared after metadata-only selection but before any fresh panel prediction or outcome.',
        controls='Actual T4 legacy R3 P01/P16/physical plus explicit C76 P01/P16/physical. Freeze all controls and both candidates before one joint quality read.',
        qualification='Original C72 strict-positive full gain and nonnegative actual-mode guards against each legacy control; C76 counterparts additional. Every absent truth zero; recoveries/displacements explicit. No post-outcome recipe/cohort rescue.',
        old_C75_gate_passed=False, no_model_refit=True, new_source_predictions_or_outcomes_read=False,
        C13_read=False, maximum_C13_consumptions=1, scoring_entries=0, maximum_new_scoring_entries=2,
        order='Qualified CPU first; observe actual terminal numeric result before qualified GPU. Max1pending. No repeat user approval required.',
        actual_T4_source_and_production_smokes_still_required=True, qualified_for_scoring=False))
    print('C76 ENTIRE SOURCE/PRODUCTION/PHYSICAL/CPU/GPU FAMILY FROZEN BEFORE FRESH OUTCOMES', sha(target), flush=True)


if __name__ == '__main__': main()
