"""Freeze the explicit R1 interface repair before any fresh source prediction."""
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    repair = base / 'interface-repair-r1'; target = base / 'complete-pipeline-binding-r1.json'
    assert not target.exists()
    assert read(repair / 'independent-run-audit.json')['composed_r1_local_production_passed']
    assert read(repair / 'head-recovery-results.json')['original_all_active_12query_51acquisition_vectors_scores_lists_bitexact']
    for name in ['c72-clean-target-holdout-20261112', 'c13-confirmation-20261110']:
        assert not (ROOT / 'outputs/local' / name / 'evaluation-consumption.json').exists()
    assert not (base / 'joint-source-prediction-freeze.json').exists()
    assert not (base / 'source-results.json').exists()
    assert not (ROOT / '.local/c42-two-new-submissions-20261111.json').exists()
    old_binding_path = base / 'complete-pipeline-binding.json'; old = read(old_binding_path)
    rewired = read(repair / 'operational-path-revision.json')['files']
    permitted = set(rewired) | {'scripts/score_c76_complete_pair.py'}
    for trial in ['cpu', 'gpu']:
        permitted |= {'kaggle/forks/c76-' + trial + '-production/' + name for name in ['notebook.ipynb', 'kernel-metadata.json', 'production-pin.json']}
    changes = {}
    for name, digest in old['files_sha256'].items():
        actual = sha(ROOT / name)
        if actual != digest:
            assert name in permitted, ('Unrelated R0 science changed', name)
            original = repair / 'previous-sources' / name
            assert original.is_file() and sha(original) == digest, name
            changes[name] = dict(old_sha256=digest, new_sha256=actual,
                original_source_archive=original.relative_to(ROOT).as_posix())
    # Verify every operational edit can be undone as recorded literal path edits.
    for name, entry in rewired.items():
        text = (ROOT / name).read_text(encoding='utf8')
        assert sha(ROOT / name) == entry['new_sha256']
        for before, after in reversed(entry['literal_replacements']): text = text.replace(after, before)
        assert text == (repair / 'previous-sources' / name).read_text(encoding='utf8'), name
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'
    manifest = read(bundle / 'pair-manifest.json')
    for name, digest in manifest['files_sha256'].items(): assert sha(bundle / name) == digest, name
    for trial, expected in [('cpu', old['candidates']['cpu']['model_sha256']), ('gpu', old['candidates']['gpu']['model_sha256'])]:
        assert sha(bundle / (trial + '-model.npz')) == expected
    files = [ROOT / name for name in old['files_sha256']]
    files += [old_binding_path, Path(__file__),
        *[ROOT / 'scripts' / name for name in ['audit_c76_production_prefix.py', 'prepare_c76_interface_r1.py',
            'recover_c76_production_heads_r1.py', 'audit_c76_head_recovery_r1.py', 'audit_c76_physical_fixture.py',
            'release_c76_source_controls.py', 'evaluate_c76_source.py', 'audit_c76_source_quality.py',
            'release_c76_production_smokes.py', 'collect_c76_production_smokes.py',
            'audit_c76_cloud_parent_pair.py', 'freeze_c76_production_evidence.py']],
        ROOT / 'configs/c76-source-evaluation-v1.json', ROOT / 'configs/c76-production-evidence-v1.json',
        repair / 'prior-source-snapshot.json', repair / 'completed-prefix-audit.json',
        repair / 'candidate-preparation.json', repair / 'operational-path-revision.json',
        repair / 'head-recovery-intent.json', repair / 'head-recovery-results.json', repair / 'independent-run-audit.json',
        bundle / 'pair-manifest.json', bundle / 'dataset-metadata.json']
    files += [bundle / name for name in manifest['files_sha256']]
    files = list(dict.fromkeys(files))
    candidates = {}
    for trial in ['cpu', 'gpu']:
        folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
        pin = read(folder / 'production-pin.json')
        assert pin['interface_revision'] == 'r1' and pin['unchanged_complete_core_and_P16_prefix_exact']
        candidates[trial] = dict(kernel=read(folder / 'kernel-metadata.json')['id'],
            notebook_sha256=sha(folder / 'notebook.ipynb'), production_pin_sha256=sha(folder / 'production-pin.json'),
            model_sha256=sha(bundle / (trial + '-model.npz')))
    write(target, dict(frozen_utc=stamp(), source_sha256=sha(__file__), revision='C76_r1_processed_ID_interface',
        family='The same C76 numerical parent and unchanged C71 CPU/GPU weights; fix preview evidence interface only',
        prior_R0_binding_sha256=sha(old_binding_path), prior_R0_candidate_retired_unscored=True,
        old_R0_full_production_run_passed=False,
        explicit_changed_R0_references=changes, candidates=candidates,
        files_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        repaired_pair_manifest_sha256=sha(bundle / 'pair-manifest.json'),
        source_core_notebooks_unchanged=True, complete_core_and_P16_science_unchanged=True,
        original_C75_gate_passed=False, no_new_fitting_or_score_arithmetic=True,
        interface_rule='Feature only the complete set of IDs with spectra, never select by health/outcome; all unprocessed preview rows kept byte-for-byte in final answers.',
        all_active_test_math_unchanged_bitexact=True, unhealthy_active_row_still_rejected=True,
        local_composed_preview_and_real_changed_physical_fixture_passed=True,
        source_panel_and_controls='Original C72 full512/all2215, original actual353positive/124mixed/35negative and six predeclared controls unchanged',
        source_quality_rule='Original strict-positive full gain and nonnegative actual-mode guards against each legacy control; every absent truth zero; no recipe/cohort rescue',
        source_timing='R1 fixed and frozen before either actual fresh source run or source/confirmation outcomes',
        actual_T4_source_and_production_still_required=True, R1_actual_cloud_CPU_interface_proof_still_required=True,
        C13_read=False, maximum_C13_consumptions=1, new_scoring_entries=0, maximum_scored_entries=2,
        order='Qualified CPU then terminal numeric score, then qualified GPU; max1pending; no repeat approval',
        qualified_for_scoring=False))
    print('C76 R1 COMPLETE FAMILY REFROZEN BEFORE FRESH SOURCE OUTCOMES', sha(target), flush=True)


if __name__ == '__main__': main()
