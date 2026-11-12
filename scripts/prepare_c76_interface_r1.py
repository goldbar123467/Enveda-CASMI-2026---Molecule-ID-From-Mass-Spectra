"""Prepare an explicit interface-only revision; preserve every original artifact."""
import ast
import copy
import shutil
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, replace_value
from build_general_anchor import guarded_source


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    out = base / 'interface-repair-r1'; previous = out / 'previous-sources'
    prefix = read(out / 'completed-prefix-audit.json'); assert prefix['completed_prefix_passed']
    assert not (out / 'candidate-preparation.json').exists()
    old_bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-20261112'
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'; assert not bundle.exists()
    source = ROOT / 'scripts/score_c76_complete_pair.py'
    old_source = previous / 'scripts/score_c76_complete_pair.py'
    addition = "    # Production previews also retain placeholder evidence for unprocessed IDs.\n    # Infer from every ID with spectra; preserve all other answer rows below.\n    raw = raw[raw.molecule_id.isin(active)].reset_index(drop=True)\n    public = public[public.molecule_id.isin(active)].reset_index(drop=True)\n    ours = ours[ours.molecule_id.isin(active)].reset_index(drop=True)\n"
    new_text = source.read_text(encoding='utf8'); assert new_text.count(addition) == 1
    assert new_text.replace(addition, '') == old_source.read_text(encoding='utf8')
    assert sha(old_source) == sha(old_bundle / 'score_c76_complete_pair.py')
    assert not (ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    old_manifest = read(old_bundle / 'pair-manifest.json')
    for name, pin in old_manifest['files_sha256'].items(): assert sha(old_bundle / name) == pin, name
    bundle.mkdir()
    for name in old_manifest['files_sha256']:
        original = source if name == 'score_c76_complete_pair.py' else old_bundle / name
        shutil.copyfile(original, bundle / name)
    manifest = copy.deepcopy(old_manifest)
    manifest.update(frozen_utc=stamp(), builder_sha256=sha(__file__), revision='r1_processed_ID_interface',
        prior_pair_manifest_sha256=sha(old_bundle / 'pair-manifest.json'),
        files_sha256={name: sha(bundle / name) for name in old_manifest['files_sha256']},
        interface_rule='Use all candidate evidence for every ID with spectra, and preserve every unprocessed answer row. Never filter active IDs by health or truth.',
        no_active_queries_candidates_or_acquisitions_dropped=True, no_model_or_arithmetic_change=True)
    manifest['source_files_sha256']['scripts/score_c76_complete_pair.py'] = sha(source)
    write(bundle / 'pair-manifest.json', manifest)
    dataset = 'clarkkitchen/casmi26-c76-complete-direct-pair-r1-20261112'
    metadata = read(old_bundle / 'dataset-metadata.json'); metadata.update(title='CASMI26 C76 complete direct pair r1', id=dataset)
    write(bundle / 'dataset-metadata.json', metadata)
    candidates = {}
    for trial in ['cpu', 'gpu']:
        folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
        prior = previous / 'kaggle/forks' / folder.name
        assert sha(folder / 'notebook.ipynb') == sha(prior / 'notebook.ipynb')
        notebook = read(prior / 'notebook.ipynb'); original = copy.deepcopy(notebook)
        stage = unwrap(''.join(notebook['cells'][28]['source']))
        stage = replace_value(stage, 'C76_PAIR_PIN', sha(bundle / 'pair-manifest.json'))
        notebook['cells'][28]['source'] = guarded_source(stage, 'c76_required_' + trial + '_head').splitlines(True)
        assert notebook['cells'][:28] == original['cells'][:28]
        write(folder / 'notebook.ipynb', notebook)
        meta = read(prior / 'kernel-metadata.json')
        meta.update(id='clarkkitchen/casmi26-c76-' + trial + '-production-r1', title='casmi26 C76 ' + trial + ' production r1')
        old_dataset = read(old_bundle / 'dataset-metadata.json')['id']; assert meta['dataset_sources'].count(old_dataset) == 1
        meta['dataset_sources'] = [dataset if name == old_dataset else name for name in meta['dataset_sources']]
        write(folder / 'kernel-metadata.json', meta)
        pin = read(prior / 'production-pin.json')
        pin.update(prepared_utc=stamp(), interface_revision='r1', interface_builder_sha256=sha(__file__),
            original_r0_notebook_sha256=sha(prior / 'notebook.ipynb'),
            notebook_sha256=sha(folder / 'notebook.ipynb'), metadata_sha256=sha(folder / 'kernel-metadata.json'),
            pair_manifest_sha256=sha(bundle / 'pair-manifest.json'),
            unchanged_complete_core_and_P16_prefix_exact=True, weights_features_and_ranking_math_unchanged=True,
            old_r0_preview_failed_before_head_inference=True, cloud_run_released=False, qualified_for_scoring=False)
        write(folder / 'production-pin.json', pin)
        candidates[trial] = dict(notebook_sha256=pin['notebook_sha256'], metadata_sha256=pin['metadata_sha256'],
            production_pin_sha256=sha(folder / 'production-pin.json'), kernel=meta['id'])
    write(out / 'candidate-preparation.json', dict(prepared_utc=stamp(), source_sha256=sha(__file__),
        old_complete_binding_sha256=sha(base / 'complete-pipeline-binding.json'),
        original_active_only_runtime_sha256=sha(old_source), repaired_runtime_sha256=sha(source),
        bundle=bundle.relative_to(ROOT).as_posix(), pair_manifest_sha256=sha(bundle / 'pair-manifest.json'),
        candidates=candidates, entire_core_and_P16_prefixes_byte_exact=True,
        no_model_refit=True, no_scoring_math_change=True, all_active_spectra_and_all400output_rows_required=True,
        prior_sources_preserved=True, fresh_source_predictions_read=False, quality_read=False,
        C13_read=False, scoring_entries=0, complete_r1_binding_still_required=True))
    print('C76 INTERFACE R1 CANDIDATES PREPARED; ORIGINAL PREFIX/WEIGHTS/MATH EXACT', flush=True)


if __name__ == '__main__': main()
