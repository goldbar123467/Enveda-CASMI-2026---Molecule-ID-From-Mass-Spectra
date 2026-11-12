"""Collect the actual local full run, replay frozen P16, then both C71 heads.

This adapter never reads source truth labels or computes a quality metric. It
does not claim a cloud version or dual T4 execution. Scientific code is imported
unchanged; only Windows file-path keys in existing receipts are normalized.
"""
import argparse
import importlib.util
import shutil
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c75-local-core-port-20261112'
OUT = ROOT / 'outputs/local/c74-complete-clean-target-20261112'
PAIR = ROOT / 'outputs/local/c71-final-clean-direct-pair-20261112'
PANEL = ROOT / 'outputs/local/c72-clean-target-holdout-20261112'
COLLECTION = BASE / 'full-collection'
POP = ROOT / 'outputs/research/pop-prior/pool_popularity.csv'
PLAN = BASE / 'complete-target-adapter-freeze.json'


def native(relative):
    result = ROOT / relative.replace('\\', '/')
    assert result.resolve().is_relative_to(ROOT.resolve())
    return result


def portable_read(path):
    value = read(path)
    # Only dictionary keys explicitly defined as file paths are normalized.
    # No molecular strings or other manifest values are transformed.
    if isinstance(value, dict) and 'files_sha256' in value:
        files = value['files_sha256']
        normalized = {name.replace('\\', '/'): pin for name, pin in files.items()}
        assert len(normalized) == len(files)
        value['files_sha256'] = normalized
    return value


def verify():
    auth = read(ROOT / '.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used'] == 0 and auth['scientific_gate_hold']
    assert not (PANEL / 'evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    for filename in ['complete-candidate-freeze.json', 'complete-pipeline-binding.json']:
        for name, pin in read(PAIR / filename)['files_sha256'].items():
            assert sha(native(name)) == pin, name
    if PLAN.exists():
        frozen = read(PLAN)
        assert frozen['source_sha256'] == sha(__file__)
        for name, pin in frozen['files_sha256'].items():
            assert sha(native(name)) == pin, name


def plan():
    assert not PLAN.exists()
    verify()
    files = [Path(__file__), ROOT / 'scripts/c74_complete_target.py',
             ROOT / 'scripts/c67_ranker.py', ROOT / 'scripts/legacy_formate_compat.py',
             ROOT / 'scripts/audit_c75_run.py',
             PAIR / 'complete-pipeline-binding.json', PAIR / 'complete-candidate-freeze.json',
             PAIR / 'cpu-model.npz', PAIR / 'gpu-model.npz',
             PANEL / 'validation-freeze.json', PANEL / 'preparation-protocol.json',
             OUT / 'target-masked-reference/mask-manifest.json',
             OUT / 'target-masked-reference/independent-mask-audit.json', POP]
    write(PLAN, dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        files_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files},
        complete_recipe_previously_bound=True, normalized_manifest_fields=['files_sha256 keys only'],
        local_source_collection=str(COLLECTION.relative_to(ROOT)),
        full_core_and_acquisition_audit_required=True, fixture_exact_parity_required=True,
        queries=512, acquisitions=2215, model_fitting=False, new_source_quality_read=False,
        C13_read=False, scoring_entries=0, actual_two_T4s=False))
    print('C75 LOCAL COLLECTION/P16/PAIR ADAPTER FROZEN; NO QUALITY READ', flush=True)


def collect():
    verify(); assert PLAN.exists() and not COLLECTION.exists()
    port = BASE / 'full-port'
    done = read(port / 'run-complete.json')
    audit = read(port / 'independent-run-audit.json')
    assert read(BASE / 'fixture-parity-audit.json')['passed']
    assert audit['passed'] and audit['queries'] == done['queries'] == 512
    assert audit['acquisitions'] == done['acquisitions'] == 2215
    assert audit['run_complete_sha256'] == sha(port / 'run-complete.json')
    assert audit['source_sha256'] == sha(ROOT / 'scripts/audit_c75_run.py')
    assert done['actual_local_single_2070'] and not done['actual_two_T4s']
    working = Path(done['working'])
    assert str(working) == '/home/imsa/.local/share/enveda-c75/full/working'
    pin = read(port / 'port-freeze.json'); report = read(working / 'c41_report.json')
    assert report['validation_contract_sha256'] == pin['query_contract_sha256']
    assert report['baseline_sha256'] == sha(working / 'c41_p01.csv')
    compatibility = read(working / 'c41_legacy_compatibility.json')
    assert set(compatibility) == {'v3', 'v4'}
    for value in compatibility.values():
        assert set(value['before_sha256']) == set(value['after_sha256']) == {'chem.py', 'frag.py'}
        assert all(value['before_sha256'][name] != value['after_sha256'][name] for name in value['before_sha256'])
    names = ['c41_report.json', 'c75_gpu_allocation.json', 'c41_gpu1_preflight.json',
             'c41_p01.csv', 'c41_p14.csv', 'c41_queries.parquet', 'c41_neural_p01.parquet',
             'neural_public.parquet', 'neural_ours.parquet', 'frozen_candidate_evidence.parquet',
             'outsider_neural_features.parquet', 'effective_query_spectra.parquet',
             'c41_main_reference_mask_audit.json', 'c41_engine2_reference_mask_receipt.json',
             'c41_legacy_compatibility.json', 'execution_receipt.json', 'submission_pipeline.log']
    for name in names: assert (working / name).is_file(), name
    COLLECTION.mkdir()
    for name in names:
        shutil.copyfile(working / name, COLLECTION / name)
        assert sha(COLLECTION / name) == sha(working / name)
    write(COLLECTION / 'collection-audit.json', dict(completed_utc=stamp(),
        source_sha256=sha(__file__), adapter_freeze_sha256=sha(PLAN), queries=512,
        kernel=None, version=None, execution='Actual local original complete scientific pipeline',
        actual_single_RTX2070=True, actual_two_T4s=False, forbidden_reference_rows=0,
        formate_source_copy_verified=True, baseline_replay_exact=True,
        files_sha256={name: sha(COLLECTION / name) for name in names},
        independent_run_audit_sha256=sha(port / 'independent-run-audit.json'),
        quality_metrics_read=False, C13_read=False, scoring_entries=0))
    write(COLLECTION / 'acquisition-fidelity-audit.json', dict(completed_utc=stamp(),
        source_sha256=sha(__file__), queries=512, acquisitions=2215,
        all_original_acquisitions_retained=True, all_original_metadata_and_peaks_exact=True,
        independent_run_audit_sha256=sha(port / 'independent-run-audit.json'), passed=True))
    shutil.copyfile(port / 'independent-run-audit.json', COLLECTION / 'independent-core-health-audit.json')
    print('C75 ACTUAL FULL LOCAL COLLECTION FROZEN', flush=True)


def load_c74():
    spec = importlib.util.spec_from_file_location('c75_exact_c74', ROOT / 'scripts/c74_complete_target.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.CLOUD = COLLECTION; module.EXTRA = COLLECTION
    module.read = portable_read
    return module


def replay():
    verify(); assert PLAN.exists()
    assert read(COLLECTION / 'collection-audit.json')['adapter_freeze_sha256'] == sha(PLAN)
    family = OUT / 'candidate-family-freeze.json'
    if family.exists(): assert sha(family) == sha(PAIR / 'complete-candidate-freeze.json')
    else: shutil.copyfile(PAIR / 'complete-candidate-freeze.json', family)
    module = load_c74()
    module.replay_c39()
    write(OUT / 'local-parent-adapter-receipt.json', dict(completed_utc=stamp(),
        source_sha256=sha(__file__), adapter_freeze_sha256=sha(PLAN),
        original_c74_source_sha256=sha(ROOT / 'scripts/c74_complete_target.py'),
        complete_parent_freeze_sha256=sha(OUT / 'complete-parent-freeze.json'),
        collection_audit_sha256=sha(COLLECTION / 'collection-audit.json'),
        only_adapted_globals=['CLOUD', 'EXTRA', 'read:files_sha256 path keys'],
        original_science_unchanged=True, new_source_quality_read=False, C13_read=False,
        scoring_entries=0, actual_two_T4s=False))
    print('C75 FROZEN P16 AND PHYSICAL PARENT REPLAYED', flush=True)


def score():
    verify(); assert PLAN.exists()
    # Torch first, matching the established Linux CUDA/Arrow runtime contract.
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from c67_ranker import assemble, predict
    folder = OUT / 'clean-head-predictions'; assert not folder.exists()
    assert read(OUT / 'local-parent-adapter-receipt.json')['adapter_freeze_sha256'] == sha(PLAN)
    parent = read(OUT / 'complete-parent-freeze.json')
    for name, pin in parent['files_sha256'].items(): assert sha(OUT / name) == pin
    features = load_c74().module('c75_frozen_identity', 'c09e_features.py')
    identity = lru_cache(maxsize=50000)(features.structure)
    original = pd.read_csv(OUT / 'physical-parent.csv', dtype=str, keep_default_na=False)
    q = pd.read_parquet(OUT / 'queries.parquet')
    assert len(q) == 2215 and len(original) == original.molecule_id.nunique() == 512
    raw = pd.read_parquet(COLLECTION / 'c41_neural_p01.parquet')
    public = pd.read_parquet(COLLECTION / 'neural_public.parquet')
    ours = pd.read_parquet(COLLECTION / 'neural_ours.parquet')
    pop = pd.read_csv(POP, usecols=['inchikey14', 'pop', 'np_coconut', 'np_lotus', 'src_natural_product'])
    rows, x, names = assemble(original, q, raw, public, ours, pop, identity)
    expected = read(ROOT / 'outputs/local/c69-clean-parent-evidence-oof-20261112/feature-names.json')
    assert names == expected and x.shape == (len(rows), 101)
    folder.mkdir(); rows.to_parquet(folder / 'candidates.parquet', index=False)
    np.save(folder / 'features.npy', x); write(folder / 'feature-names.json', names)
    receipts = {}
    for trial in ['cpu', 'gpu']:
        model_path = PAIR / (trial + '-model.npz'); model = np.load(model_path)
        lists, scores = predict(rows, x, model, gpu=trial == 'gpu')
        # Reconstruct scores and formula-slot assignment separately from predict.
        z = np.minimum(10., np.maximum(-10., (x.astype(float)-model['mean'])/model['std']))
        rebuilt = -np.log(rows.parent_rank.to_numpy(float)) + z.dot(model['weights'])
        if trial == 'gpu':
            h = z.dot(model['hidden_weight'].T)+model['hidden_bias']
            gelu = h*.5*(1.+np.tanh((2./np.pi)**.5*(h+.044715*h*h*h)))
            rebuilt += .25*np.tanh(gelu.dot(model['output_weight'].T)+model['output_bias']).reshape(-1)
        np.testing.assert_allclose(scores, rebuilt, atol=1e-10, rtol=1e-10)
        independently = {item.molecule_id: item.smiles.split(';') for item in original.itertuples(index=False)}
        for (_, _), group in rows.groupby(['molecule_id', 'formula'], sort=False):
            if len(group) < 2 or not group.available.all(): continue
            ix = sorted(group.index.tolist(), key=lambda i: (-float(rebuilt[i]), int(rows.loc[i, 'parent_rank'])))
            mid = group.molecule_id.iloc[0]
            for slot, i in zip(group.parent_rank, ix): independently[mid][int(slot)-1] = rows.loc[i, 'smiles']
        assert lists == independently
        frame = original.copy(); frame['smiles'] = frame.molecule_id.map(lambda mid: ';'.join(lists[mid]))
        for old, new in zip(original.smiles, frame.smiles):
            a = old.split(';'); b = new.split(';')
            assert len(a) == len(b) and set(a) == set(b) and len(set(b)) == len(b)
            assert [identity(s)[2] for s in a] == [identity(s)[2] for s in b]
            assert len({identity(s)[0] for s in b}) == len(b)
        frame.to_csv(folder / (trial + '.csv'), index=False)
        np.save(folder / (trial + '-scores.npy'), scores)
        receipts[trial] = dict(model_sha256=sha(model_path), independent_scores_and_lists_exact=True,
            changed_lists=int((frame.smiles != original.smiles).sum()),
            membership_preserved=True, all_formula_slots_preserved=True)
    files = [p for p in folder.iterdir() if p.is_file()]
    write(folder / 'score-receipt.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        adapter_freeze_sha256=sha(PLAN), complete_pipeline_binding_sha256=sha(PAIR / 'complete-pipeline-binding.json'),
        complete_parent_freeze_sha256=sha(OUT / 'complete-parent-freeze.json'),
        queries=512, acquisitions=2215, features=101, trials=receipts,
        files_sha256={p.name: sha(p) for p in files},
        new_source_quality_read=False, C13_read=False, scoring_entries=0,
        qualified_for_scoring=False))
    print('C75 BOTH FROZEN C71 PREDICTION SETS COMPLETE; QUALITY STILL SEALED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['plan', 'collect', 'replay', 'score'])
    {'plan': plan, 'collect': collect, 'replay': replay, 'score': score}[parser.parse_args().phase]()
