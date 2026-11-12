"""Portable, outcome-blind C76 physical parent and fixed C71 head deployment.

The same entry point serves controlled validation, real previews and production.
It accepts spectra and prediction evidence only. It never loads query truth.
"""
import os
for _name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(_name, '2')
import torch
import argparse
import hashlib
import importlib.util
import json
import shutil
import types
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

CPU_PIN = '96c4e05c7a419f38a0299b102e9007bb317fd1ff4605d65f8663512c6f6889e6'
GPU_PIN = '85b4dcddc650ffb3036733b76473c3eaa93094b5d7b50eea1dde5436f5a52c9d'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf8')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(work, bundle, reference, output, trials=('cpu', 'gpu')):
    work, bundle, reference, output = map(Path, [work, bundle, reference, output])
    manifest = read(bundle / 'pair-manifest.json')
    for name, pin in manifest['files_sha256'].items():
        assert sha(bundle / name) == pin, name
    assert sha(bundle / 'score_c76_complete_pair.py') == sha(__file__)
    assert sha(bundle / 'cpu-model.npz') == CPU_PIN
    assert sha(bundle / 'gpu-model.npz') == GPU_PIN
    assert set(trials) <= {'cpu', 'gpu'} and trials
    assert not output.exists(), 'Keep any partial inference for inspection'
    ranker = load_module('c76_pinned_ranker', bundle / 'c67_ranker.py')
    features = load_module('c76_pinned_identity', bundle / 'identity_features.py')
    compatibility = load_module('c76_pinned_compat', bundle / 'legacy_formate_compat.py')
    chemistry = types.ModuleType('c76_pinned_mass')
    legacy = bundle / 'legacy_chem_source.py'
    exec(compile(compatibility.patch_chem_source(legacy.read_text(encoding='utf8')),
                 str(legacy), 'exec'), chemistry.__dict__)
    identity = lru_cache(maxsize=50000)(features.structure)
    receipt = read(work / 'c09e_external_receipt.json')
    assert receipt['required_stage_completed'] and receipt['official_identity_duplicates'] == 0
    assert receipt['source_files'] == manifest['P16_source_files']
    assert receipt['policy_sha256'] == manifest['P16_policy_sha256']
    for name, pin in receipt['reference_files'].items():
        assert sha(reference / name) == pin, name
    p01_path, p16_path = work / 'c09e_p01_submission.csv', work / 'c76_p16.csv'
    query_path = work / 'effective_query_spectra.parquet'
    assert sha(p01_path) == receipt['baseline_sha256']
    assert sha(p16_path) == receipt['csv_sha256']
    assert sha(query_path) == receipt['query_sha256']
    p01 = pd.read_csv(p01_path, dtype=str, keep_default_na=False)
    p16 = pd.read_csv(p16_path, dtype=str, keep_default_na=False)
    q = pd.read_parquet(query_path)
    assert not {'smiles', 'normalized_smiles', 'key', 'inchikey14', 'formula'} & set(q.columns)
    assert q.spectrum_id.is_unique and p01.molecule_id.is_unique and p16.molecule_id.is_unique
    active = set(q.molecule_id.astype(str))
    assert active and active <= set(p01.molecule_id)
    assert p01.molecule_id.tolist() == p16.molecule_id.tolist()
    execution = read(work / 'execution_receipt.json')
    assert execution['processed_molecules'] == len(active)
    physical, events = ranker.physical_parent(p01, p16, q,
        pd.read_parquet(work / 'c09e_matches.parquet'),
        pd.read_parquet(reference / 'ext_meta.parquet'), chemistry.neutral_mass)
    # P16 and the physical rule cannot alter rows without processed acquisitions.
    inactive = ~p01.molecule_id.isin(active)
    pd.testing.assert_frame_equal(p01.loc[inactive], p16.loc[inactive], check_exact=True)
    pd.testing.assert_frame_equal(p01.loc[inactive], physical.loc[inactive], check_exact=True)
    parent = physical[physical.molecule_id.isin(active)].reset_index(drop=True)
    raw = pd.read_parquet(work / 'neural_candidate_features.parquet')
    public = pd.read_parquet(work / 'neural_public.parquet')
    ours = pd.read_parquet(work / 'neural_ours.parquet')
    # Production previews also retain placeholder evidence for unprocessed IDs.
    # Infer from every ID with spectra; preserve all other answer rows below.
    raw = raw[raw.molecule_id.isin(active)].reset_index(drop=True)
    public = public[public.molecule_id.isin(active)].reset_index(drop=True)
    ours = ours[ours.molecule_id.isin(active)].reset_index(drop=True)
    for frame in [raw, public, ours]:
        assert set(frame.molecule_id) == active
        assert frame.neural_ok.astype(bool).all()
        assert np.isfinite(frame.neural_score.to_numpy()).all()
    pop = pd.read_csv(bundle / 'pool_popularity.csv', usecols=[
        'inchikey14', 'pop', 'np_coconut', 'np_lotus', 'src_natural_product'])
    rows, x, names = ranker.assemble(parent, q, raw, public, ours, pop, identity)
    assert names == read(bundle / 'feature-names.json') and x.shape == (len(rows), 101)
    output.mkdir(parents=True)
    physical.to_csv(output / 'physical-parent.csv', index=False)
    write(output / 'physical-parent-events.json', events)
    rows.to_parquet(output / 'candidates.parquet', index=False)
    np.save(output / 'features.npy', x)
    write(output / 'feature-names.json', names)
    results = {}
    for trial in trials:
        model_path = bundle / (trial + '-model.npz')
        with np.load(model_path, allow_pickle=False) as model:
            lists, scores = ranker.predict(rows, x, model, gpu=trial == 'gpu')
            # Independently reconstruct arithmetic and formula-slot placement.
            z = np.minimum(10., np.maximum(-10., (x.astype(float) - model['mean']) / model['std']))
            rebuilt = -np.log(rows.parent_rank.to_numpy(float)) + z.dot(model['weights'])
            if trial == 'gpu':
                h = z.dot(model['hidden_weight'].T) + model['hidden_bias']
                gelu = h * .5 * (1. + np.tanh((2. / np.pi)**.5 * (h + .044715 * h*h*h)))
                rebuilt += .25 * np.tanh(gelu.dot(model['output_weight'].T) + model['output_bias']).reshape(-1)
        np.testing.assert_allclose(scores, rebuilt, atol=1e-10, rtol=1e-10)
        independent = {r.molecule_id: r.smiles.split(';') for r in parent.itertuples(index=False)}
        for (_, _), group in rows.groupby(['molecule_id', 'formula'], sort=False):
            if len(group) < 2 or not group.available.all():
                continue
            ordered = sorted(group.index, key=lambda i: (-float(rebuilt[i]), int(rows.loc[i, 'parent_rank'])))
            mid = group.molecule_id.iloc[0]
            for slot, i in zip(group.parent_rank, ordered):
                independent[mid][int(slot)-1] = rows.loc[i, 'smiles']
        assert lists == independent
        final = physical.copy()
        mask = final.molecule_id.isin(active)
        final.loc[mask, 'smiles'] = final.loc[mask, 'molecule_id'].map(lambda mid: ';'.join(lists[mid]))
        changed = rank1 = 0
        for old, new in zip(physical.smiles, final.smiles):
            a, b = old.split(';'), new.split(';')
            assert 1 <= len(a) == len(b) <= 25 and set(a) == set(b)
            assert [identity(s)[2] for s in a] == [identity(s)[2] for s in b]
            assert len({identity(s)[0] for s in b}) == len(b)
            changed += a != b
            rank1 += identity(a[0])[0] != identity(b[0])[0]
        pd.testing.assert_frame_equal(final.loc[inactive], physical.loc[inactive], check_exact=True)
        final.to_csv(output / (trial + '.csv'), index=False)
        np.save(output / (trial + '-scores.npy'), scores)
        results[trial] = dict(model_sha256=sha(model_path), changed_lists=changed, rank1_changes=rank1,
            independent_scores_and_lists_exact=True, candidate_membership_changes=0,
            formula_position_changes=0, official_identity_duplicates=0)
    result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        pair_manifest_sha256=sha(bundle / 'pair-manifest.json'),
        P16_receipt_sha256=sha(work / 'c09e_external_receipt.json'),
        queries=len(active), acquisitions=len(q), valid_rows=len(physical), features=101,
        exact_unprocessed_fallback_rows=int(inactive.sum()), trials=results,
        files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        quality_labels_read=False, C13_read=False, training_performed=False,
        qualified_for_scoring=False)
    write(output / 'score-receipt.json', result)
    print('C76 COMPLETE PAIR INFERENCE FINISHED', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['work', 'bundle', 'reference', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trial', choices=['cpu', 'gpu', 'both'], default='both')
    args = parser.parse_args()
    run(args.work, args.bundle, args.reference, args.output,
        ('cpu', 'gpu') if args.trial == 'both' else (args.trial,))
