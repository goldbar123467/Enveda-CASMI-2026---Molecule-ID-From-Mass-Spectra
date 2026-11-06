"""Mean prediction scores from the five pinned L06 control checkpoints.

Executed beside the unchanged public neural_scores.py on Kaggle. Each checkpoint
is loaded and scored sequentially; no weight averaging or extra training occurs.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def aligned(frame, index):
    frame = frame.set_index(['molecule_id', 'smiles'])
    assert not frame.index.has_duplicates
    assert len(frame) == len(index) and set(frame.index) == set(index)
    return frame.reindex(index)


def combine(frames):
    assert len(frames) == 5, 'All five control folds are required'
    first = frames[0].set_index(['molecule_id', 'smiles'])
    assert not first.index.has_duplicates
    result = first.copy()
    parts = [aligned(frame, first.index) for frame in frames]
    for part in parts:
        assert np.array_equal(part.processed, first.processed)
        assert part.loc[first.processed.astype(bool), 'neural_ok'].astype(bool).all()
        for column in ['n_spectra', 'n_adducts']:
            assert np.array_equal(part[column], first[column])
    for column in ['neural_cos', 'neural_fp', 'neural_score', 'neural_std']:
        values = np.stack([part[column].to_numpy(dtype=np.float64) for part in parts])
        assert np.isfinite(values).all(), column
        result[column] = values.mean(axis=0)
    result['neural_ok'] = np.logical_and.reduce([part.neural_ok.astype(bool).to_numpy() for part in parts])
    return result.reset_index()


def score(bundle, _checkpoint, evidence, queries, output, device='cuda'):
    from neural_scores import score as score_one
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle / 'long-control-manifest.json').read_text())
    assert manifest['family'] == 'control' and manifest['folds'] == list(range(5))
    frames, paths = [], []
    for fold in manifest['folds']:
        name = f'fold-{fold}.pt'
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == manifest['files'][name]
        path = output.with_name(output.stem + f'-fold-{fold}.parquet')
        score_one(str(bundle), name, evidence, queries, str(path), device)
        assert json.loads(Path(str(path) + '.errors.json').read_text()) == [], 'Input scoring failure'
        frames.append(pd.read_parquet(path)); paths.append(path.name)
        print('CONTROL FOLD SCORED', fold, flush=True)
    result = combine(frames)
    result.to_parquet(output, index=False)
    Path(str(output) + '.errors.json').write_text('[]')
    receipt = dict(family='control', models=5, folds=manifest['folds'],
                   manifest_sha256=hashlib.sha256((bundle / 'long-control-manifest.json').read_bytes()).hexdigest(),
                   aggregation='arithmetic mean of five raw neural candidate scores',
                   scored_molecules=int(result.loc[result.neural_ok.astype(bool), 'molecule_id'].nunique()),
                   files=paths, input_errors=0)
    Path(str(output) + '.ensemble.json').write_text(json.dumps(receipt, indent=2))
    print('DONE LONG CONTROL ENSEMBLE', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    score(*sys.argv[1:])
