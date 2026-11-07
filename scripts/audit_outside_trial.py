"""Independently reconstruct P13 outsider fusion from downloaded smoke outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PIN = 'c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46'


def indexed(path):
    frame = pd.read_parquet(path)
    assert frame[['molecule_id', 'smiles']].notna().all().all()
    frame['molecule_id'] = frame.molecule_id.astype(str)
    frame['_repeat'] = frame.groupby(['molecule_id', 'smiles'], sort=False).cumcount()
    return frame.set_index(['molecule_id', 'smiles', '_repeat'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    folder = parser.parse_args().folder
    audit = json.loads((folder/'p13_model_audit.json').read_text())
    assert audit['checkpoint_sha256'] == PIN
    receipt = json.loads((folder/'p13_qualification_receipt.json').read_text())
    assert receipt['real_scored_molecules'] == 12 and receipt['merge_status'] == 'applied'
    public, private, final = [indexed(folder/name) for name in
                             ['outsider_public.parquet', 'outsider_ours.parquet', 'outsider_neural_features.parquet']]
    assert public.index.is_unique and set(public.index) == set(private.index) == set(final.index)
    private, final = private.reindex(public.index), final.reindex(public.index)
    expected = np.zeros(len(final))
    # Independent NumPy grouping and population standard deviation, no runtime helper imports.
    for molecule in public.index.get_level_values(0).unique():
        positions = np.flatnonzero(public.index.get_level_values(0) == molecule)
        a, b = public.iloc[positions], private.iloc[positions]
        za, zb = [f.neural_score.fillna(0).to_numpy(dtype=float) for f in [a, b]]
        za = (za-za.mean()) / max(za.std(ddof=0), 1e-6)
        zb = (zb-zb.mean()) / max(zb.std(ddof=0), 1e-6)
        ok_a, ok_b = [f.neural_ok.fillna(False).to_numpy(dtype=bool) for f in [a, b]]
        expected[positions] = np.where(ok_a & ok_b, (za+zb)/2, np.where(ok_a, za, np.where(ok_b, zb, 0)))
    assert np.isfinite(final.neural_score).all()
    error = float(np.abs(final.neural_score.to_numpy()-expected).max())
    assert error < 1e-6, error
    ok = public.neural_ok.fillna(False).astype(bool) | private.neural_ok.fillna(False).astype(bool)
    assert np.array_equal(final.neural_ok.astype(bool), ok)
    assert ok.any()
    for column in ['neural_cos', 'neural_fp', 'neural_std']:
        a, b = public[column].to_numpy(), private[column].to_numpy()
        oa, ob = public.neural_ok.fillna(False).to_numpy(bool), private.neural_ok.fillna(False).to_numpy(bool)
        values = np.where(oa & ob, (a+b)/2, np.where(oa, a, np.where(ob, b, 0)))
        assert np.allclose(final[column], values, atol=1e-6, rtol=0), column
    report = dict(rows=len(final), outsider_molecules=int(ok.groupby(level=0).any().sum()),
                  duplicate_candidate_rows=int(public.index.droplevel(2).duplicated().sum()),
                  independent_blend_max_abs_error=error, checkpoint_sha256=PIN,
                  csv_sha256=hashlib.sha256((folder/'submission.csv').read_bytes()).hexdigest(),
                  smoke_only=True, competition_score=None)
    (folder/'independent_outside_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
