"""Lightweight, identity-separated coefficient sweep over cached fair NP holdout scores.

No training or spectrum preprocessing. Run with .venv-gpu for NumPy/pandas.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tail_fusion import paired_summary, partition

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "outputs/local/science-20261105"
    out.mkdir(parents=True, exist_ok=True)
    mol = pd.read_parquet(ROOT / "outputs/local/n0/cache/molecules.parquet")
    cache = ROOT / "outputs/local/seeds"
    candidates = np.load(cache / "validation_candidates_np.npz")
    models, ids = [], None
    for path in sorted((cache / "scores").glob("np_np*_e64_*.npz")):
        with np.load(path) as z:
            current = sorted(int(k[2:]) for k in z.files if k.startswith("c_"))
            if ids is None:
                ids = current
            assert ids == current, "Molecule order differs across seeds"
            assert all(candidates[str(i)][0] == i for i in ids), "Truth must be at column zero"
            cos = np.stack([z[f"c_{i}"] for i in ids])
            fp = np.stack([z[f"f_{i}"] for i in ids])
            assert cos.shape == fp.shape and np.isfinite(cos).all() and np.isfinite(fp).all()
            models.append((path.name, cos, fp))
    assert len(models) == 3 and len(ids) == 234
    mask = np.array([partition(mol.key.iloc[i]) == 0 for i in ids])

    def rr(scores):
        ranks = 1 + (scores[:, 1:] >= scores[:, :1]).sum(1)
        return np.where(ranks <= 25, 1 / ranks, 0.)

    sweep = [dict(fingerprint_factor=w, mrr25=float(rr(models[0][1] + w * models[0][2])[mask].mean()))
             for w in [0., .25, .5, 1., 2., 4.]]
    selected = max(sweep, key=lambda x: (x["mrr25"], -abs(x["fingerprint_factor"] - 1)))["fingerprint_factor"]
    results = []
    for name, cos, fp in models:
        before, after = rr(cos + fp), rr(cos + selected * fp)
        results.append(dict(model=name, development=paired_summary(before[mask], after[mask]),
                            confirmation=paired_summary(before[~mask], after[~mask])))
    report = dict(protocol="NP identities excluded from these three historical models; first-seed development fits a coefficient, frozen for all confirmations; no candidate generator",
                  n=len(ids), development=int(mask.sum()), confirmation=int((~mask).sum()),
                  selected_fingerprint_factor=selected, original_factor=1., sweep=sweep, models=results,
                  hidden_score_comparable=False)
    text = json.dumps(report, indent=2)
    (out / "np_cached_sweep.json").write_text(text)
    (out / "np_cached_sweep.log").write_text(text + "\n[0s] DONE cached NP sweep\n")
    print(text)


if __name__ == "__main__":
    main()
