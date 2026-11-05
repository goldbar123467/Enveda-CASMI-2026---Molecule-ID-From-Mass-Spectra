"""Replay the CUSTOM consensus on Saner Turhaner's labelled panel with our kit-split holdout scorers.

The panel (`sanpier/catalogue-experiment-saner`, labelled_candidate_scores.parquet) holds 120 training molecules with
the frozen pipeline's 25-candidate lists and the answer marked. All 120 sit in the kit's validation split, so models
trained on the kit split (`local_seeds.py holdout ...`, the kit's best_validation_model.pt) never saw them. Their
neural scores replace the panel's, then rank 1 = argmax 1/(3+b) + w_top/(3+n), ranks 2-25 by w_mid (the pipeline's
leader_safe_order), and MRR@25 is measured. The fragment check and merge are not replayed. The panel is far easier
than the hidden test (baseline top-1 0.875), so this shows direction and how often rank 1 moves, not board size.

    .venv-gpu/Scripts/python.exe scripts/panel_replay.py outputs/local/saner_panel/panel_fps.npz

Candidate fingerprints come from `x8-panel-fps` (kit features.structure on Kaggle; RDKit is not installed locally).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import local_seeds as L  # noqa: E402  (kit import, cache paths)

PANEL = ROOT / "outputs/local/saner_panel/labelled_candidate_scores.parquet"
MODELS = {
    "e16 kit": L.KIT / "best_validation_model.pt",
    "e16 s1": ROOT / "outputs/local/n1/seeds/casmi26_holdout_1.pt",
    "e16 s2": ROOT / "outputs/local/n1/seeds/casmi26_holdout_2.pt",
    "e32 s3": ROOT / "outputs/local/seeds/holdout/casmi26_holdout_3_e32.pt",
    "e64 s4": ROOT / "outputs/local/seeds/holdout/casmi26_holdout_4_e64.pt",
    "e64 s5": ROOT / "outputs/local/seeds/holdout/casmi26_holdout_5_e64.pt",
    "e64 s6": ROOT / "outputs/local/seeds/holdout/casmi26_holdout_6_e64.pt",
    "e128 s8": ROOT / "outputs/forks/n5-e128-curves/seeds_out/holdout/casmi26_holdout_8_e128.pt",
    "e128 s13": ROOT / "outputs/local/seeds/holdout/casmi26_holdout_13_e128.pt",
    "e128 s14": ROOT / "outputs/local/seeds/holdout/casmi26_holdout_14_e128.pt",
    "e256 s10": ROOT / "outputs/forks/n9-e256-curves/seeds_out/holdout/casmi26_holdout_10_e256.pt",
}
SETS = {
    "panel (Saner)": None,
    "e16 x3": ["e16 kit", "e16 s1", "e16 s2"],
    "e32": ["e32 s3"],
    "e64 x3": ["e64 s4", "e64 s5", "e64 s6"],
    "e128": ["e128 s8"],
    "e256": ["e256 s10"],
    "e128+e256": ["e128 s8", "e256 s10"],
    "e128 x3": ["e128 s8", "e128 s13", "e128 s14"],
    "e128x3+e256": ["e128 s8", "e128 s13", "e128 s14", "e256 s10"],
}
GRID = [(0.0, 0.0), (0.25, 0.5), (0.25, 0.75), (0.25, 1.0), (0.3, 0.75), (0.35, 0.75), (0.4, 0.75), (0.5, 0.75), (0.5, 1.0), (1.0, 1.0)]


def rrf_order(s, w, offset=3.0):
    n = len(s)
    if w == 0:
        return np.arange(n)
    order = np.argsort(-s, kind="stable")
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n + 1)
    return np.argsort(-(1 / (offset + np.arange(1, n + 1)) + w / (offset + ranks)), kind="stable")


def leader_safe(s, w_top, w_mid):
    first = rrf_order(s, w_top)
    if w_top == w_mid:
        return first
    lead = int(first[0])
    return np.array([lead] + [int(i) for i in rrf_order(s, w_mid) if int(i) != lead])


@torch.inference_mode()
def per_model_terms(model, cfg, data, rows, F, device):
    """cos and fp terms per spectrum x candidate, as the pipeline's neural_scores computes them."""
    M = model.encode_molecules(F)
    cos, fps = [], []
    for st in range(0, len(rows), 64):
        Z, P = model.encode_spectra(*data.batch(rows[st:st + 64], device))
        cos.append((Z @ M.T).cpu().numpy())
        fps.append((P @ F.T / cfg["fp_bits"]).cpu().numpy())
    return np.concatenate(cos), np.concatenate(fps)


def main(fps_path):
    panel = pd.read_parquet(PANEL)
    z = np.load(fps_path, allow_pickle=False)
    # x8 saved the SMILES as an object array; rebuild its order (panel.smiles.unique()) instead of unpickling.
    smiles = panel.smiles.astype(str).unique()
    assert len(smiles) == len(z["fp"]) == len(z["keys"])
    assert (pd.Series(z["keys"]) == panel.drop_duplicates("smiles").metric_key.astype(str).values).mean() > 0.95
    fp_of = dict(zip(smiles.tolist(), z["fp"]))
    # Memory-mapped cache (no RAM preload: only ~120 molecules' spectra are read).
    torch.set_num_threads(2)
    cfg = json.loads((L.KIT / "config.json").read_text())
    cfg["work"] = str(L.CACHE)
    data, device = L.T.Data(cfg), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_of = {k: i for i, k in enumerate(data.mol.key.astype(str))}
    models = {n: L.load(cfg, p, device).eval() for n, p in MODELS.items() if p.is_file()}
    print("models", list(models), "missing", [n for n, p in MODELS.items() if not p.is_file()], flush=True)
    groups = []
    for mid, g in panel.groupby("molecule_id", sort=False):
        g = g.sort_values("baseline_rank", kind="stable").reset_index(drop=True)
        i = idx_of[mid.replace("holdout_", "")]
        assert data.mol.split.iloc[i] == "validation", mid
        rows = data.by_mol[i]
        rows = rows[data.spec.source.values[rows] == 0] if (data.spec.source.values[rows] == 0).any() else rows
        F = torch.tensor(np.stack([fp_of[s] for s in g.smiles.astype(str)]), device=device, dtype=torch.float32)
        terms = {n: per_model_terms(m, cfg, data, rows, F, device) for n, m in models.items()}
        groups.append((mid, g, terms))
    print("molecules", len(groups), "spectra/molecule median", int(np.median([len(next(iter(t.values()))[0]) for _, _, t in groups])))

    def scores(g, terms, names):
        if names is None:
            return g.neural_score.astype(float).values
        cos = np.mean([terms[n][0] for n in names], 0)
        fps = np.mean([terms[n][1] for n in names], 0)
        return (cos / cfg["temperature"] + cfg["fp_score_weight"] * fps).mean(0)

    out = {}
    for set_name, names in SETS.items():
        if names is not None and not all(n in models for n in names):
            continue
        res = {}
        for w_top, w_mid in GRID:
            rr, top1, moved, gained, lost, neural_top1 = [], 0, 0, 0, 0, 0
            for mid, g, terms in groups:
                s = scores(g, terms, names)
                lab = g.label.values
                order = leader_safe(s, w_top, w_mid)
                hit = np.flatnonzero(lab[order])
                rr.append(1 / (hit[0] + 1) if len(hit) and hit[0] < 25 else 0.0)
                top1 += int(lab[order[0]] == 1)
                neural_top1 += int(lab[int(np.argmax(s))] == 1)
                if order[0] != 0:
                    moved += 1
                    gained += int(lab[order[0]] == 1)
                    lost += int(lab[0] == 1)
            res[f"{w_top:g}/{w_mid:g}"] = dict(mrr=round(float(np.mean(rr)), 4), top1=top1, moved=moved, gained=gained, lost=lost)
        res["neural_top1"] = neural_top1
        out[set_name] = res
        print(f"{set_name:14s} neural-alone top1 {neural_top1:3d}/120 | " + " | ".join(
            f"{k} {v['mrr']:.4f} t1 {v['top1']} mv {v['moved']} +{v['gained']}/-{v['lost']}" for k, v in res.items() if k != "neural_top1"),
            flush=True)
    (ROOT / "outputs/local/saner_panel/replay_report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
