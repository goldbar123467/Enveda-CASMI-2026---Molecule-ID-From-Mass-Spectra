"""Train and evaluate seeds of Saner Turhaner's spectrum-structure scorer on the workstation GPU (RTX 2070).

Same recipe as the Kaggle kernels n1/n2/n3 (kit `sanpier/casmi-spectrum-model-v1`, cache from n0-scorer-prep):

    .venv-gpu/Scripts/python.exe scripts/local_seeds.py holdout 3 4 5      # train-split runs -> holdout/casmi26_holdout_<k>.pt
    .venv-gpu/Scripts/python.exe scripts/local_seeds.py refit 17 18        # full-data refits -> refit/casmi26_seed_<k>.pt
    .venv-gpu/Scripts/python.exe scripts/local_seeds.py diag               # ensemble-size curve on the kit's 1000-molecule holdout
    ... holdout 1 2 3 --split np / diag --split np                         # same on held-out Enveda natural products
    ... holdout 1 --split np --epochs 32                                   # epoch-count check (history has every epoch)

Seed k uses config seed 20261023 + 1000*k (the kit's own model is k = 0). Holdout models are the kit's
best_validation_model.pt, n1's holdout seeds 1-2 (outputs/local/n1/seeds) and every local holdout run.
CPU use is kept low (2 torch threads); the data arrays are preloaded into RAM once per process.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# CASMI_KIT / CASMI_CACHE / CASMI_OUT override the paths (Kaggle jobs from scripts/build_gpu_job.py set them).
KIT = Path(os.environ.get("CASMI_KIT", ROOT / "outputs/local/kit"))
CACHE = Path(os.environ.get("CASMI_CACHE", ROOT / "outputs/local/n0/cache"))
OUT = Path(os.environ.get("CASMI_OUT", ROOT / "outputs/local/seeds"))
N1_HOLDOUT = ROOT / "outputs/local/n1/seeds"
BASE_SEED = 20261023
T0 = time.time()
# Config overrides from --set key=value (e.g. width=256 layers=4); they become part of every checkpoint/report name.
OVERRIDES = {}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


sys.path.insert(0, str(KIT))
import train as T  # noqa: E402
from model import SpectrumCandidateModel  # noqa: E402

# hard_pools is deterministic in (allowed ids, count); build each distinct pool set once per process.
_pool_cache, _hard_pools = {}, T.hard_pools


def cached_pools(mol, allowed, count):
    a = np.asarray(allowed)
    key = (len(a), int(a.sum()), int(a[0]), count)
    if key not in _pool_cache:
        t = time.time()
        _pool_cache[key] = _hard_pools(mol, a, count)
        log("POOLS", key, round(time.time() - t), "s")
    return _pool_cache[key]


T.hard_pools = cached_pools


def setup(split="kit"):
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.set_num_threads(2)
    cfg = json.loads((KIT / "config.json").read_text())
    cfg.update(OVERRIDES)
    cfg["work"] = str(CACHE)
    data = T.Data(cfg)
    if split == "np":
        # Test-like holdout: the Enveda natural-product molecules listed in np_holdout_keys.txt (built from x6) leave
        # training entirely; the kit's random validation molecules go back into training.
        keys = set((OUT / "np_holdout_keys.txt").read_text().split())
        is_val = data.mol.key.isin(keys).values
        data.mol["split"] = np.where(is_val, "validation", "train")
        data.train, data.val = np.where(~is_val)[0], np.where(is_val)[0]
        log("NP SPLIT", len(keys), "keys listed,", len(data.val), "held out")
    t = time.time()
    data.x = np.array(data.x)
    data.meta = np.array(data.meta)
    data.fp = np.array(data.fp)
    log("PRELOADED", round(time.time() - t), "s;", len(data.mol), "molecules", len(data.spec), "spectra",
        len(data.train), "train", len(data.val), "validation")
    return cfg, data, torch.device("cuda")


def load(cfg, path, device):
    m = SpectrumCandidateModel(cfg).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
    return m.eval()


def candidates(cfg, data, split):
    path = OUT / f"validation_candidates_{split}.npz"
    if path.is_file():
        z = np.load(path)
        return {int(k): z[k] for k in z.files}
    c = T.candidate_sets(data, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(i): v for i, v in c.items()})
    return c


@torch.inference_mode()
def model_scores(cfg, data, m, cands, device):
    """{molecule: (cos term, fp term)} for one model over every candidate set (gold at index 0)."""
    out = {}
    for i, ids in cands.items():
        sr = data.by_mol[i]
        sr = sr[data.spec.source.values[sr] == 0]
        if not len(sr):
            continue
        zsum = 0
        psum = 0
        n = 0
        for start in range(0, len(sr), cfg["batch_size"]):
            batch = sr[start:start + cfg["batch_size"]]
            z, p = m.encode_spectra(*data.batch(batch, device))
            zsum = zsum + z.sum(0)
            psum = psum + p.sum(0)
            n += len(batch)
        fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
        cos = (m.encode_molecules(fp) @ (zsum / n)) / cfg["temperature"]
        fpt = cfg["fp_score_weight"] * (fp @ (psum / n)) / cfg["fp_bits"]
        out[i] = (cos.float().cpu().numpy(), fpt.float().cpu().numpy())
    return out


def metrics(score_lists):
    ranks = np.array([1 + int(np.sum(s[1:] >= s[0])) for s in score_lists])
    return dict(mrr25=float(np.mean(np.where(ranks <= 25, 1 / ranks, 0.))), top1=float(np.mean(ranks == 1)),
                top5=float(np.mean(ranks <= 5)), n=len(ranks))


def ensemble(per_model, keys, mode="raw"):
    """Average the models' total scores (raw: what the submission's CKPT_ENS module does)."""
    lists = []
    for i in keys:
        parts = [pm[i][0] + pm[i][1] for pm in per_model]
        if mode == "z":
            parts = [(s - s.mean()) / (s.std() + 1e-9) for s in parts]
        lists.append(np.mean(parts, axis=0))
    return metrics(lists)


def tag_of(epochs, cosine, lr=None):
    """Checkpoint/report suffix: '' for the kit recipe (16 epochs, constant LR 2e-4), else
    _e<epochs>[c][_lr<lr>][_<key><value>-...] (the last part from --set overrides)."""
    if epochs == 16 and not cosine and lr is None and not OVERRIDES:
        return ""
    return (f"_e{epochs}" + ("c" if cosine else "") + (f"_lr{lr:g}" if lr is not None else "")
            + ("_" + "-".join(f"{k}{v}" for k, v in sorted(OVERRIDES.items())) if OVERRIDES else ""))


def use_cosine(total_steps):
    """Swap torch.optim.AdamW (which train_stage builds) for one whose LR follows a cosine from lr to 0."""
    base = torch.optim.AdamW

    class CosineAdamW(base):
        def __init__(self, params, lr, **kw):
            super().__init__(params, lr=lr, **kw)
            self._lr0, self._t = lr, 0

        def step(self, closure=None):
            self._t += 1
            f = 0.5 * (1 + math.cos(math.pi * min(self._t, total_steps) / total_steps))
            for g in self.param_groups:
                g["lr"] = self._lr0 * f
            return super().step(closure)

    torch.optim.AdamW = CosineAdamW
    return base


def train(kind, seeds, epochs, split, cosine=False, lr=None):
    cfg, data, device = setup(split)
    tag = kind if split == "kit" else f"{kind}_{split}"
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    report_path = OUT / f"{tag}_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    for k in seeds:
        name = f"casmi26_holdout_{k}.pt" if kind == "holdout" else f"casmi26_seed_{k}.pt"
        name = name.replace(".pt", tag_of(epochs, cosine, lr) + ".pt")
        if (folder / name).is_file():
            log("SKIP", name, "exists")
            continue
        # Unique per checkpoint: two GPUs running the same seed with different --set values must not share (and
        # delete) one run folder (n13 lost its width-384 runs that way).
        run = OUT / "runs" / f"{tag}_{name[:-3]}"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        # Fixed recipe like the refits: no early stopping, keep the last epoch. The kit's own holdout run keeps the
        # best epoch on its validation set, which would select on the very molecules the diagnostic scores.
        c = dict(cfg, seed=BASE_SEED + 1000 * k, patience=10 ** 6, **({"lr": lr} if lr is not None else {}))
        plain = use_cosine(epochs * cfg["steps_per_epoch"]) if cosine else None
        t = time.time()
        if kind == "holdout":
            res = T.train_stage(data, c, run, device, epochs=epochs, deadline=time.monotonic() + 3 * 3600)
            shutil.move(str(run / "last_validation_model.pt"), str(folder / name))
            hist = json.loads((run / "training_history.json").read_text())
        else:
            res = T.train_stage(data, c, run, device, refit=True, epochs=epochs, deadline=time.monotonic() + 3 * 3600)
            shutil.move(str(run / "final_model.pt"), str(folder / name))
            hist = json.loads((run / "refit_history.json").read_text())
        if OVERRIDES:
            # The pipeline's load_model builds the model from the kit's config.json; a patched load_model applies these.
            ck = torch.load(folder / name, map_location="cpu", weights_only=True)
            ck["cfg_overrides"] = dict(OVERRIDES)
            torch.save(ck, folder / name)
        if plain is not None:
            torch.optim.AdamW = plain
        report[name] = dict(res, seconds=round(time.time() - t), seed=c["seed"], epochs=epochs, cosine=cosine, lr=c["lr"], history=hist)
        report_path.write_text(json.dumps(report, indent=2, default=str))
        shutil.rmtree(run, ignore_errors=True)
        log("DONE", tag, name, report[name]["seconds"], "s", res)


def holdout_models(split, epochs=16, cosine=False, lr=None):
    """(name, path) of the holdout models trained on this split for `epochs` epochs, kit/seed order."""
    suffix = tag_of(epochs, cosine, lr)
    def seeds(*dirs):
        return {int(m.group(1)): p for d in dirs for p in d.glob("casmi26_holdout_*.pt")
                if (m := re.fullmatch(rf"casmi26_holdout_(\d+){suffix}\.pt", p.name))}
    if split == "np":
        found = seeds(OUT / "holdout_np")
        return [(f"np{k}{suffix}", found[k]) for k in sorted(found)]
    found = seeds(N1_HOLDOUT, OUT / "holdout")
    kit = [("kit", KIT / "best_validation_model.pt")] if epochs == 16 else []
    return kit + [(f"{k}{suffix}", found[k]) for k in sorted(found)]


def diag(orders, split, epochs=16, cosine=False, lr=None):
    cfg, data, device = setup(split)
    cands = candidates(cfg, data, split)
    log("CANDIDATES", len(cands))
    cache_dir = OUT / "scores"
    cache_dir.mkdir(parents=True, exist_ok=True)
    names, per_model = [], []
    for name, path in holdout_models(split, epochs, cosine, lr):
        f = cache_dir / f"{split}_{name}_{path.stat().st_size}_{int(path.stat().st_mtime)}.npz"
        if f.is_file():
            z = np.load(f)
            s = {int(k[2:]): (z[k], z["f_" + k[2:]]) for k in z.files if k.startswith("c_")}
        else:
            s = model_scores(cfg, data, load(cfg, path, device), cands, device)
            np.savez_compressed(f, **{f"c_{i}": v[0] for i, v in s.items()}, **{f"f_{i}": v[1] for i, v in s.items()})
        names.append(name)
        per_model.append(s)
        log("SCORED", name, json.dumps(metrics([s[i][0] + s[i][1] for i in s])))
    keys = sorted(per_model[0])
    result = dict(models=names, single={n: ensemble([pm], keys) for n, pm in zip(names, per_model)}, curve={}, z={})
    rng = random.Random(0)
    for n in range(2, len(per_model) + 1):
        # Mean over random subsets of size n (the first subset is the models in listed order).
        subsets = [list(range(n))] + [rng.sample(range(len(per_model)), n) for _ in range(orders - 1)]
        if n == len(per_model):
            subsets = subsets[:1]
        rows = [ensemble([per_model[j] for j in s], keys) for s in subsets]
        result["curve"][n] = {k: float(np.mean([r[k] for r in rows])) for k in ("mrr25", "top1", "top5")}
        result["z"][n] = ensemble([per_model[j] for j in subsets[0]], keys, mode="z")
        log("ENSEMBLE", n, json.dumps(result["curve"][n]))
    tag = split + tag_of(epochs, cosine, lr)
    (OUT / f"diag_{tag}_report.json").write_text(json.dumps(result, indent=2))
    log("DIAG written", OUT / f"diag_{tag}_report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["holdout", "refit", "diag"])
    ap.add_argument("seeds", nargs="*", type=int)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay to 0 over the run (kit: constant LR)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (kit: 2e-4)")
    ap.add_argument("--split", choices=["kit", "np"], default="kit",
                    help="kit: the kit's random 10%% holdout; np: Enveda natural products held out (np_holdout_keys.txt)")
    ap.add_argument("--orders", type=int, default=5, help="random subsets per ensemble size in diag")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a kit config value (int/float), e.g. --set width=256 --set layers=4")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        kit = json.loads((KIT / "config.json").read_text())
        assert k in kit, f"unknown config key {k}"
        OVERRIDES[k] = type(kit[k])(float(v)) if isinstance(kit[k], int) else type(kit[k])(v)
    if a.mode == "diag":
        diag(a.orders, a.split, a.epochs, a.cosine, a.lr)
    else:
        assert a.seeds and 0 not in a.seeds, "seed 0 is the kit's own model"
        assert a.mode == "holdout" or a.split == "kit", "refits use all molecules"
        train(a.mode, a.seeds, a.epochs, a.split, a.cosine, a.lr)


if __name__ == "__main__":
    main()
