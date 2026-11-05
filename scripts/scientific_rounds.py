"""Two complete 32-epoch RTX 2070 holdout rounds, candidate InfoNCE and locked-leader fusion.

Requires the credited Saner Turhaner kit/cache already downloaded under outputs/local.
Training recipe is derived from that kit's train.py. The new objective removes unrelated
in-batch negatives, keeping 32 chemically selected decoys per query (MSAlign motivation).
No full-data refit may be evaluated on these holdouts. No submission is made by this script.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import local_seeds as L
from tail_fusion import baseline_order, paired_summary, partition, passes, reciprocal_rank, tail_order

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/local/science-20261105"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_loss(z, positive, negative, temperature):
    """Column zero is the target; other queries' positives never become negatives."""
    pos = (z * positive).sum(-1, keepdim=True)
    neg = torch.einsum("bd,bkd->bk", z, negative)
    return F.cross_entropy(torch.cat([pos, neg], 1) / temperature,
                           torch.zeros(len(z), device=z.device, dtype=torch.long))


def fit(data, cfg, device, seed, epochs):
    folder = OUT / f"round-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "holdout.pt"
    if checkpoint.exists():
        raise RuntimeError(f"Existing checkpoint {checkpoint}; use a fresh seed")
    L.T.seed_all(seed)
    rng = np.random.default_rng(seed)
    pools = L.T.hard_pools(data.mol, data.train, cfg["negative_pool"])
    candidates = L.T.candidate_sets(data, cfg)
    counts = np.asarray(data.fp[data.train]).sum(0, dtype=np.float64)
    posweight = np.clip((len(data.train) - counts) / (counts + 1), 1, 20)
    weights = torch.as_tensor(posweight, device=device, dtype=torch.float32)
    np.save(folder / "log_pos_weight.npy", np.log(posweight))
    model = L.SpectrumCandidateModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        model.train()
        losses = []
        for step in range(cfg["steps_per_epoch"]):
            ids = rng.choice(data.train, cfg["batch_size"], replace=False)
            rows = np.array([rng.choice(data.by_mol[int(i)]) for i in ids])
            # Unique negatives, all from training; the held-out identities never enter this loss.
            neg = np.stack([rng.choice(pools[int(i)], 32, replace=False) for i in ids])
            fp = torch.tensor(data.fp[ids], device=device, dtype=torch.float32)
            nf = torch.tensor(data.fp[neg.reshape(-1)], device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                z, pred = model.encode_spectra(*data.batch(rows, device, augment=True))
                mol = model.encode_molecules(fp)
                nmol = model.encode_molecules(nf).reshape(len(ids), 32, -1)
                retrieval = candidate_loss(z, mol, nmol, cfg["temperature"])
                auxiliary = F.binary_cross_entropy_with_logits(pred, fp, pos_weight=weights)
                loss = retrieval + cfg["aux_weight"] * auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        score, frame = L.T.validate(model, data, candidates, device, cfg)
        row = dict(epoch=epoch, steps=len(losses), loss=float(np.mean(losses)),
                   retrieval_mrr25=score, top1=float(frame.top1.mean()),
                   seconds=round(time.monotonic() - start, 2))
        history.append(row)
        L.T.atomic_save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                             stage="holdout", epoch=epoch, seed=seed,
                             objective="candidate-only-32", cfg=cfg), checkpoint)
        (folder / "history.json").write_text(json.dumps(history, indent=2))
        frame.to_csv(folder / "retrieval.csv", index=False)
        print("EPOCH", seed, json.dumps(row), flush=True)
    assert len(history) == epochs and all(h["steps"] == 500 for h in history)
    del opt, scaler
    return model.eval(), folder


@torch.inference_mode()
def panel_terms(model, cfg, data, device):
    panel = pd.read_parquet(ROOT / "outputs/local/saner_panel/labelled_candidate_scores.parquet")
    fps = np.load(ROOT / "outputs/local/saner_panel/panel_fps.npz", allow_pickle=False)
    smiles = panel.smiles.astype(str).unique()
    assert len(smiles) == len(fps["fp"])
    assert np.array_equal(panel.drop_duplicates("smiles").metric_key.astype(str), fps["keys"])
    fpof = dict(zip(smiles, fps["fp"]))
    idxof = {key: i for i, key in enumerate(data.mol.key)}
    terms = []
    for mid, g in panel.groupby("molecule_id", sort=True):
        g = g.sort_values("baseline_rank", kind="stable")
        i = idxof[mid.replace("holdout_", "")]
        assert i in set(data.val), "Panel molecule was used for training"
        rows = data.by_mol[i]
        rows = rows[data.spec.source.values[rows] == 0]
        if not len(rows):
            raise RuntimeError(f"Missing competition spectra {mid}")
        fp = torch.tensor(np.stack([fpof[s] for s in g.smiles]), device=device, dtype=torch.float32)
        mol = model.encode_molecules(fp)
        cos, fpt = [], []
        for st in range(0, len(rows), 64):
            z, pred = model.encode_spectra(*data.batch(rows[st:st + 64], device))
            cos.append((z @ mol.T).float().cpu().numpy())
            fpt.append((pred @ fp.T / cfg["fp_bits"]).float().cpu().numpy())
        terms.append(dict(key=mid, labels=g.label.to_numpy(), public=g.neural_score.to_numpy(),
                          cos=np.concatenate(cos).mean(0), fp=np.concatenate(fpt).mean(0),
                          fingerprints=fp.cpu().numpy()))
    assert len(terms) == 120
    return terms


def evaluate(terms, cfg, folder, logweight, frozen=None):
    # Weighted BCE yields logits logit(p) + log(pos_weight); subtracting that shift is
    # the Bernoulli calibration ablation. Candidate-independent likelihood terms cancel.
    grid = [dict(alpha=a, fp_weight=w, correction=c)
            for a in [0., .1, .25, .5] for w in [0., 2.] for c in [0., 1.]]
    def score(t, config):
        corrected_fp = t["fp"] - config["correction"] * (t["fingerprints"] @ logweight) / cfg["fp_bits"]
        ours = t["cos"] / cfg["temperature"] + config["fp_weight"] * corrected_fp
        order = tail_order(t["public"], ours, config["alpha"])
        assert order[0] == baseline_order(t["public"])[0]
        return reciprocal_rank(t["labels"], order)
    for t in terms:
        t["before"] = reciprocal_rank(t["labels"], baseline_order(t["public"]))
    dev = [t for t in terms if partition(t["key"]) == 0]
    confirm = [t for t in terms if partition(t["key"]) == 1]
    sweep = [dict(config=c, mrr25=float(np.mean([score(t, c) for t in dev]))) for c in grid]
    # Predeclared tie-break: smallest alpha, no prior correction, original fingerprint weight.
    best = max(sweep, key=lambda r: (r["mrr25"], -r["config"]["alpha"],
                                    -r["config"]["correction"], r["config"]["fp_weight"]))["config"]
    selected = best if frozen is None else frozen
    report = dict(selected=selected, selected_on="round-1 development only" if frozen else "development only",
                  sweep=sweep, baseline="s3 consensus proxy: panel neural, 0.25/0.75; fragment merge not replayed",
                  full_pipeline_evaluation=False, hidden_score_comparable=False)
    for name, group in [("development", dev), ("confirmation", confirm), ("all", terms)]:
        report[name] = paired_summary([t["before"] for t in group], [score(t, selected) for t in group])
    report["local_gate_passed"] = passes(report["confirmation"]) and selected["alpha"] > 0
    # Keep enough per-molecule evidence to audit both gains and losses.
    report["molecules"] = [dict(key=t["key"], partition=partition(t["key"]), before=t["before"],
                                after=score(t, selected)) for t in terms]
    (folder / "panel_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[202611051, 202611052])
    ap.add_argument("--epochs", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, data, device = L.setup("kit")
    assert cfg["steps_per_epoch"] == 500
    print("DEVICE", torch.cuda.get_device_name(), "threads", torch.get_num_threads(), flush=True)
    manifest = dict(seeds=a.seeds, epochs=a.epochs, steps_per_epoch=500, negatives=32,
                    kit_sha256={n: sha(L.KIT / n) for n in ["train.py", "model.py", "config.json"]},
                    public_best=0.477, objective="candidate-only InfoNCE + unchanged auxiliary BCE",
                    gate="Both seeded confirmation delta and paired bootstrap lower 95% bound >0; no leader moves",
                    submission="Requires a full pipeline smoke and comparable holdout evidence; this script cannot submit")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results, frozen = [], None
    for index, seed in enumerate(a.seeds):
        model, folder = fit(data, cfg, device, seed, a.epochs)
        terms = panel_terms(model, cfg, data, device)
        report = evaluate(terms, cfg, folder, np.load(folder / "log_pos_weight.npy"), frozen)
        if frozen is None:
            frozen = report["selected"]
        results.append(report)
        print("ROUND_RESULT", seed, json.dumps({k:v for k,v in report.items() if k not in ["sweep", "molecules"]}), flush=True)
        print(f"[0s] DONE round-{index + 1} seed={seed} epochs={a.epochs}", flush=True)
        del model, terms
        torch.cuda.empty_cache()
    result = dict(rounds=[dict(seed=s, confirmation=r["confirmation"], selected=r["selected"],
                              local_gate_passed=r["local_gate_passed"]) for s, r in zip(a.seeds, results)],
                  both_local_gates_passed=all(r["local_gate_passed"] for r in results),
                  submit_allowed=False,
                  reason="Full frozen-pipeline holdout is unavailable locally; proxy gains cannot prove hidden score >0.477")
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
