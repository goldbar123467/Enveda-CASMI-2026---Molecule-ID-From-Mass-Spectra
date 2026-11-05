"""Full-data refits of Saner Turhaner's scorer (kit sanpier/casmi-spectrum-model-v1, unchanged train.py) on Kaggle
TPU v3-8 with PyTorch/XLA. Written to /kaggle/working by the n4/x7 notebooks and run as a subprocess (xmp.spawn needs
an importable worker, which a notebook's __main__ is not).

    python tpu_train.py --mode single --k0 301 --epochs 2 --steps 100          # one TPU core, timing probe
    python tpu_train.py --mode spawn  --k0 301 --epochs 64 --save             # one seed per TPU core, k0 + ordinal

Seed k uses config seed 20261023 + 1000*k, like scripts/local_seeds.py. With --save the refit goes to
/kaggle/working/seeds/casmi26_seed_<k>_e<epochs>.pt. Each replica writes reports/replica_<k>.json.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

BASE_SEED = 20261023
W = Path("/kaggle/working")
INP = Path("/kaggle/input")


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "spawn"], default="single")
    ap.add_argument("--k0", type=int, default=301)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--hours", type=float, default=8.0, help="per-replica training deadline")
    return ap.parse_args()


A = args()


def worker(index):
    t0 = time.time()
    k = A.k0 + index
    rep = dict(k=k, index=index, pid=os.getpid())
    (W / "reports").mkdir(exist_ok=True)
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        rep["device"] = str(dev)
        kit = next(p.parent for p in INP.rglob("train.py") if (p.parent / "model.py").is_file())
        cache = next(p.parent for p in INP.rglob("peaks.bin") if (p.parent / "fingerprints.npy").is_file())
        sys.path.insert(0, str(kit))
        import train as T

        class AdamW(torch.optim.AdamW):
            # Cut the lazy XLA graph once per optimizer step.
            def step(self, *a, **kw):
                out = super().step(*a, **kw)
                xm.mark_step()
                return out

        torch.optim.AdamW = AdamW
        cfg = json.loads((kit / "config.json").read_text())
        cfg.update(work=str(cache), steps_per_epoch=A.steps)
        data = T.Data(cfg)
        data.x = np.array(data.x)
        data.meta = np.array(data.meta)
        data.fp = np.array(data.fp)
        rep["load_seconds"] = round(time.time() - t0, 1)
        out = W / "runs" / f"refit_{k}"
        out.mkdir(parents=True, exist_ok=True)
        c = dict(cfg, seed=BASE_SEED + 1000 * k)
        t = time.time()
        res = T.train_stage(data, c, out, dev, refit=True, epochs=A.epochs,
                            deadline=time.monotonic() + A.hours * 3600)
        rep.update(res, train_seconds=round(time.time() - t, 1),
                   history=json.loads((out / "refit_history.json").read_text()))
        if A.save:
            (W / "seeds").mkdir(exist_ok=True)
            shutil.move(str(out / "final_model.pt"), str(W / "seeds" / f"casmi26_seed_{k}_e{A.epochs}.pt"))
        shutil.rmtree(out, ignore_errors=True)
    except Exception:
        rep["error"] = traceback.format_exc()[-4000:]
    rep["seconds"] = round(time.time() - t0, 1)
    (W / "reports" / f"replica_{k}.json").write_text(json.dumps(rep, indent=2, default=str))
    print("REPLICA", json.dumps({k: v for k, v in rep.items() if k != "history"}, default=str), flush=True)


def _mp_fn(index):
    # Module level so spawned children can import it.
    import torch_xla.runtime as xr
    worker(xr.global_ordinal())


if __name__ == "__main__":
    if A.mode == "single":
        worker(0)
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=())
