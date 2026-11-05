"""Compute entry point bound to the user's SSH-verified Vast instance."""
import argparse
import json
import os
from pathlib import Path

import workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fp-batch-size", type=int, default=32)
    args = parser.parse_args()
    workload.require_server_cloud(args.authorization, args.device)
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        raise ValueError("Workers must fit the instance CPU allocation")
    if args.fp_batch_size < 1:
        raise ValueError("Fingerprint batch size must be positive")
    for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = str(args.workers)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if args.config:
        workload.CONFIG.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    workload.CONFIG["fp_workers"] = args.workers
    workload.CONFIG["fp_batch_size"] = args.fp_batch_size
    workload.CONFIG["process_context"] = "spawn"
    workload.CONFIG["auto_query_workers"] = True
    workload.CONFIG["optimize_frag_match"] = True
    workload.CONFIG["ranker_params"] = {**workload.CONFIG["ranker_params"], "n_jobs": args.workers}
    import torch
    import pyarrow
    torch.set_num_threads(4 if args.device == "gpu" else args.workers)
    torch.set_num_interop_threads(1)
    pyarrow.set_cpu_count(args.workers)
    workload.main(device=args.device, server_authorization=args.authorization, output_directory=args.output)


if __name__ == "__main__":
    main()
