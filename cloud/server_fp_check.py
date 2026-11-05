"""Guarded cloud-only equivalence/timing check for batched pretrained FP inference."""
import argparse
import json
import time
from pathlib import Path
import workload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    proof = workload.require_server_cloud(args.authorization, "gpu")
    root = Path(proof["project_root"]).resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(root/"outputs"):
        raise ValueError("Output must stay in the authorized project")
    workload.INPUT_ROOT = root/"input"
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frame = pd.read_parquet(workload.find_input("test.parquet"))
    groups = list(frame.groupby("molecule_id", sort=False).groups.values())[:24]
    spectra = []
    for rows in groups:
        rows = np.asarray(rows)
        mode = workload.ion_mode(frame.adduct.iloc[rows].tolist(), frame.ionization_mode.iloc[rows].tolist())
        spectra.append([{"mz": np.asarray(frame.ms2_mzs.iloc[r], np.float64),
                         "it": np.asarray(frame.ms2_normalized_intensities.iloc[r], np.float64),
                         "prec": float(frame.precursor_mz.iloc[r]), "adduct": frame.adduct.iloc[r],
                         "instrument": frame.instrument_type.iloc[r] or "timsTOF",
                         "ce": workload.collision_energy(frame.collision_energy_ev.iloc[r]), "mode": float(m)}
                        for r, m in zip(rows, mode) if m != 0 and frame.ms2_mzs.iloc[r] is not None])
    models = workload.load_fp_models(len(np.load(workload.find_input("fp_bits.npy"))), "gpu")
    # First call absorbs CUDA initialization; compare warmed paths on identical real inputs.
    workload.molecule_logits(models, spectra[0])
    torch.cuda.synchronize()
    timings, predictions = {}, {}
    for batch in (1, 16, 32):
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        predictions[batch] = workload.molecules_logits(models, spectra, batch)
        torch.cuda.synchronize()
        timings[str(batch)] = {"seconds": time.monotonic()-started,
                               "peak_vram_bytes": torch.cuda.max_memory_allocated()}
    reference = predictions[1]
    differences = {}
    for batch in (16, 32):
        difference = max(float(np.max(np.abs(a-b))) for a,b in zip(reference,predictions[batch]) if a is not None)
        differences[str(batch)] = difference
        if any((a is None)!=(b is None) or (a is not None and not np.allclose(a,b,rtol=2e-4,atol=2e-4))
               for a,b in zip(reference,predictions[batch])):
            raise RuntimeError("Batched inference failed the float32 equivalence gate")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"molecules": len(spectra), "timings": timings, "max_abs_logit_difference": differences,
              "equivalence": "passed", "dtype": "float32", "torch": torch.__version__}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
