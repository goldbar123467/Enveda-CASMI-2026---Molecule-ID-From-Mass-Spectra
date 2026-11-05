"""Cloud workload embedded into both Kaggle notebooks at prepare time.

Retrieval over a candidate pool (training structures + COCONUT):

  1. Library channel: spectral-entropy similarity between each test spectrum and every
     training spectrum whose neutral mass lies within PPM of the molecule's neutral mass.
  2. Analog channel: hybrid (direct + mass-shifted) entropy similarity against one
     representative spectrum per training structure and ion mode within +-ANALOG_WINDOW Da.
  3. Fingerprint channel (v2): public spectrum->fingerprint transformers predict bit logits z;
     each candidate's f.z is its log-likelihood up to a constant.
  4. Candidates in the PPM window score
        A * library * [library >= tau] + max_a sim_a^P * Tanimoto(c, a) + B * z-scored f.z

A, tau, P and B are chosen in the cloud on two held-out simulations built from
enveda-np-examples: Class 1 removes only the query spectra from the library; Class 2
removes every spectrum of the query structures.

This file must never run on the local host. main() calls the Kaggle-only guard before
importing numerical libraries, reading competition data, or allocating compute.
"""

import json
import os
import platform
import re
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")

CONFIG = {
    "version": "v6-pubchem-popularity-double",
    # v2: public spectrum->fingerprint transformers (prvsiyan/casmi26-fp-models-v2) score f.z.
    "use_fp_model": True,
    "grid_fz_weight": [0.0, 0.05, 0.1, 0.2, 0.4],
    # v3: LightGBM lambdarank over channel features, trained on simulated queries in the cloud.
    "use_ranker": True,
    "seed": 2026,
    "train_queries_class2": 900,
    "train_queries_class1": 450,
    "train_query_spectra": 6,
    "ranker_seeds": [0, 1, 2],
    "ranker_params": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 40,
                      "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    # v4: MetFrag-lite fragmentation explain score as ranker features, plus a gated library
    # boost and f.z on top of the ranker score.
    "use_frag": True,
    "grid_ranker_lib_weight": [0.0, 1.0, 2.0, 4.0, 8.0],
    "grid_ranker_lib_gate": [0.7, 0.8, 0.9],
    "grid_ranker_fz_weight": [0.0, 0.1, 0.2, 0.4, 0.8],
    # v5: PubChem popularity prior (dmitriigluzdov/casmi26-pubchem-popularity-prior), weight set by LB.
    "pop_weight": 0.30,
    # Phase 1 E3: enable only for the ChEBI/LIPID MAPS experiment.
    "use_bio_db": False,
    "ppm": 10.0,
    "ppm_fallback": 30.0,
    "mz_tol": 0.01,
    "int_floor": 0.002,
    "max_peaks": 256,
    "above_precursor": 2.0,
    "analog_window": 200.0,
    "n_analog": 200,
    "analog_prefilter": 4000,  # exact hybrid similarity only for the top rows by a cheap bound
    "grid_lib_weight": [0.0, 1.0, 2.0, 4.0],
    "grid_lib_gate": [0.5, 0.6, 0.7, 0.8, 0.9],
    "grid_sim_power": [2.0, 3.0],
    # v1/v2 public LB fit LB ~ 0.19*C1 + 0.22*C2, so Class 1 gets ~0.45 of the objective.
    "objective_class1_weight": 0.45,
    "max_query_spectra": 16,
    "top_n": 25,
    "train_batch_rows": 200_000,
    "fp_workers": 4,
    # Operational pilots preserve exact evidence/scores and fall back if slower or different.
    "auto_query_workers": True,
    "optimize_frag_match": True,
}

ELECTRON = 0.00054857990946
PROTON = 1.007276466621
MONO = {
    "C": 12.0, "H": 1.00782503207, "D": 2.01410177812, "N": 14.0030740048,
    "O": 15.99491461956, "P": 30.97376163, "S": 31.97207100, "F": 18.99840322,
    "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473, "Si": 27.9769265325,
    "B": 11.0093054, "Se": 79.9165213, "Na": 22.9897692809, "K": 38.96370668,
    "As": 74.9215965, "Fe": 55.9349375,
}
H2O = 2 * MONO["H"] + MONO["O"]
# adduct -> (molecules n, charge z, ion mass shift d); m/z = (n * M + d) / z
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, MONO["N"] + 4 * MONO["H"] - ELECTRON),
    "[M+Na]+": (1, 1, MONO["Na"] - ELECTRON),
    "[M+K]+": (1, 1, MONO["K"] - ELECTRON),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M]+": (1, 1, -ELECTRON),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[M-H]-": (1, 1, -PROTON),
    "[M-H2O-H]-": (1, 1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, 1, MONO["C"] + 2 * MONO["H"] + 2 * MONO["O"] - PROTON),
    "[M+C2H4O2-H]-": (1, 1, 2 * MONO["C"] + 4 * MONO["H"] + 2 * MONO["O"] - PROTON),
    "[M+Cl]-": (1, 1, MONO["Cl"] + ELECTRON),
    "[M]-": (1, 1, ELECTRON),
    "[M-2H]2-": (1, 2, -2 * PROTON),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, MONO["Na"] - ELECTRON),
    "[2M-H]-": (2, 1, -PROTON),
}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_FP = {}
# Input vocabularies of the published fingerprint models; order must match their training.
FP_ADDUCTS = [
    "[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
    "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
    "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>",
]
FP_MAX_PEAKS = 128


def require_kaggle_cloud() -> None:
    if not (
        sys.platform == "linux"
        and Path("/kaggle/input").is_dir()
        and Path("/kaggle/working").is_dir()
        and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    ):
        raise RuntimeError(
            "Local workload execution is prohibited. Use the Kaggle cloud CPU/GPU notebook."
        )


def server_identity():
    """Non-secret identity of the explicitly authorized Vast container."""
    import hashlib
    import socket

    machine = Path("/etc/machine-id")
    return {
        "hostname": socket.gethostname(),
        "container_id": os.environ.get("CONTAINER_ID"),
        "machine_id_sha256": hashlib.sha256(machine.read_bytes().strip()).hexdigest() if machine.is_file() else None,
        "vast_guide_present": Path("/etc/vast-agents-guide.md").is_file(),
    }


def require_server_cloud(authorization, device="cpu"):
    # Windows and WSL on the user's PC must refuse before any numerical imports or data access.
    if sys.platform != "linux" or "microsoft" in platform.release().lower():
        raise RuntimeError("Local workload execution is prohibited. Use the authorized cloud server.")
    proof = json.loads(Path(authorization).read_text(encoding="utf-8"))
    actual = server_identity()
    fields = ("hostname", "container_id", "machine_id_sha256")
    if (proof.get("runtime") != "vast-ssh" or not actual["vast_guide_present"]
            or any(not proof.get(k) or proof[k] != actual[k] for k in fields)):
        raise RuntimeError("This machine is not the SSH-verified, authorized Vast instance.")
    allowed = proof.get("allowed_devices", [proof.get("allowed_device")])
    if device not in {"cpu", "gpu"} or device not in allowed:
        raise RuntimeError("The requested device is not authorized for this server entry point.")
    from pathlib import PurePosixPath
    if not PurePosixPath(proof["project_root"]).is_absolute():
        raise RuntimeError("The authorized project root must be absolute.")
    return proof


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def find_input(pattern):
    import glob

    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/{pattern}", recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f"No attached input provides {pattern}.")
    return hits[0]


def install_rdkit():
    try:
        import rdkit

        return rdkit.__version__
    except ImportError:
        pass
    import glob
    import importlib
    import subprocess

    tag = f"-cp{sys.version_info.major}{sys.version_info.minor}-"
    wheels = sorted(w for w in glob.glob(f"{INPUT_ROOT}/**/rdkit-*.whl", recursive=True) if tag in w)
    if not wheels:
        raise RuntimeError(f"No offline RDKit wheel matches {tag}; attach metric/rdkit-2026-3-3-wheel.")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", wheels[0]],
        check=True,
    )
    importlib.invalidate_caches()
    import rdkit

    return rdkit.__version__


def formula_mass(formula):
    if not isinstance(formula, str) or not formula:
        return float("nan")
    total = 0.0
    for element, count in _FORMULA_TOKEN.findall(formula):
        if element not in MONO:
            return float("nan")
        total += MONO[element] * (int(count) if count else 1)
    return total


def neutral_mass(precursor_mz, adducts):
    import numpy as np

    out = np.full(len(precursor_mz), np.nan)
    adducts = np.asarray(adducts, dtype=object)
    for name, (n, z, d) in ADDUCTS.items():
        hit = adducts == name
        if hit.any():
            out[hit] = (precursor_mz[hit] * z - d) / n
    return out


def ion_mode(adducts, modes):
    """+1/-1 from the adduct's trailing charge sign, else from ionization_mode."""
    import numpy as np

    out = np.zeros(len(adducts), np.int8)
    for i, (a, m) in enumerate(zip(adducts, modes)):
        if isinstance(a, str) and a.endswith("+"):
            out[i] = 1
        elif isinstance(a, str) and a.endswith("-"):
            out[i] = -1
        elif isinstance(m, str) and m.lower().startswith("p"):
            out[i] = 1
        elif isinstance(m, str) and m.lower().startswith("n"):
            out[i] = -1
    return out


def make_kernels():
    import numpy as np
    from numba import njit, prange

    ln4 = float(np.log(4.0))

    @njit(parallel=True)
    def count_keep(ms, ml, mzv, its, il, itv, prec, floor, topk, above):
        out = np.zeros(len(ms), np.int64)
        for s in prange(len(ms)):
            n = min(ml[s], il[s])
            lim = prec[s] + above if prec[s] == prec[s] else 1e12
            mx = 0.0
            for t in range(n):
                m = mzv[ms[s] + t]
                v = itv[its[s] + t]
                if m > 0.0 and m <= lim and v > mx:
                    mx = v
            if mx <= 0.0:
                continue
            thr = floor * mx
            c = 0
            for t in range(n):
                m = mzv[ms[s] + t]
                v = itv[its[s] + t]
                if m > 0.0 and m <= lim and v >= thr and v > 0.0:
                    c += 1
            out[s] = min(c, topk)
        return out

    @njit(parallel=True)
    def fill_clean(ms, ml, mzv, its, il, itv, prec, floor, topk, above, off, out_mz, out_it):
        """Floor, top-k, sort by m/z, normalise to sum 1, entropy-weight low-entropy spectra."""
        for s in prange(len(ms)):
            k = off[s + 1] - off[s]
            if k == 0:
                continue
            n = min(ml[s], il[s])
            lim = prec[s] + above if prec[s] == prec[s] else 1e12
            mx = 0.0
            for t in range(n):
                m = mzv[ms[s] + t]
                v = itv[its[s] + t]
                if m > 0.0 and m <= lim and v > mx:
                    mx = v
            thr = floor * mx
            cm = np.empty(n, np.float64)
            cv = np.empty(n, np.float64)
            c = 0
            for t in range(n):
                m = mzv[ms[s] + t]
                v = itv[its[s] + t]
                if m > 0.0 and m <= lim and v >= thr and v > 0.0:
                    cm[c] = m
                    cv[c] = v
                    c += 1
            sel_m = cm[:c].copy()
            sel_v = cv[:c].copy()
            if c > k:
                keep = np.argsort(sel_v)[c - k:]
                sel_m = sel_m[keep]
                sel_v = sel_v[keep]
            o = np.argsort(sel_m)
            peak_m = sel_m[o]
            peak_v = sel_v[o] / sel_v.sum()
            ent = 0.0
            for t in range(k):
                ent -= peak_v[t] * np.log(peak_v[t])
            if ent < 3.0:
                weighted = peak_v ** (0.25 + 0.25 * ent)
                peak_v = weighted / weighted.sum()
            base = off[s]
            for t in range(k):
                out_mz[base + t] = peak_m[t]
                out_it[base + t] = peak_v[t]

    @njit(fastmath=True)
    def pair_gain(a, b):
        # Entropy similarity is the sum of this term over matched peak pairs.
        if a <= 0.0 or b <= 0.0:
            return 0.0
        s = a + b
        return (s * np.log(s) - a * np.log(a) - b * np.log(b)) / ln4

    @njit(fastmath=True)
    def sim_direct(qm, qp, cm, cp, tol):
        i = 0
        j = 0
        total = 0.0
        while i < len(qm) and j < len(cm):
            d = qm[i] - cm[j]
            if d < -tol:
                i += 1
            elif d > tol:
                j += 1
            else:
                total += pair_gain(float(qp[i]), float(cp[j]))
                i += 1
                j += 1
        return total

    @njit(fastmath=True)
    def sim_hybrid(qm, qp, cm, cp, tol, shift):
        """Each peak pairs once, directly or shifted by the precursor mass difference."""
        n = len(qm)
        m = len(cm)
        cap = 4 * (n + m) + 16
        gain = np.empty(cap, np.float64)
        pi = np.empty(cap, np.int64)
        pj = np.empty(cap, np.int64)
        k = 0
        for offset in (0.0, shift):
            j0 = 0
            for i in range(n):
                while j0 < m and cm[j0] + offset < qm[i] - tol:
                    j0 += 1
                j = j0
                while j < m and cm[j] + offset <= qm[i] + tol and k < cap:
                    gain[k] = pair_gain(float(qp[i]), float(cp[j]))
                    pi[k] = i
                    pj[k] = j
                    k += 1
                    j += 1
        if k == 0:
            return 0.0
        used_q = np.zeros(n, np.bool_)
        used_c = np.zeros(m, np.bool_)
        total = 0.0
        for t in np.argsort(-gain[:k]):
            if used_q[pi[t]] or used_c[pj[t]]:
                continue
            used_q[pi[t]] = True
            used_c[pj[t]] = True
            total += gain[t]
        return total

    @njit(fastmath=True)
    def sim_shifted(qm, qp, cm, cp, tol, shift):
        i = 0
        j = 0
        total = 0.0
        while i < len(qm) and j < len(cm):
            d = qm[i] - (cm[j] + shift)
            if d < -tol:
                i += 1
            elif d > tol:
                j += 1
            else:
                total += pair_gain(float(qp[i]), float(cp[j]))
                i += 1
                j += 1
        return total

    @njit(parallel=True, nogil=True)
    def bound(qm, qp, rows, off, mz, it, tol, shifts):
        """Allocation-free direct + shifted sum; ranks rows before the exact hybrid pass."""
        out = np.zeros(len(rows), np.float32)
        for k in prange(len(rows)):
            a = off[rows[k]]
            b = off[rows[k] + 1]
            if b <= a:
                continue
            out[k] = sim_direct(qm, qp, mz[a:b], it[a:b], tol)
            if abs(shifts[k]) > tol:
                out[k] += sim_shifted(qm, qp, mz[a:b], it[a:b], tol, shifts[k])
        return out

    @njit(parallel=True, nogil=True)
    def search(qm, qp, rows, off, mz, it, tol, shifts, hybrid):
        out = np.zeros(len(rows), np.float32)
        for k in prange(len(rows)):
            a = off[rows[k]]
            b = off[rows[k] + 1]
            if b <= a:
                continue
            if hybrid and abs(shifts[k]) > tol:
                out[k] = sim_hybrid(qm, qp, mz[a:b], it[a:b], tol, shifts[k])
            else:
                out[k] = sim_direct(qm, qp, mz[a:b], it[a:b], tol)
        return out

    return {"count_keep": count_keep, "fill_clean": fill_clean, "search": search, "bound": bound}


def clean_parts(kernels, ms, ml, mzv, its, il, itv, prec, cfg):
    import numpy as np

    args = (
        ms.astype(np.int64), ml.astype(np.int64), mzv.astype(np.float64),
        its.astype(np.int64), il.astype(np.int64), itv.astype(np.float64),
        prec.astype(np.float64), cfg["int_floor"], cfg["max_peaks"], cfg["above_precursor"],
    )
    counts = kernels["count_keep"](*args)
    off = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=off[1:])
    out_mz = np.empty(off[-1], np.float32)
    out_it = np.empty(off[-1], np.float32)
    kernels["fill_clean"](*args, off, out_mz, out_it)
    return counts, out_mz, out_it


def list_parts(array):
    """Start, length, and flat values of an Arrow list column (null lists become empty)."""
    import numpy as np

    offsets = array.offsets.to_numpy().astype(np.int64)
    lengths = np.diff(offsets)
    if array.null_count:
        lengths[array.is_null().to_numpy(zero_copy_only=False)] = 0
    values = array.values.to_numpy(zero_copy_only=False).astype(np.float64)
    return offsets[:-1], lengths, values


def clean_lists(kernels, mz_lists, it_lists, prec, cfg):
    import numpy as np

    mz_lists = [x if x is not None else [] for x in mz_lists]
    it_lists = [x if x is not None else [] for x in it_lists]
    ml = np.array([len(x) for x in mz_lists], np.int64)
    il = np.array([len(x) for x in it_lists], np.int64)
    ms = np.concatenate([[0], np.cumsum(ml)[:-1]]).astype(np.int64)
    its = np.concatenate([[0], np.cumsum(il)[:-1]]).astype(np.int64)
    mzv = np.concatenate([np.asarray(x, np.float64) for x in mz_lists]) if ml.sum() else np.zeros(0)
    itv = np.concatenate([np.asarray(x, np.float64) for x in it_lists]) if il.sum() else np.zeros(0)
    counts, out_mz, out_it = clean_parts(kernels, ms, ml, mzv, its, il, itv, prec, cfg)
    off = np.concatenate([[0], np.cumsum(counts)])
    return [(out_mz[off[i]:off[i + 1]], out_it[off[i]:off[i + 1]]) for i in range(len(counts))]


def _init_fp_worker(bits):
    _FP["bits"] = bits


def _fp_worker(smiles):
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt

    if "m2" not in _FP:
        from rdkit.Chem import rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.*")
        _FP["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        _FP["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        _FP["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        bits = np.concatenate([
            _FP["m2"].GetFingerprintAsNumPy(mol).astype(np.uint8),
            _FP["m3"].GetFingerprintAsNumPy(mol).astype(np.uint8),
            _FP["rk"].GetFingerprintAsNumPy(mol).astype(np.uint8),
            np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8),
        ])[_FP["bits"]]
        return np.packbits(bits), float(ExactMolWt(mol))
    except Exception:
        return None


class Pool:
    """Candidate structures with bit-packed fingerprints, sorted by neutral mass."""

    def __init__(self, fp, mass, keys, smiles, nbits):
        import numpy as np

        order = np.argsort(mass, kind="mergesort")
        self.fp = fp[order]
        self.mass = mass[order]
        self.keys = np.asarray(keys, dtype=object)[order]
        self.smiles = np.asarray(smiles, dtype=object)[order]
        self.nbits = nbits
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.pop = np.zeros(len(self.keys), np.float32)

    def load_popularity(self):
        """log1p(PubChem substances) + log1p(PubMed articles) per structure, 0 when unknown."""
        import numpy as np
        import pandas as pd

        table = pd.read_csv(find_input("pool_popularity.csv"), usecols=["inchikey14", "pop"])
        lookup = dict(zip(table.inchikey14, table["pop"].astype(np.float32)))
        self.pop = np.array([lookup.get(k, 0.0) for k in self.keys], np.float32)
        self.pop[~np.isfinite(self.pop)] = 0.0
        log(f"popularity known for {(self.pop > 0).mean():.1%} of pool structures")

    def window(self, mass, ppm):
        import numpy as np

        lo = np.searchsorted(self.mass, mass * (1 - ppm / 1e6), "left")
        hi = np.searchsorted(self.mass, mass * (1 + ppm / 1e6), "right")
        return np.arange(lo, hi)

    def bits(self, idx):
        import numpy as np

        return np.unpackbits(self.fp[idx], axis=1)[:, : self.nbits].astype(np.float32)


def instrument_family(name):
    if name is None:
        return 4
    text = str(name).lower()
    if "timstof" in text:
        return 0
    if any(s in text for s in ("orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive")):
        return 1
    if "tof" in text:
        return 2
    if "trap" in text or "qq" in text:
        return 3
    return 4


def fp_prep_peaks(mz, it, prec, max_peaks=FP_MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):
    """The fingerprint models' input transform: window-diversified top-N peaks, sqrt intensity."""
    import numpy as np

    empty = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    keep = mz <= prec + 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0 or it.max() <= 0:
        return empty
    keep = it >= floor * it.max()
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        counts, sel = {}, []
        for i in order:
            c = counts.get(bucket[i], 0)
            if c < per_win:
                counts[bucket[i]] = c + 1
                sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            chosen = set(sel.tolist())
            rest = np.array([i for i in order if i not in chosen])
            if len(rest):
                sel = np.concatenate([sel, rest[: max_peaks - len(sel)]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(np.float32), np.sqrt(it / it.max()).astype(np.float32)


def merge_peaks(spectra):
    """All of a molecule's peaks as one list; near-duplicate m/z keep the stronger peak."""
    import numpy as np

    mz = np.concatenate([np.asarray(s["mz"], np.float64) for s in spectra])
    it = np.concatenate([np.asarray(s["it"], np.float64) / max(float(np.max(s["it"])), 1e-9)
                         if len(s["it"]) else np.zeros(0) for s in spectra])
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]:
                keep[j - 1] = False
            else:
                keep[j] = False
    return mz[keep], it[keep]


def make_fp_net():
    """FPNet as published with prvsiyan/casmi26-fp-models-v2; names must match the checkpoints."""
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SinEmb(nn.Module):
        def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
            super().__init__()
            wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, dim // 2), power) + lo)
            self.register_buffer("inv", (2 * math.pi) / wav)

        def forward(self, x):
            a = x.unsqueeze(-1) * self.inv
            return torch.cat([torch.sin(a), torch.cos(a)], -1)

    class Block(nn.Module):
        def __init__(self, d, h, drop):
            super().__init__()
            self.h = h
            self.n1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.o = nn.Linear(d, d)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, pad):
            B, N, D = x.shape
            q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=(~pad)[:, None, None, :])
            x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
            return x + self.drop(self.ff(self.n2(x)))

    class FPNet(nn.Module):
        def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
            super().__init__()
            self.mz_emb = SinEmb(d)
            self.nl_emb = SinEmb(d)
            self.pk = nn.Linear(2 * d + 1, d)
            self.prec_emb = SinEmb(d)
            self.ad = nn.Embedding(len(FP_ADDUCTS), d)
            self.ins = nn.Embedding(5, d)
            self.gl = nn.Linear(d + 3, d)
            self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

        def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
            nl = (prec[:, None] - mz).clamp(min=0)
            p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
            g = self.gl(torch.cat([self.prec_emb(prec), (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                                   torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
            x = torch.cat([g.unsqueeze(1), p], 1)
            pad = torch.cat([torch.zeros(len(mz), 1, dtype=torch.bool, device=pad.device), pad], 1)
            for block in self.blocks:
                x = block(x, pad)
            x = self.norm(x)
            mask = (~pad[:, 1:]).float().unsqueeze(-1)
            mean = (x[:, 1:] * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.head(torch.cat([x[:, 0], mean], -1))

    return FPNet


def load_fp_models(nbits, device):
    """(single-spectrum nets, merged-spectrum nets, torch device) from the attached checkpoints."""
    import glob

    import torch

    paths = sorted(p for p in glob.glob(f"{INPUT_ROOT}/**/fp_*.pt", recursive=True)
                   if Path(p).name in ("fp_single_s2.pt", "fp_merged_m1.pt"))
    if len(paths) != 2:
        raise FileNotFoundError(f"Expected fp_single_s2.pt and fp_merged_m1.pt, found {paths}")
    if device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested but the cloud runtime has no available CUDA GPU.")
    dev = "cuda" if device == "gpu" else "cpu"
    FPNet = make_fp_net()
    single, merged = [], []
    for path in paths:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        if int(ck["nbits"]) != nbits:
            raise ValueError(f"{path} predicts {ck['nbits']} bits; the pool has {nbits}")
        net = FPNet(int(ck["nbits"]), d=int(ck["d"]), layers=int(ck["layers"]))
        net.load_state_dict(ck["model"])
        (merged if "merged" in Path(path).name else single).append(net.to(dev).eval())
        log(f"loaded {Path(path).name}: d={ck['d']} layers={ck['layers']} step={ck.get('step')}")
    return single, merged, dev


def molecule_logits(models, spectra):
    """Fingerprint logits z for one molecule: mean of per-spectrum and merged-spectrum views."""
    import numpy as np
    import torch

    single, merged, dev = models

    def forward(nets, items):
        n = max(len(a) for a, *_ in items)
        mz = np.zeros((len(items), n), np.float32)
        it = np.zeros((len(items), n), np.float32)
        pad = np.ones((len(items), n), bool)
        for i, (a, b, *_) in enumerate(items):
            mz[i, :len(a)] = a
            it[i, :len(b)] = b
            pad[i, :len(a)] = False
        t = lambda x: torch.as_tensor(x, device=dev)
        args = (t(mz), t(it), t(pad),
                t(np.array([x[2] for x in items], np.float32)),
                t(np.array([FP_ADDUCTS.index(x[3]) if x[3] in FP_ADDUCTS else len(FP_ADDUCTS) - 1 for x in items])),
                t(np.array([instrument_family(x[4]) for x in items])),
                t(np.array([x[5] for x in items], np.float32)),
                t(np.array([x[6] for x in items], np.float32)))
        with torch.no_grad():
            return np.mean([net(*args).float().mean(0).cpu().numpy() for net in nets], axis=0)

    views = []
    items = []
    for s in spectra:
        a, b = fp_prep_peaks(s["mz"], s["it"], s["prec"])
        if len(a):
            items.append((a, b, s["prec"], s["adduct"], s["instrument"], s["ce"], s["mode"]))
    if single and items:
        views.append(forward(single, items))
    if merged and spectra:
        prec = float(np.median([s["prec"] for s in spectra]))
        a, b = fp_prep_peaks(*merge_peaks(spectra), prec)
        if len(a):
            mode = float(np.mean([s["mode"] for s in spectra]))
            views.append(forward(merged, [(a, b, prec, spectra[0]["adduct"], spectra[0]["instrument"], 25.0, mode)]))
    return np.mean(views, axis=0).astype(np.float32) if views else None


def molecules_logits(models, spectra_lists, batch_size=1):
    """Batch independent spectra; retain each molecule's single/merged view means."""
    if not models:
        return [None] * len(spectra_lists)
    if batch_size <= 1 or models[2] != "cuda":
        return [molecule_logits(models, spectra) if spectra else None for spectra in spectra_lists]
    import numpy as np
    import torch

    single, merged, dev = models
    views = [[] for _ in spectra_lists]
    for nets, merged_view in ((single, False), (merged, True)):
        if not nets:
            continue
        items, owners = [], []
        for owner, spectra in enumerate(spectra_lists):
            if merged_view and spectra:
                prec = float(np.median([s["prec"] for s in spectra]))
                a, b = fp_prep_peaks(*merge_peaks(spectra), prec)
                if len(a):
                    items.append((a, b, prec, spectra[0]["adduct"], spectra[0]["instrument"],
                                  25.0, float(np.mean([s["mode"] for s in spectra]))))
                    owners.append(owner)
            elif not merged_view:
                for s in spectra:
                    a, b = fp_prep_peaks(s["mz"], s["it"], s["prec"])
                    if len(a):
                        items.append((a, b, s["prec"], s["adduct"], s["instrument"], s["ce"], s["mode"]))
                        owners.append(owner)
        grouped = [[] for _ in spectra_lists]
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            n = max(len(x[0]) for x in batch)
            mz = np.zeros((len(batch), n), np.float32)
            it = np.zeros_like(mz)
            pad = np.ones((len(batch), n), bool)
            for i, (a, b, *_) in enumerate(batch):
                mz[i, :len(a)], it[i, :len(b)], pad[i, :len(a)] = a, b, False
            t = lambda x: torch.as_tensor(x, device=dev)
            args = (t(mz), t(it), t(pad), t(np.array([x[2] for x in batch], np.float32)),
                    t(np.array([FP_ADDUCTS.index(x[3]) if x[3] in FP_ADDUCTS else len(FP_ADDUCTS)-1 for x in batch])),
                    t(np.array([instrument_family(x[4]) for x in batch])),
                    t(np.array([x[5] for x in batch], np.float32)),
                    t(np.array([x[6] for x in batch], np.float32)))
            with torch.no_grad():
                predictions = np.mean([net(*args).float().cpu().numpy() for net in nets], axis=0)
            for owner, prediction in zip(owners[start:start + batch_size], predictions):
                grouped[owner].append(prediction)
        for owner, predictions in enumerate(grouped):
            if predictions:
                views[owner].append(np.mean(predictions, axis=0))
    return [np.mean(v, axis=0).astype(np.float32) if v else None for v in views]


def fragment_masses(smiles, max_pair_bonds=34):
    """Neutral masses of the components left by breaking one bond, or two (small molecules)."""
    import numpy as np
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.zeros(0)
    weights = [MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * MONO["H"] for a in mol.GetAtoms()]
    if not weights or min(weights) == 0.0:
        return np.zeros(0)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    n = len(weights)

    def split(skip):
        parent = list(range(n))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for k, (a, b) in enumerate(bonds):
            if k not in skip:
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[ra] = rb
        sums = {}
        for atom, w in enumerate(weights):
            r = root(atom)
            sums[r] = sums.get(r, 0.0) + w
        return sums.values() if len(sums) > 1 else ()

    masses = {sum(weights)}
    for i in range(len(bonds)):
        masses.update(split({i}))
    if len(bonds) <= max_pair_bonds:
        for i in range(len(bonds)):
            for j in range(i + 1, len(bonds)):
                masses.update(split({i, j}))
    return np.array(sorted(masses))


def _frag_worker(smiles):
    try:
        return fragment_masses(smiles)
    except Exception:
        import numpy as np

        return np.zeros(0)


def explain_score(frag, specs, tol):
    """Max over spectra of the sqrt-intensity fraction matched by a fragment ion (+-2 H)."""
    import numpy as np

    if len(frag) == 0:
        return 0.0
    best = 0.0
    for qm, qp, mode in specs:
        if len(qm) == 0:
            continue
        charge = PROTON if mode > 0 else -PROTON
        ions = np.sort(np.concatenate([frag + dh * MONO["H"] + charge for dh in (-2, -1, 0, 1, 2)]))
        w = np.sqrt(np.asarray(qp, np.float64))
        idx = np.searchsorted(ions, qm)
        hit = np.zeros(len(qm), bool)
        for shift in (-1, 0):
            k = np.clip(idx + shift, 0, len(ions) - 1)
            hit |= np.abs(ions[k] - qm) <= tol
        best = max(best, float(w[hit].sum() / max(w.sum(), 1e-12)))
    return best


def add_fragment_scores(evidence_lists, queries_lists, pool, cfg):
    """Attach ev['frag'] for every candidate; fragment masses computed once per structure."""
    import multiprocessing

    import numpy as np

    needed = sorted({int(c) for evs in evidence_lists for ev in evs for c in ev["cand"]})
    log(f"MetFrag-lite: fragmenting {len(needed):,} candidate structures")
    started = time.monotonic()
    context = cfg.get("process_context", "fork")
    with multiprocessing.get_context(context).Pool(cfg["fp_workers"]) as workers:
        frags = workers.map(_frag_worker, [pool.smiles[c] for c in needed], chunksize=64)
    generated = time.monotonic()
    log(f"MetFrag-lite generation: {generated - started:.1f}s")
    table = dict(zip(needed, frags))
    pairs = [(ev,q) for evs,queries in zip(evidence_lists,queries_lists) for ev,q in zip(evs,queries)]
    def reference(pair):
        ev,q = pair
        return np.array([explain_score(table[int(c)],q["specs"],cfg["mz_tol"]) for c in ev["cand"]],np.float32)
    def cached(pair):
        ev,q = pair
        prepared = []
        for qm,qp,mode in q["specs"]:
            if len(qm):
                w = np.sqrt(np.asarray(qp,np.float64))
                prepared.append((qm,w,max(w.sum(),1e-12),1 if mode>0 else -1))
        values = []
        for c in ev["cand"]:
            frag,ions_by_mode,best = table[int(c)],{},0.0
            if len(frag):
                for qm,w,den,mode in prepared:
                    if mode not in ions_by_mode:
                        charge = PROTON if mode>0 else -PROTON
                        ions_by_mode[mode] = np.sort(np.concatenate([frag+dh*MONO["H"]+charge for dh in (-2,-1,0,1,2)]))
                    ions = ions_by_mode[mode]
                    idx,hit = np.searchsorted(ions,qm),np.zeros(len(qm),bool)
                    for shift in (-1,0):
                        k = np.clip(idx+shift,0,len(ions)-1)
                        hit |= np.abs(ions[k]-qm)<=cfg["mz_tol"]
                    best = max(best,float(w[hit].sum()/den))
            values.append(best)
        return np.array(values,np.float32)
    use_cached,first = False,[]
    if cfg.get("optimize_frag_match"):
        began = time.monotonic()
        first = list(map(reference,pairs[:32]))
        reference_s = time.monotonic()-began
        began = time.monotonic()
        trial = list(map(cached,pairs[:32]))
        cached_s = time.monotonic()-began
        exact = all(np.array_equal(a,b) for a,b in zip(first,trial))
        use_cached = exact and cached_s<reference_s*0.95
        cfg["fragment_match_plan"] = {"cached":use_cached,"pilot_queries":len(first),"scores_exact":exact,
                                       "reference_s":round(reference_s,3),"cached_s":round(cached_s,3)}
        log(f"fragment matching pilot: {cfg['fragment_match_plan']}")
    for (ev,_),scores in zip(pairs[:len(first)],first):
        ev["frag"] = scores
    for pair in pairs[len(first):]:
        pair[0]["frag"] = cached(pair) if use_cached else reference(pair)
    matched = time.monotonic()
    log(f"MetFrag-lite matching: {matched - generated:.1f}s")
    return {"fragment_generation_s": round(generated-started, 3), "fragment_matching_s": round(matched-generated, 3)}


def collision_energy(value):
    import numpy as np

    try:
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) else 25.0
    except (TypeError, ValueError):
        return 25.0


def unseen_candidate_rows(keys, known):
    """Select each new structure once, preserving source order and extending known."""
    rows = []
    for row, key in enumerate(keys):
        if key not in known:
            rows.append(row)
            known.add(key)
    return rows


def build_pool(struct_keys, struct_smiles, cfg):
    import multiprocessing
    import pickle

    import numpy as np

    folder = Path(find_input("coco_fp.npy")).parent
    meta = pickle.load(open(folder / "coco_meta.pkl", "rb"))
    co_fp = np.load(folder / "coco_fp.npy")
    co_mass = np.load(folder / "coco_mass.npy")
    co_keys = np.asarray(meta["keys"], dtype=object)
    co_smiles = np.asarray(meta["smiles"], dtype=object)
    _FP["bits"] = np.load(folder / "fp_bits.npy")
    known = set(co_keys.tolist())
    if cfg.get("use_bio_db", False):
        bio_folder = Path(find_input("bio_fp.npy")).parent
        with (bio_folder / "bio_meta.pkl").open("rb") as handle:
            bio_meta = pickle.load(handle)
        bio_fp = np.load(bio_folder / "bio_fp.npy")
        bio_mass = np.load(bio_folder / "bio_mass.npy")
        bio_keys = np.asarray(bio_meta["keys"], dtype=object)
        bio_smiles = np.asarray(bio_meta["smiles"], dtype=object)
        if (bio_fp.ndim != 2 or bio_fp.shape[1] != co_fp.shape[1]
                or bio_fp.dtype != co_fp.dtype
                or bio_meta.get("nbits", meta["nbits"]) != meta["nbits"]
                or not (len(bio_fp) == len(bio_mass) == len(bio_keys) == len(bio_smiles))):
            raise ValueError("ChEBI/LIPID MAPS arrays do not match the COCONUT fingerprint format")
        valid = np.flatnonzero(np.isfinite(bio_mass))
        added = valid[unseen_candidate_rows(bio_keys[valid], known)]
        co_fp = np.vstack([co_fp, bio_fp[added]])
        co_mass = np.concatenate([co_mass, bio_mass[added]])
        co_keys = np.concatenate([co_keys, bio_keys[added]])
        co_smiles = np.concatenate([co_smiles, bio_smiles[added]])
        log(f"ChEBI/LIPID MAPS added {len(added):,} of {len(bio_keys):,} structures")
    todo = [i for i, k in enumerate(struct_keys) if k not in known]
    log(f"COCONUT {len(co_keys):,} structures; fingerprinting {len(todo):,} training structures")
    if _fp_worker("CCO") is None:
        raise RuntimeError("Fingerprint worker initialization failed on a valid synthetic SMILES")
    # Spawned workers receive bit selection explicitly; forked Kaggle workers use the same initializer.
    with multiprocessing.get_context(cfg.get("process_context", "fork")).Pool(
            cfg["fp_workers"], initializer=_init_fp_worker, initargs=(_FP["bits"],)) as workers:
        results = workers.map(_fp_worker, [struct_smiles[i] for i in todo], chunksize=500)
    if todo and sum(r is not None for r in results) < len(todo)*0.9:
        raise RuntimeError("More than 10% of training structures failed fingerprint generation; refusing partial pool")
    ok = [j for j, r in enumerate(results) if r is not None]
    tr_fp = np.stack([results[j][0] for j in ok]) if ok else np.zeros((0, co_fp.shape[1]), np.uint8)
    tr_mass = np.array([results[j][1] for j in ok])
    fp = np.vstack([co_fp, tr_fp])
    mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_keys, np.asarray([struct_keys[todo[j]] for j in ok], dtype=object)])
    smiles = np.concatenate([co_smiles, np.asarray([struct_smiles[todo[j]] for j in ok], dtype=object)])
    good = np.isfinite(mass)
    log(f"pool {int(good.sum()):,} structures ({len(todo) - len(ok)} training SMILES failed)")
    return Pool(fp[good], mass[good], keys[good], smiles[good], int(meta["nbits"]))


def load_library(path, struct_index, kernels, cfg):
    """Clean every training spectrum once; keep compact per-spectrum metadata."""
    import numpy as np
    import pyarrow.parquet as pq

    cols = ["inchikey14", "adduct", "precursor_mz", "molecular_formula", "instrument_type",
            "ionization_mode", "ingest_lib", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    parts = {k: [] for k in ("counts", "mz", "it", "sid", "prec", "nm_prec", "nm_form", "mode", "tims", "npex",
                             "simple", "lib")}
    form_cache = {}
    lib_codes = {}
    raw = {}  # uncleaned enveda-np-examples spectra, for the fingerprint models' own transform
    for batch in pq.ParquetFile(path).iter_batches(batch_size=cfg["train_batch_rows"], columns=cols):
        base = sum(len(c) for c in parts["counts"])
        col = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
        ms, ml, mzv = list_parts(col["ms2_mzs"])
        its, il, itv = list_parts(col["ms2_normalized_intensities"])
        prec = col["precursor_mz"].to_numpy(zero_copy_only=False).astype(np.float64)
        counts, out_mz, out_it = clean_parts(kernels, ms, ml, mzv, its, il, itv, prec, cfg)
        adduct = np.asarray(col["adduct"].to_pylist(), dtype=object)
        formula = col["molecular_formula"].to_pylist()
        for f in set(formula):
            if f not in form_cache:
                form_cache[f] = formula_mass(f)
        instrument = col["instrument_type"].to_pylist()
        lib = np.asarray(col["ingest_lib"].to_pylist(), dtype=object)
        parts["counts"].append(counts)
        parts["mz"].append(out_mz)
        parts["it"].append(out_it)
        parts["sid"].append(struct_index.get_indexer(col["inchikey14"].to_pylist()).astype(np.int64))
        parts["prec"].append(prec)
        parts["nm_prec"].append(neutral_mass(prec, adduct))
        parts["nm_form"].append(np.array([form_cache[f] for f in formula]))
        parts["mode"].append(ion_mode(adduct, col["ionization_mode"].to_pylist()))
        parts["tims"].append(np.array(
            [isinstance(x, str) and "tims" in x.lower() for x in instrument]
        ) | (lib == "enveda-180") | (lib == "enveda-np-examples"))
        parts["npex"].append(lib == "enveda-np-examples")
        parts["simple"].append((adduct == "[M+H]+") | (adduct == "[M-H]-"))
        parts["lib"].append(np.array([lib_codes.setdefault(x, len(lib_codes)) for x in lib], np.int16))
        npex = np.flatnonzero(lib == "enveda-np-examples")
        if len(npex):
            energies = col["collision_energy_ev"].to_pylist()
            for i in npex:
                raw[base + int(i)] = {
                    "mz": mzv[ms[i]:ms[i] + ml[i]].copy(), "it": itv[its[i]:its[i] + il[i]].copy(),
                    "prec": float(prec[i]), "adduct": adduct[i], "instrument": instrument[i] or "timsTOF",
                    "ce": collision_energy(energies[i]), "mode": float(parts["mode"][-1][i]),
                }
        log(f"  library rows {sum(len(c) for c in parts['counts']):,}")
    L = {k: np.concatenate(v) for k, v in parts.items()}
    L["raw"] = raw
    L["lib_names"] = {code: name for name, code in lib_codes.items()}
    L["off"] = np.concatenate([[0], np.cumsum(L.pop("counts"))]).astype(np.int64)
    L["npk"] = np.diff(L["off"])
    L["nm"] = np.where(np.isfinite(L["nm_form"]), L["nm_form"], L["nm_prec"])
    ok = np.isfinite(L["nm"]) & (L["sid"] >= 0) & (L["npk"] > 0)
    L["valid"] = ok
    order = np.argsort(np.where(ok, L["nm"], np.inf), kind="mergesort")
    L["order"] = order[: int(ok.sum())]
    L["snm"] = L["nm"][L["order"]]
    return L


def build_reps(L, active):
    """One spectrum per (structure, ion mode): timsTOF first, then [M+H]+/[M-H]-, then most peaks."""
    import numpy as np

    idx = np.flatnonzero(active & L["valid"])
    priority = L["tims"][idx].astype(np.int64) * 2 + L["simple"][idx].astype(np.int64)
    order = np.lexsort((-L["npk"][idx], -priority, L["mode"][idx], L["sid"][idx]))
    idx = idx[order]
    sid = L["sid"][idx]
    mode = L["mode"][idx]
    first = np.ones(len(idx), bool)
    first[1:] = (sid[1:] != sid[:-1]) | (mode[1:] != mode[:-1])
    rep = idx[first]
    out = {}
    for m in (1, -1):
        r = rep[L["mode"][rep] == m]
        r = r[np.argsort(L["nm"][r], kind="mergesort")]
        out[m] = (r, L["sid"][r], L["nm"][r])
    return out


def score_query(query, L, reps, lib_active, pool, sid2pool, kernels, cfg, z=None):
    """Evidence for one molecule: candidate pool indices, library, analog and f.z channels."""
    import numpy as np

    nm = query["nm"]
    tol = cfg["mz_tol"]
    lib = {}
    lo = np.searchsorted(L["snm"], nm * (1 - cfg["ppm"] / 1e6), "left")
    hi = np.searchsorted(L["snm"], nm * (1 + cfg["ppm"] / 1e6), "right")
    rows = L["order"][lo:hi]
    rows = rows[lib_active[rows]]
    if len(rows):
        zeros = np.zeros(len(rows))
        for qm, qp, _ in query["specs"]:
            sims = kernels["search"](qm, qp, rows, L["off"], L["mz"], L["it"], tol, zeros, False)
            for s, v in zip(L["sid"][rows], sims):
                if v > lib.get(s, 0.0):
                    lib[s] = float(v)
    analog = {}
    for qm, qp, mode in query["specs"]:
        r, rs, rn = reps[mode]
        lo = np.searchsorted(rn, nm - cfg["analog_window"], "left")
        hi = np.searchsorted(rn, nm + cfg["analog_window"], "right")
        if hi <= lo:
            continue
        rows_w, sids_w, shifts_w = r[lo:hi], rs[lo:hi], nm - rn[lo:hi]
        if len(rows_w) > cfg["analog_prefilter"]:
            ub = kernels["bound"](qm, qp, rows_w, L["off"], L["mz"], L["it"], tol, shifts_w)
            keep = np.argpartition(-ub, cfg["analog_prefilter"] - 1)[: cfg["analog_prefilter"]]
            rows_w, sids_w, shifts_w = rows_w[keep], sids_w[keep], shifts_w[keep]
        sims = kernels["search"](qm, qp, rows_w, L["off"], L["mz"], L["it"], tol, shifts_w, True)
        k = min(cfg["n_analog"], len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        for s, v in zip(sids_w[top], sims[top]):
            if v > analog.get(s, 0.0):
                analog[s] = float(v)
    ranked = sorted(analog.items(), key=lambda x: -x[1])
    ranked = [(sid2pool[s], v) for s, v in ranked if sid2pool[s] >= 0][: cfg["n_analog"]]
    a_idx = np.array([p for p, _ in ranked], np.int64)
    a_sim = np.array([v for _, v in ranked], np.float32)

    cand = pool.window(nm, cfg["ppm"])
    if len(cand) == 0:
        cand = pool.window(nm, cfg["ppm_fallback"])
    extra = [sid2pool[s] for s in lib if sid2pool[s] >= 0]
    cand = np.unique(np.concatenate([cand, np.asarray(extra, np.int64)])).astype(np.int64)
    fallback = len(cand) == 0
    if fallback:
        cand = a_idx[: cfg["top_n"]]
    pool_sid = {p: s for s, p in ((s, sid2pool[s]) for s in lib)}
    lib_vec = np.array([lib.get(pool_sid.get(c, -1), 0.0) for c in cand], np.float32)
    cf = pool.bits(cand) if len(cand) else np.zeros((0, pool.nbits), np.float32)
    if len(a_idx) and len(cand):
        af = pool.bits(a_idx)
        inter = cf @ af.T
        tan = inter / (cf.sum(1)[:, None] + af.sum(1)[None, :] - inter + 1e-9)
    else:
        tan = np.zeros((len(cand), 0), np.float32)
    fz = cf @ z if z is not None else np.zeros(len(cand), np.float32)
    if len(fz) > 1:
        fz = (fz - fz.mean()) / (fz.std() + 1e-9)
    else:
        fz = np.zeros(len(cand), np.float32)
    err = np.abs(pool.mass[cand] - nm) / nm * 1e6
    return {"cand": cand, "lib": lib_vec, "tan": tan.astype(np.float32), "asim": a_sim,
            "fz": fz.astype(np.float32), "pop": pool.pop[cand], "err": err, "fallback": fallback}


def score_queries(queries, L, reps, active, pool, sid2pool, kernels, cfg, logits=None, stage="search"):
    """Ordered query search, with a bounded cloud-only serial/threaded throughput pilot."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    import numba

    pairs = list(zip(queries, logits if logits is not None else [None]*len(queries)))
    if not pairs:
        return []
    def one(pair):
        return score_query(pair[0], L, reps, active, pool, sid2pool, kernels, cfg, pair[1])
    outer, inner = 1, numba.get_num_threads()
    first = []
    if cfg.get("auto_query_workers") and "search_plan" not in cfg:
        # Compile once before concurrent entry, and reuse this result.
        first = [one(pairs[0])]
        pilot = pairs[1:33]
        began = time.monotonic()
        serial = list(map(one, pilot))
        serial_s = time.monotonic()-began
        candidates = []
        if pilot and numba.threading_layer() in {"omp", "tbb"}:
            budget = numba.get_num_threads()
            choices = [(n,max(1,budget//n)) for n in (2,4,6,8) if n<=budget]
            if budget>=16:
                choices.append((4,4))
            for outer_trial,inner_trial in choices:
                if outer_trial*inner_trial > numba.get_num_threads():
                    continue
                began = time.monotonic()
                with ThreadPoolExecutor(outer_trial,initializer=numba.set_num_threads,initargs=(inner_trial,)) as executor:
                    threaded = list(executor.map(one,pilot))
                elapsed = time.monotonic()-began
                equivalent = all(a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
                                 for a,b in zip(serial,threaded))
                candidates.append({"outer_workers":outer_trial,"inner_threads":inner_trial,
                                   "seconds":round(elapsed,3),"evidence_exact":equivalent})
            acceptable = [p for p in candidates if p["evidence_exact"] and p["seconds"]<serial_s*0.95]
            if acceptable:
                selected = min(acceptable,key=lambda p:p["seconds"])
                outer,inner = selected["outer_workers"],selected["inner_threads"]
        cfg["search_plan"] = {"outer_workers": outer, "inner_threads": inner,
                              "pilot_queries": len(pilot), "serial_s": round(serial_s,3),
                              "candidates":candidates,"threading_layer":numba.threading_layer()}
        log(f"search throughput pilot: {cfg['search_plan']}")
        first.extend(serial)
    elif "search_plan" in cfg:
        outer, inner = cfg["search_plan"]["outer_workers"], cfg["search_plan"]["inner_threads"]
    began = time.monotonic()
    result = list(first)
    def collect(iterator):
        for ev in iterator:
            result.append(ev)
            if len(result)%50 == 0 or len(result) == len(pairs):
                elapsed = time.monotonic()-began
                count = len(result)-len(first)
                log(f"{stage}: {len(result)}/{len(pairs)} queries; {count/max(elapsed,0.001):.2f} queries/s")
    remaining = pairs[len(first):]
    if outer > 1:
        with ThreadPoolExecutor(outer, initializer=numba.set_num_threads, initargs=(inner,)) as executor:
            collect(executor.map(one, remaining))
    else:
        collect(map(one, remaining))
    return result


def rank(evidence, params, top_n):
    """Score = A * library * [library >= tau] + max_a sim_a^P * Tanimoto + B * standardised f.z.

    v1 showed an ungated library term promotes same-mass impostors in the Class-2 simulation,
    so library similarity only counts once it clears the gate tau.
    """
    import numpy as np

    if evidence["tan"].shape[1]:
        ap = (evidence["tan"] * (evidence["asim"] ** params["sim_power"])[None, :]).max(1)
    else:
        ap = np.zeros(len(evidence["cand"]), np.float32)
    lib = evidence["lib"] * (evidence["lib"] >= params["lib_gate"])
    score = (params["lib_weight"] * lib + ap + params["fz_weight"] * evidence["fz"]
             + params.get("pop_weight", 0.0) * evidence["pop"])
    order = np.lexsort((evidence["err"], -score))[:top_n]
    return evidence["cand"][order]


def mrr(evidences, truths, pool, params, top_n):
    total = 0.0
    for ev, truth in zip(evidences, truths):
        keys = pool.keys[rank(ev, params, top_n)]
        for position, key in enumerate(keys):
            if key == truth:
                total += 1.0 / (position + 1)
                break
    return total / max(len(truths), 1)


RANK_FEATURES = [
    "lib", "lib_max", "lib_gap", "lib_rank", "ap1", "ap2", "ap3", "ap4", "ap3_max", "ap3_gap", "ap3_rank",
    "best_tan", "top_tan", "top_sim", "mean_tan", "close_analogs", "err_ppm", "log_n", "n_lib_hits",
]
FRAG_FEATURES = ["frag", "frag_gap", "frag_rank"]


def feature_names(cfg):
    return RANK_FEATURES + (FRAG_FEATURES if cfg["use_frag"] else [])


def rank_features(ev):
    """Per-candidate features from the library, analog and (if present) fragmentation channels.

    Deliberately excludes f.z (the public fingerprint models were trained on most training
    structures, so f.z on simulated training queries would be optimistic) and candidate
    provenance (in simulations the answer is always a training structure).
    """
    import numpy as np

    n = len(ev["cand"])
    lib = ev["lib"].astype(np.float32)
    tan, asim = ev["tan"], ev["asim"]
    if tan.shape[1]:
        aps = [(tan * (asim ** p)[None, :]).max(1) for p in (1.0, 2.0, 3.0, 4.0)]
        w = asim ** 3
        best_tan, top_tan, top_sim = tan.max(1), tan[:, 0], float(asim[0])
        mean_tan = (tan * w[None, :]).sum(1) / (w.sum() + 1e-9)
        close = ((tan >= 0.8) & (asim[None, :] >= 0.5)).sum(1)
    else:
        aps = [np.zeros(n, np.float32)] * 4
        best_tan = top_tan = mean_tan = close = np.zeros(n, np.float32)
        top_sim = 0.0

    def rank_norm(x):
        r = np.empty(n)
        r[np.argsort(-x, kind="mergesort")] = np.arange(n)
        return r / max(n - 1, 1)

    ap3 = aps[2]
    columns = [
        lib, np.full(n, lib.max()), lib - lib.max(), rank_norm(lib),
        *aps, np.full(n, ap3.max()), ap3 - ap3.max(), rank_norm(ap3),
        best_tan, top_tan, np.full(n, top_sim), mean_tan, close,
        ev["err"], np.full(n, np.log(n)), np.full(n, float((lib > 0).sum())),
    ]
    if "frag" in ev:
        frag = ev["frag"]
        columns += [frag, frag - frag.max(), rank_norm(frag)]
    return np.column_stack(columns).astype(np.float32)


def sample_training_queries(L, held_sids, cfg):
    """Disjoint structure sets for ranker training, excluding the validation structures.

    class2: every spectrum of the structure leaves the library (no reference spectra).
    class1: only the query spectra leave; the structure keeps spectra from other libraries.
    Query spectra prefer the test instrument (timsTOF), then [M+H]+/[M-H]-.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(cfg["seed"])
    rows = np.flatnonzero(L["valid"] & ~np.isin(L["sid"], held_sids))
    frame = pd.DataFrame({"row": rows, "sid": L["sid"][rows], "lib": L["lib"][rows],
                          "pri": L["tims"][rows].astype(int) * 2 + L["simple"][rows].astype(int)})
    frame = frame.sort_values(["sid", "pri", "row"], ascending=[True, False, True])
    n_libs = frame.groupby("sid").lib.nunique()
    sids = n_libs.index.to_numpy()
    c2 = rng.choice(sids, size=min(cfg["train_queries_class2"], len(sids)), replace=False)
    multi = np.setdiff1d(n_libs.index[n_libs >= 2].to_numpy(), c2)
    c1 = rng.choice(multi, size=min(cfg["train_queries_class1"], len(multi)), replace=False)
    by_sid = frame.groupby("sid")
    k = cfg["train_query_spectra"]
    picks = {"class2": {int(s): by_sid.get_group(s).row.to_numpy()[:k] for s in c2}}
    picks["class1"] = {}
    for s in c1:
        g = by_sid.get_group(s)
        first_lib = g.lib.iloc[0]  # the library holding the best-priority spectrum
        picks["class1"][int(s)] = g.row[g.lib == first_lib].to_numpy()[:k]
    class1_rows = np.ones(len(L["sid"]), bool)
    class1_rows[np.concatenate(list(picks["class1"].values()))] = False
    masks = {"class2": ~np.isin(L["sid"], c2), "class1": class1_rows}
    return picks, masks


def train_rankers(samples, cfg, names):
    """LightGBM lambdarank on (features, label, query) rows; bagged over seeds."""
    import lightgbm as lgb
    import numpy as np

    X = np.concatenate([x for x, _ in samples])
    y = np.concatenate([lab for _, lab in samples])
    groups = [len(lab) for _, lab in samples]
    models = []
    for seed in cfg["ranker_seeds"]:
        model = lgb.LGBMRanker(objective="lambdarank", random_state=seed, verbose=-1, **cfg["ranker_params"])
        model.fit(X, y, group=groups)
        models.append(model)
    importance = np.mean([m.booster_.feature_importance("gain") for m in models], axis=0)
    return models, {name: round(float(v), 1) for name, v in zip(names, importance)}


def ranker_scores(ev, rankers):
    """Seed-averaged ranker score, z-scored within the molecule; cached on the evidence."""
    import numpy as np

    if "rk" not in ev:
        s = np.zeros(len(ev["cand"]))
        if len(s) > 1:
            s = np.mean([m.booster_.predict(rank_features(ev), num_threads=1) for m in rankers], axis=0)
            s = (s - s.mean()) / (s.std() + 1e-9)
        ev["rk"] = s
    return ev["rk"]


def rank_learned(ev, rankers, params, top_n):
    """Ranker + A * library * [library >= tau] + B * f.z.

    v3 alone lost Class 1 (C1 0.878 -> 0.633): trained mostly on Class-2 queries, it learned
    that structures with library spectra are usually wrong. The gated library term restores
    confident library matches without giving up the ranker's Class-2 ordering.
    """
    import numpy as np

    if len(ev["cand"]) == 0:
        return ev["cand"]
    lib = ev["lib"] * (ev["lib"] >= params["lib_gate"])
    score = (ranker_scores(ev, rankers) + params["lib_weight"] * lib + params["fz_weight"] * ev["fz"]
             + params.get("pop_weight", 0.0) * ev["pop"])
    order = np.lexsort((ev["err"], -score))[:top_n]
    return ev["cand"][order]


def mrr_of(ranked_lists, truths, pool):
    total = 0.0
    for picks, truth in zip(ranked_lists, truths):
        for position, key in enumerate(pool.keys[picks]):
            if key == truth:
                total += 1.0 / (position + 1)
                break
    return total / max(len(truths), 1)


def library_queries(L, rows_by_sid, struct_keys, cfg):
    """Pseudo-molecules from library rows, using the measured precursor neutral mass."""
    import numpy as np

    queries, truths = [], []
    for sid, rows in rows_by_sid.items():
        rows = rows[: cfg["max_query_spectra"]]
        nm = np.nanmedian(L["nm_prec"][rows]) if np.isfinite(L["nm_prec"][rows]).any() else L["nm"][rows[0]]
        specs = [(L["mz"][L["off"][r]:L["off"][r + 1]], L["it"][L["off"][r]:L["off"][r + 1]], int(L["mode"][r]))
                 for r in rows if L["npk"][r] > 0 and L["mode"][r] != 0]
        raw = [L["raw"][r] for r in rows if r in L["raw"] and L["mode"][r] != 0]
        if specs and np.isfinite(nm):
            queries.append({"nm": float(nm), "specs": specs, "raw": raw})
            truths.append(struct_keys[sid])
    return queries, truths


def environment_report(device):
    package_versions = {}
    for name in ("numpy", "pandas", "pyarrow", "numba", "torch", "lightgbm", "scikit-learn", "rdkit"):
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = None
    gpu = None
    if device == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A GPU was requested but the cloud runtime has no available CUDA GPU.")
        gpu = {
            "count": torch.cuda.device_count(),
            "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    return {"python": platform.python_version(), "packages": package_versions, "gpu": gpu}


def main(device: str = "cpu", *, server_authorization=None, output_directory=None) -> None:
    # This must precede numerical imports and any dataset access.
    global INPUT_ROOT, OUTPUT_ROOT
    if server_authorization is None:
        require_kaggle_cloud()
        if output_directory is not None:
            raise ValueError("Custom output directories are reserved for the authorized server.")
        INPUT_ROOT, OUTPUT_ROOT = Path("/kaggle/input"), Path("/kaggle/working")
        execution = {"runtime": "kaggle"}
    else:
        proof = require_server_cloud(server_authorization, device)
        root = Path(proof["project_root"]).resolve()
        INPUT_ROOT = root / "input"
        OUTPUT_ROOT = Path(output_directory).resolve() if output_directory else root / "outputs" / CONFIG["version"]
        if not OUTPUT_ROOT.is_relative_to(root / "outputs"):
            raise ValueError("Server outputs must stay inside the authorized project's outputs directory.")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        execution = {"runtime": "vast-ssh", **server_identity()}
    if device not in {"cpu", "gpu"}:
        raise ValueError("Device must be cpu or gpu")
    started = time.time()
    timings = {}
    cfg = dict(CONFIG)
    rdkit_version = install_rdkit()
    report = {"task": "submission", "device": device, "config": cfg, "rdkit": rdkit_version, "execution": execution,
              "competition_rerun": bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN")),
              **environment_report(device)}

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    comp = Path(find_input("test.parquet")).parent
    train_path = comp / "train.parquet"
    test = pd.read_parquet(comp / "test.parquet")
    log(f"test: {len(test):,} spectra / {test.molecule_id.nunique():,} molecules")

    structures = pq.read_table(train_path, columns=["inchikey14", "normalized_smiles"]).to_pandas()
    structures = structures.dropna().drop_duplicates("inchikey14")
    struct_keys = structures.inchikey14.to_numpy(dtype=object)
    struct_smiles = structures.normalized_smiles.to_numpy(dtype=object)
    struct_index = pd.Index(struct_keys)
    del structures
    pool = build_pool(struct_keys, struct_smiles, cfg)
    if cfg["pop_weight"]:
        pool.load_popularity()
    sid2pool = np.array([pool.index.get(k, -1) for k in struct_keys], np.int64)
    timings["pool_s"] = round(time.time() - started)

    kernels = make_kernels()
    L = load_library(train_path, struct_index, kernels, cfg)
    log(f"library: {len(L['sid']):,} spectra, {int(L['valid'].sum()):,} usable")
    timings["library_s"] = round(time.time() - started)

    # Held-out simulations from the instrument-matched natural-product examples.
    npex_rows = np.flatnonzero(L["npex"] & L["valid"])
    rows_by_sid = {}
    for r in npex_rows:
        rows_by_sid.setdefault(int(L["sid"][r]), []).append(r)
    rows_by_sid = {s: np.array(v) for s, v in rows_by_sid.items()}
    queries, truths = library_queries(L, rows_by_sid, struct_keys, cfg)
    held_sids = np.array(sorted(rows_by_sid))
    models = load_fp_models(pool.nbits, device) if cfg["use_fp_model"] else None
    fp_started = time.monotonic()
    query_z = molecules_logits(models, [q["raw"] for q in queries], cfg.get("fp_batch_size", 1))
    timings["validation_fp_s"] = round(time.monotonic()-fp_started, 3)
    log(f"fingerprint logits for {sum(z is not None for z in query_z)}/{len(queries)} validation queries")
    sims = {
        "class1": ~L["npex"],
        "class2": ~np.isin(L["sid"], held_sids),
    }
    evidence = {}
    for name, active in sims.items():
        reps = build_reps(L, active)
        evidence[name] = score_queries(queries, L, reps, active, pool, sid2pool, kernels, cfg, query_z, name)
        log(f"{name} simulation: {len(queries)} queries scored")
    timings["validation_s"] = round(time.time() - started)

    # Ranker training simulations: disjoint structures, each with its own library mask.
    train_sets = []
    if cfg["use_ranker"]:
        picks, masks = sample_training_queries(L, held_sids, cfg)
        for name in ("class2", "class1"):
            train_q, train_t = library_queries(L, picks[name], struct_keys, cfg)
            reps_t = build_reps(L, masks[name])
            evs = score_queries(train_q, L, reps_t, masks[name], pool, sid2pool, kernels, cfg, stage=f"train {name}")
            train_sets.append((name, train_q, train_t, evs))
            log(f"ranker simulation {name}: {len(train_q)} queries scored")
        timings["train_sims_s"] = round(time.time() - started)

    # Test evidence.
    reps = build_reps(L, np.ones(len(L["sid"]), bool))
    all_active = np.ones(len(L["sid"]), bool)
    test = test.reset_index(drop=True)
    cleaned = clean_lists(kernels, list(test.ms2_mzs), list(test.ms2_normalized_intensities),
                          test.precursor_mz.to_numpy(np.float64), cfg)
    test_nm = neutral_mass(test.precursor_mz.to_numpy(np.float64), test.adduct.to_numpy(dtype=object))
    test_mode = ion_mode(list(test.adduct), list(test.ionization_mode))
    test_ids, test_queries, test_evs, test_raw = [], [], [], []
    stats = {"fallback": 0, "no_spectra": 0, "no_logits": 0}
    for molecule_id, rows in test.groupby("molecule_id", sort=False).groups.items():
        rows = np.asarray(rows)
        specs = [(cleaned[r][0], cleaned[r][1], int(test_mode[r])) for r in rows
                 if len(cleaned[r][0]) and test_mode[r] != 0]
        nm = np.nanmedian(test_nm[rows]) if np.isfinite(test_nm[rows]).any() else np.nan
        if not np.isfinite(nm):
            sign = 1.0 if test_mode[rows[0]] >= 0 else -1.0
            nm = float(np.median(test.precursor_mz.to_numpy()[rows])) - sign * PROTON
        stats["no_spectra"] += int(not specs)
        raw = []
        if models:
            raw = [{"mz": np.asarray(test.ms2_mzs[r], np.float64),
                    "it": np.asarray(test.ms2_normalized_intensities[r], np.float64),
                    "prec": float(test.precursor_mz[r]), "adduct": test.adduct[r],
                    "instrument": test.instrument_type[r] or "timsTOF",
                    "ce": collision_energy(test.collision_energy_ev[r]), "mode": float(test_mode[r])}
                   for r in rows if test_mode[r] != 0 and test.ms2_mzs[r] is not None]
        query = {"nm": float(nm), "specs": specs}
        test_ids.append(molecule_id)
        test_queries.append(query)
        test_raw.append(raw)
    fp_started = time.monotonic()
    test_z = molecules_logits(models, test_raw, cfg.get("fp_batch_size", 1))
    timings["test_fp_s"] = round(time.monotonic()-fp_started, 3)
    log(f"test fingerprint inference: {len(test_queries)} molecules in {timings['test_fp_s']:.1f}s")
    test_evs = score_queries(test_queries, L, reps, all_active, pool, sid2pool, kernels, cfg, test_z, "test")
    stats["no_logits"] = sum(z is None for z in test_z) if models else 0
    stats["fallback"] = sum(int(ev["fallback"]) for ev in test_evs)
    timings["test_s"] = round(time.time() - started)

    # v4: MetFrag-lite explain score for every candidate of every query.
    if cfg["use_frag"]:
        timings.update(add_fragment_scores(
            [evidence["class1"], evidence["class2"], *[s[3] for s in train_sets], test_evs],
            [queries, queries, *[s[1] for s in train_sets], test_queries], pool, cfg))
        timings["frag_s"] = round(time.time() - started)

    import itertools

    grid = []
    w1 = cfg["objective_class1_weight"]
    fz_grid = cfg["grid_fz_weight"] if models else [0.0]
    for a, tau, p, b in itertools.product(cfg["grid_lib_weight"], cfg["grid_lib_gate"],
                                          cfg["grid_sim_power"], fz_grid):
        if a == 0 and tau != cfg["grid_lib_gate"][0]:
            continue  # the gate is irrelevant without a library term
        params = {"lib_weight": a, "lib_gate": tau, "sim_power": p, "fz_weight": b}
        c1 = mrr(evidence["class1"], truths, pool, params, cfg["top_n"])
        c2 = mrr(evidence["class2"], truths, pool, params, cfg["top_n"])
        grid.append({**params, "class1": round(c1, 4), "class2": round(c2, 4),
                     "objective": round(w1 * c1 + (1 - w1) * c2, 4)})
    best = max(grid, key=lambda g: g["objective"])
    for g in sorted(grid, key=lambda g: -g["objective"])[:25]:
        log(f"  A={g['lib_weight']:<4} tau={g['lib_gate']:<4} P={g['sim_power']:<4} B={g['fz_weight']:<5} "
            f"C1={g['class1']:.4f} C2={g['class2']:.4f} obj={g['objective']:.4f}")
    log(f"selected {best}")
    report["validation"] = {"queries": len(queries), "grid": grid, "selected": best}

    # v3: learned ranker on simulated queries outside the validation structures.
    rankers, ranker_best = None, None
    if train_sets:
        samples, counts = [], {}
        for name, _, train_t, evs in train_sets:
            kept = 0
            for ev, truth in zip(evs, train_t):
                labels = (pool.keys[ev["cand"]] == truth).astype(np.int32)
                if labels.sum() == 1 and len(labels) > 1:
                    samples.append((rank_features(ev), labels))
                    kept += 1
            counts[name] = {"queries": len(evs), "answer_in_window": kept}
            log(f"ranker training {name}: {len(evs)} queries, {kept} with the answer in the window")
        rankers, importance = train_rankers(samples, cfg, feature_names(cfg))
        timings["ranker_s"] = round(time.time() - started)
        ranker_grid = []
        for a, tau, b in itertools.product(cfg["grid_ranker_lib_weight"], cfg["grid_ranker_lib_gate"],
                                           cfg["grid_ranker_fz_weight"] if models else [0.0]):
            if a == 0 and tau != cfg["grid_ranker_lib_gate"][0]:
                continue
            params = {"lib_weight": a, "lib_gate": tau, "fz_weight": b}
            c1 = mrr_of([rank_learned(ev, rankers, params, cfg["top_n"]) for ev in evidence["class1"]], truths, pool)
            c2 = mrr_of([rank_learned(ev, rankers, params, cfg["top_n"]) for ev in evidence["class2"]], truths, pool)
            ranker_grid.append({**params, "class1": round(c1, 4), "class2": round(c2, 4),
                                "objective": round(w1 * c1 + (1 - w1) * c2, 4)})
        for g in sorted(ranker_grid, key=lambda g: -g["objective"])[:15]:
            log(f"  ranker A={g['lib_weight']:<4} tau={g['lib_gate']:<4} B={g['fz_weight']:<5} "
                f"C1={g['class1']:.4f} C2={g['class2']:.4f} obj={g['objective']:.4f}")
        ranker_best = max(ranker_grid, key=lambda g: g["objective"])
        report["ranker"] = {"training": counts, "rows": int(sum(len(lab) for _, lab in samples)),
                            "importance": importance, "grid": ranker_grid, "selected": ranker_best}
        if ranker_best["objective"] <= best["objective"]:
            log(f"ranker {ranker_best['objective']:.4f} does not beat the linear blend {best['objective']:.4f}")
            rankers = None
    report["final_ranking"] = "learned" if rankers else "linear"
    log(f"final ranking: {report['final_ranking']}")
    # The popularity prior is set from the leaderboard, not chosen on validation: the validation
    # compounds are common natural products, so popularity would look better than it is.
    final = dict(ranker_best if rankers else best, pop_weight=cfg["pop_weight"])
    report["final_params"] = final
    if cfg["pop_weight"]:
        ranked = (lambda ev: rank_learned(ev, rankers, final, cfg["top_n"])) if rankers else \
                 (lambda ev: rank(ev, final, cfg["top_n"]))
        c1 = mrr_of([ranked(ev) for ev in evidence["class1"]], truths, pool)
        c2 = mrr_of([ranked(ev) for ev in evidence["class2"]], truths, pool)
        report["with_popularity"] = {"pop_weight": cfg["pop_weight"], "class1": round(c1, 4), "class2": round(c2, 4)}
        log(f"with popularity {cfg['pop_weight']}: C1={c1:.4f} C2={c2:.4f} (optimistic: common compounds)")
    report["final_params"] = final

    rows_out = []
    for molecule_id, ev in zip(test_ids, test_evs):
        if rankers:
            picks = rank_learned(ev, rankers, final, cfg["top_n"])
        else:
            picks = rank(ev, final, cfg["top_n"])
        smiles = [s for s in pool.smiles[picks] if isinstance(s, str) and s]
        rows_out.append({"molecule_id": molecule_id, "smiles": ";".join(smiles) if smiles else "C"})

    submission = pd.DataFrame(rows_out)
    expected = test.molecule_id.unique()
    assert submission.molecule_id.is_unique and set(submission.molecule_id) == set(expected)
    assert submission.smiles.notna().all() and (submission.smiles.str.count(";") < cfg["top_n"]).all()
    submission.to_csv(OUTPUT_ROOT / "submission.csv", index=False)
    cands = np.array([len(ev["cand"]) for ev in test_evs])
    stats.update({"molecules": len(submission), "median_candidates": float(np.median(cands)),
                  "max_candidates": int(cands.max())})
    report.update({"test": stats, "timings": timings, "total_s": round(time.time() - started)})
    (OUTPUT_ROOT / "run_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    log(f"wrote submission.csv ({len(submission)} molecules) in {report['total_s']}s")


if __name__ == "__main__":
    main()
