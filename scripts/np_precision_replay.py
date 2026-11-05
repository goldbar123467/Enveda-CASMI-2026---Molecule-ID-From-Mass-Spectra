"""Isolate input precision on fixed, identity-excluded NP retrieval queries on the RTX 2070.

Raw spectra and q16 spectra use identical molecule/metadata/duplicate handling. This
tests a preprocessing mismatch, not candidate generation or a hidden leaderboard score.
"""
import hashlib
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import local_seeds as L
from features import peaks, metadata
from professional_replay import raw_scores
from tail_fusion import paired_summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local/professional'


def rr(scores):
    rank = 1 + int((scores[1:] >= scores[0]).sum())
    return 1 / rank if rank <= 25 else 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', action='store_true', help='Use only enveda-np-examples timsTOF spectra')
    args = parser.parse_args()
    start = time.time()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    cfg = json.loads((L.KIT / 'config.json').read_text())
    cfg['work'] = str(L.CACHE)
    data, device = L.T.Data(cfg), torch.device('cuda')
    raw = pd.read_parquet(OUT / 'np_raw_canonical.parquet')
    if args.native:
        raw = raw[raw.ingest_lib.eq('enveda-np-examples') & raw.instrument_type.eq('timsTOF')]
    mismatch = int((raw.canonical_key != raw.molecule_id).sum())
    raw = raw[raw.canonical_key == raw.molecule_id]
    heldout = set((L.OUT / 'np_holdout_keys.txt').read_text().split())
    assert set(raw.molecule_id).issubset(heldout)
    candidates = L.candidates(cfg, data, 'np')
    assert len(candidates) == 234 and all(data.mol.key.iloc[i] in heldout for i in candidates)
    queryof, errors, mz_errors = {}, [], []
    for mid, group in raw.groupby('molecule_id', sort=False):
        records, seen = [], set()
        for row in group.itertuples(index=False):
            try:
                x, length = peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, cfg['max_peaks'])
                if length < cfg['min_peaks']:
                    continue
                meta, adduct = metadata(row.precursor_mz, row.adduct, row.ionization_mode, row.collision_energy_ev)
                quantized = x.astype('<f2')
                digest = hashlib.sha256(quantized.tobytes() + meta.astype('<f4').tobytes() + bytes([adduct])).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                mz_errors.extend((1000 * np.abs(x[:length, 0] - quantized[:length, 0].astype(np.float32))).tolist())
                records.append((x, length, meta, adduct))
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
        queryof[str(mid)] = records
    print('QUERIES', len(queryof), 'accepted spectra', sum(map(len, queryof.values())), 'excluded canonical mismatch', mismatch, flush=True)
    models = {str(seed): L.load(cfg, L.OUT / f'holdout_np/casmi26_holdout_{seed}_e64.pt', device) for seed in [7, 8, 9]}
    evidence, score_cache = [], {}
    for index, (i, ids) in enumerate(candidates.items()):
        key = str(data.mol.key.iloc[i])
        assert ids[0] == i
        records = queryof.get(key, [])
        row = dict(key=key, spectra=len(records), cached_spectra=len(data.by_mol[i]))
        if not records:
            for name in models:
                row[f'{name}_raw'] = row[f'{name}_q16'] = 0.
        else:
            fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
            for name, model in models.items():
                for mode in ['raw', 'q16']:
                    score = raw_scores(model, cfg, records, fp, device, mode == 'q16')
                    row[f'{name}_{mode}'] = rr(score)
                    score_cache[f'{key}__{name}__{mode}'] = score
        evidence.append(row)
        if (index + 1) % 25 == 0:
            print('SCORED', index + 1, 'of', len(candidates), round(time.time() - start), 's', flush=True)
    frame = pd.DataFrame(evidence)
    report = dict(device=torch.cuda.get_device_name(), n=234, fresh_test=False,
                  input_source='native Enveda timsTOF' if args.native else 'all reference collections for NP identities',
                  candidate_count=256, excluded_canonical_mismatch_spectra=mismatch,
                  accepted_spectra=int(frame.spectra.sum()), missing_query_molecules=int((frame.spectra == 0).sum()),
                  exact_cached_spectrum_counts=int((frame.spectra == frame.cached_spectra).sum()),
                  parsing_errors=len(errors), mz_absolute_quantization_error_da={
                      str(q): float(np.quantile(mz_errors, q)) for q in [.5, .9, .99, 1.]},
                  models={}, seconds=time.time() - start,
                  limitation='Same 234 molecules across seeds; not 702 independent observations. Fixed decoys, no full pipeline.')
    for name in models:
        report['models'][name] = paired_summary(frame[f'{name}_raw'], frame[f'{name}_q16'])
    # The paired molecular unit is preserved for the mean score over the three seeds.
    report['mean_seed_rr'] = paired_summary(frame[[f'{n}_raw' for n in models]].mean(1), frame[[f'{n}_q16' for n in models]].mean(1))
    prefix = 'np_native_precision' if args.native else 'np_precision'
    np.savez_compressed(OUT / f'{prefix}_scores.npz', **score_cache)
    frame.to_parquet(OUT / f'{prefix}_per_molecule.parquet', index=False)
    (OUT / f'{prefix}_replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f'[{time.time() - start:.0f}s] DONE NP precision replay', flush=True)


if __name__ == '__main__':
    main()
