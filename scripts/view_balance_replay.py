"""Two fixed acquisition-view hypotheses on exact raw queries, RTX 2070.

No fitted weights. The public channel stays unchanged on the 120-molecule panel;
only the second model's aggregation changes. NP models exclude all 234 identities.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import local_seeds as L
from features import peaks, metadata
from professional_replay import rank
from tail_fusion import paired_summary, reciprocal_rank, standardize
from view_aggregation import aggregate_views

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local/professional'


def records_for(group, cfg, dedup=False):
    records, seen = [], set()
    for row in group.itertuples(index=False):
        x, length = peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, cfg['max_peaks'])
        if length < cfg['min_peaks']:
            continue
        meta, adduct = metadata(row.precursor_mz, row.adduct, row.ionization_mode, row.collision_energy_ev)
        if dedup:
            digest = hashlib.sha256(x.astype('<f2').tobytes() + meta.astype('<f4').tobytes() + bytes([adduct])).digest()
            if digest in seen:
                continue
            seen.add(digest)
        records.append((x, length, meta, adduct))
    return records


@torch.inference_mode()
def terms(model, cfg, records, fp, device):
    m = model.encode_molecules(fp)
    parts = []
    for start in range(0, len(records), 64):
        b = records[start:start + 64]
        z, p = model.encode_spectra(torch.tensor(np.stack([r[0] for r in b]), device=device),
            torch.tensor([r[1] for r in b], device=device, dtype=torch.long),
            torch.tensor(np.stack([r[2] for r in b]), device=device),
            torch.tensor([r[3] for r in b], device=device, dtype=torch.long))
        parts.append((z @ m.T / cfg['temperature'] + cfg['fp_score_weight'] * (p @ fp.T) / cfg['fp_bits']).cpu().numpy())
    matrix = np.concatenate(parts)
    return dict(uniform=aggregate_views(matrix),
                adduct=aggregate_views(matrix, [r[3] for r in records]),
                polarity=aggregate_views(matrix, [r[2][3] for r in records]))


def rr(score):
    r = 1 + int((score[1:] >= score[0]).sum())
    return 1 / r if r <= 25 else 0.


def main():
    start = time.time()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    cfg = json.loads((L.KIT / 'config.json').read_text())
    cfg['work'] = str(L.CACHE)
    data, device = L.T.Data(cfg), torch.device('cuda')
    panel = pd.read_parquet(ROOT / 'outputs/local/saner_panel/labelled_candidate_scores.parquet')
    fps = np.load(ROOT / 'outputs/local/saner_panel/panel_fps.npz', allow_pickle=False)
    fpof = dict(zip(panel.smiles.unique(), fps['fp']))
    assert np.array_equal(panel.drop_duplicates('smiles').metric_key.astype(str), fps['keys'])
    q = pd.concat([pd.read_parquet(p) for p in (ROOT / 'outputs/research/raw-panel').rglob('*.parquet')])
    queryof = dict(tuple(q.groupby('molecule_id', sort=False)))
    annotation = pd.read_parquet(ROOT / 'outputs/research/pop-prior/saner_annotations.parquet').set_index('inchikey14')
    modelroot = ROOT / 'outputs/research/v80-output/casmi_spectrum_model_ours_v80'
    models = {name: L.load(cfg, modelroot / f'{name}_validation_model.pt', device) for name in ['best', 'last']}
    rows = []
    for mid, group in panel.groupby('molecule_id', sort=True):
        group = group.sort_values('baseline_rank', kind='stable').reset_index(drop=True)
        records = records_for(queryof[mid], cfg)
        assert len(records) == int(group.n_spectra.iloc[0])
        fp = torch.tensor(np.stack([fpof[s] for s in group.smiles]), device=device, dtype=torch.float32)
        public = group.neural_score.to_numpy()
        row = dict(key=mid)
        for name, model in models.items():
            for mode, score in terms(model, cfg, records, fp, device).items():
                blend = .5 * standardize(public) + .5 * standardize(score)
                row[f'{name}_{mode}'] = reciprocal_rank(group.label, rank(public, blend, group, annotation, True, False))
        rows.append(row)
    panel_frame = pd.DataFrame(rows)
    del models
    torch.cuda.empty_cache()
    np_raw = pd.read_parquet(OUT / 'np_raw_canonical.parquet')
    assert (np_raw.canonical_key == np_raw.molecule_id).all()
    np_queries = dict(tuple(np_raw.groupby('molecule_id', sort=False)))
    models = {str(seed): L.load(cfg, L.OUT / f'holdout_np/casmi26_holdout_{seed}_e64.pt', device) for seed in [7, 8, 9]}
    candidates = L.candidates(cfg, data, 'np')
    heldout = set((L.OUT / 'np_holdout_keys.txt').read_text().split())
    rows, total_spectra = [], 0
    for index, (i, ids) in enumerate(candidates.items()):
        key = str(data.mol.key.iloc[i])
        assert key in heldout and ids[0] == i
        records = records_for(np_queries[key], cfg, True)
        assert len(records) == len(data.by_mol[i])
        total_spectra += len(records)
        fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
        row, ensemble = dict(key=key), {mode: [] for mode in ['uniform', 'adduct', 'polarity']}
        for name, model in models.items():
            for mode, score in terms(model, cfg, records, fp, device).items():
                row[f'{name}_{mode}'] = rr(score)
                ensemble[mode].append(standardize(score))
        for mode, scores in ensemble.items():
            row[f'ensemble_{mode}'] = rr(np.mean(scores, axis=0))
        rows.append(row)
        if (index + 1) % 50 == 0:
            print('NP SCORED', index + 1, round(time.time() - start), 's', flush=True)
    np_frame = pd.DataFrame(rows)
    assert len(panel_frame) == 120 and len(np_frame) == 234
    report = dict(device=torch.cuda.get_device_name(), np_spectra=total_spectra,
                  hypotheses=['equal-adduct', 'equal-polarity'], fitted_parameters=0,
                  fresh_test=False, panel={}, natural_products={}, seconds=time.time() - start)
    for mode in ['adduct', 'polarity']:
        report['panel'][mode] = {name: paired_summary(panel_frame[f'{name}_uniform'], panel_frame[f'{name}_{mode}']) for name in ['best', 'last']}
        report['natural_products'][mode] = {name: paired_summary(np_frame[f'{name}_uniform'], np_frame[f'{name}_{mode}']) for name in ['7', '8', '9', 'ensemble']}
    panel_frame.to_parquet(OUT / 'view_panel_per_molecule.parquet', index=False)
    np_frame.to_parquet(OUT / 'view_np_per_molecule.parquet', index=False)
    (OUT / 'view_balance_replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f'[{time.time() - start:.0f}s] DONE acquisition view replay', flush=True)


if __name__ == '__main__':
    main()
