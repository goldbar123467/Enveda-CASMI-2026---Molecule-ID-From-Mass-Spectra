"""Fixed public-479/480 recipe regression checks on the RTX 2070.

The 120-molecule panel has informed earlier public work: this is a regression panel,
not a fresh test. Uses holdout checkpoints only. Full fragment merge is checked on Kaggle.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import local_seeds as L
from panel_replay import per_model_terms
from tail_fusion import baseline_order, paired_summary, reciprocal_rank, rrf_order, standardize
from features import peaks, metadata
from metric_identity import unique_metric_order

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/local/professional"
V80 = ROOT / "outputs/research/v80-output/casmi_spectrum_model_ours_v80"


def rank(public, neural, group, annotation, document=False, lock=False):
    leader = int(baseline_order(public)[0] if lock else rrf_order(neural, .25)[0])
    nranks = np.empty(len(neural), dtype=int)
    nranks[np.argsort(-neural, kind="stable")] = np.arange(1, len(neural) + 1)
    score = 1 / (3 + group.baseline_rank.to_numpy()) + .75 / (3 + nranks)
    if document:
        a = annotation.reindex(group.metric_key.astype(str)).fillna(0)
        lib_hit = float(group.library_max.max()) >= .87
        npflag = (a.np_coconut.astype(bool) | a.np_lotus.astype(bool) | a.src_natural_product.astype(bool)).to_numpy()
        score += .10 * a['pop'].to_numpy() / 20 + (.05 if lib_hit else 0) * a.src_spectral_library.to_numpy() + .02 * npflag
    tail = [int(i) for i in np.argsort(-score, kind="stable") if i != leader]
    # Metric identity deduplication cannot push the first instance of a correct identity down.
    return np.array(unique_metric_order([leader] + tail, group.metric_key.tolist()))


@torch.inference_mode()
def raw_scores(model, cfg, records, fp, device, quantize=False):
    molecule = model.encode_molecules(fp)
    scores = []
    for start in range(0, len(records), 64):
        b = records[start:start + 64]
        x = np.stack([r[0] for r in b])
        if quantize:
            x = x.astype(np.float16).astype(np.float32)
        z, p = model.encode_spectra(torch.tensor(x, device=device),
            torch.tensor([r[1] for r in b], device=device, dtype=torch.long),
            torch.tensor(np.stack([r[2] for r in b]), device=device),
            torch.tensor([r[3] for r in b], device=device, dtype=torch.long))
        scores.append((z @ molecule.T / cfg['temperature'] + cfg['fp_score_weight'] * (p @ fp.T) / cfg['fp_bits']).cpu().numpy())
    return np.concatenate(scores).mean(0)


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    cfg = json.loads((L.KIT / "config.json").read_text())
    cfg['work'] = str(L.CACHE)
    data, device = L.T.Data(cfg), torch.device('cuda')
    panel = pd.read_parquet(ROOT / "outputs/local/saner_panel/labelled_candidate_scores.parquet")
    queries = pd.concat([pd.read_parquet(p) for p in (ROOT / 'outputs/research/raw-panel').rglob('*.parquet')])
    queryof = dict(tuple(queries.groupby('molecule_id', sort=False)))
    fps = np.load(ROOT / "outputs/local/saner_panel/panel_fps.npz", allow_pickle=False)
    assert np.array_equal(panel.drop_duplicates('smiles').metric_key.astype(str), fps['keys'])
    fpof = dict(zip(panel.smiles.unique(), fps['fp']))
    annotation = pd.read_parquet(ROOT / "outputs/research/pop-prior/saner_annotations.parquet").set_index('inchikey14')
    idxof = dict(zip(data.mol.key, range(len(data.mol))))
    checkpoints = {'kit': L.KIT / 'best_validation_model.pt',
                   'v80_best': V80 / 'best_validation_model.pt', 'v80_last': V80 / 'last_validation_model.pt'}
    for seed in [202611051, 202611052]:
        checkpoints[f'candidate_{seed}'] = ROOT / f'outputs/local/science-20261105/round-{seed}/holdout.pt'
    models, provenance = {}, {}
    for name, path in checkpoints.items():
        ck = torch.load(path, map_location='cpu', weights_only=True)
        assert ck['stage'] == 'holdout', 'Full-data checkpoint is forbidden on its training panel'
        models[name] = L.load(cfg, path, device)
        provenance[name] = dict(path=str(path.relative_to(ROOT)), epoch=ck['epoch'], stage=ck['stage'])
    all_rows, score_cache = [], {}
    for mid, group in panel.groupby('molecule_id', sort=True):
        group = group.sort_values('baseline_rank', kind='stable').reset_index(drop=True)
        idx = idxof[mid.replace('holdout_', '')]
        assert data.mol.split.iloc[idx] == 'validation'
        rows = data.by_mol[idx]
        rows = rows[data.spec.source.values[rows] == 0]
        fp = torch.tensor(np.stack([fpof[s] for s in group.smiles]), device=device, dtype=torch.float32)
        records = []
        for row in queryof[mid].itertuples(index=False):
            x, length = peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, cfg['max_peaks'])
            if length < cfg['min_peaks']:
                continue
            meta, adduct = metadata(row.precursor_mz, row.adduct, row.ionization_mode, row.collision_energy_ev)
            records.append((x, length, meta, adduct))
        assert len(records) == int(group.n_spectra.iloc[0]), (mid, len(records), group.n_spectra.iloc[0])
        scores = {}
        for name, model in models.items():
            for mode in ['raw', 'q16']:
                scores[f'{name}_{mode}'] = raw_scores(model, cfg, records, fp, device, mode == 'q16')
                score_cache[f'{mid}__{name}__{mode}'] = scores[f'{name}_{mode}']
        public = group.neural_score.to_numpy()
        row = dict(molecule_id=mid, baseline=reciprocal_rank(group.label, baseline_order(public)))
        row['public479'] = reciprocal_rank(group.label, rank(public, public, group, annotation, document=True, lock=True))
        row['kit_score_max_error'] = float(np.max(np.abs(public - scores['kit_raw'])))
        row['kit_q16_score_error'] = float(np.max(np.abs(public - scores['kit_q16'])))
        for name in scores:
            if name == 'kit_raw':
                continue
            blended = .5 * standardize(public) + .5 * standardize(scores[name])
            for document in [False, True]:
                for lock in [False, True]:
                    tag = f'{name}_doc{int(document)}_lock{int(lock)}'
                    order = rank(public, blended, group, annotation, document, lock)
                    row[tag] = reciprocal_rank(group.label, order)
        all_rows.append(row)
    frame = pd.DataFrame(all_rows)
    np.savez_compressed(OUT / 'precision_score_cache.npz', **score_cache)
    assert len(frame) == 120
    frame.to_parquet(OUT / 'public_recipe_per_molecule.parquet', index=False)
    print('PUBLIC SCORE REPRODUCTION ERROR', frame.kit_score_max_error.describe().to_dict(), flush=True)
    assert frame.kit_score_max_error.max() < 1e-4, 'Public score reproduction failed'
    report = dict(device=torch.cuda.get_device_name(), threads=2, checkpoint_provenance=provenance,
                  n=120, full_pipeline=False, fresh_test=False, fixed_weights_from_public_recipe=True,
                  annotation_coverage=len(annotation), candidate_identities=panel.metric_key.nunique(),
                  kit_score_max_error=float(frame.kit_score_max_error.max()),
                  kit_q16_mean_max_error=float(frame.kit_q16_score_error.mean()), comparison={})
    for column in frame:
        if column not in ['molecule_id', 'baseline', 'kit_score_max_error', 'kit_q16_score_error']:
            report['comparison'][column] = paired_summary(frame.baseline, frame[column])
    report['seconds'] = time.time() - start
    frame.to_parquet(OUT / 'public_recipe_per_molecule.parquet', index=False)
    (OUT / 'public_recipe_replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f'[{time.time() - start:.0f}s] DONE public recipe GPU regression', flush=True)


if __name__ == '__main__':
    main()
