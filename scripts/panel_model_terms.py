"""Exact raw-query neural terms for fixed holdout checkpoints, on one local GPU."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import local_seeds as L
from native_retrieval import embedding, load_holdout
from view_balance_replay import records_for

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--heldout-keys', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    panel = pd.read_parquet(ROOT / 'outputs/local/saner_panel/labelled_candidate_scores.parquet')
    keys = set(panel.molecule_id.str.replace('holdout_', '', regex=False))
    mol = pd.read_parquet(L.CACHE / 'molecules.parquet')
    excluded = (set(args.heldout_keys.read_text().split()) if args.heldout_keys
                else set(mol.loc[mol.split.eq('validation'), 'key']))
    assert keys <= excluded, 'Every panel identity must be excluded from every training source'
    cfg = json.loads((L.KIT / 'config.json').read_text())
    model, cfg, epoch = load_holdout(cfg, args.checkpoint, torch.device('cuda'))
    queries = pd.concat([pd.read_parquet(p) for p in (ROOT / 'outputs/research/raw-panel').rglob('*.parquet')])
    qmap = dict(tuple(queries.groupby('molecule_id', sort=False)))
    stored = np.load(ROOT / 'outputs/local/saner_panel/panel_fps.npz', allow_pickle=False)
    assert np.array_equal(panel.drop_duplicates('smiles').metric_key.astype(str), stored['keys'])
    fpof = dict(zip(panel.smiles.unique(), stored['fp']))
    arrays = {}
    for mid, group in panel.groupby('molecule_id', sort=True):
        group = group.sort_values('baseline_rank', kind='stable')
        records = records_for(qmap[mid], cfg, False)
        assert len(records) == int(group.n_spectra.iloc[0])
        z, p = embedding(model, records, torch.device('cuda'))
        fp = torch.tensor(np.stack([fpof[s] for s in group.smiles]), device='cuda', dtype=torch.float32)
        cos = (model.encode_molecules(fp) @ z).cpu().numpy()
        fps = (fp @ p / cfg['fp_bits']).cpu().numpy()
        arrays[mid + '__cos'] = cos
        arrays[mid + '__fp'] = fps
        arrays[mid + '__score'] = cos / cfg['temperature'] + cfg['fp_score_weight'] * fps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    audit = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 stage='holdout', epoch=epoch, panel_molecules=len(keys), excluded_identities=len(excluded),
                 training_provenance=model.training_provenance,
                 heldout_identity_sha256=hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
                 raw_queries=True, gpu=torch.cuda.get_device_name(), seconds=time.time()-started)
    args.out.with_suffix('.json').write_text(json.dumps(audit, indent=2))
    print(f'[{time.time()-started:.0f}s] DONE PANEL TERMS', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
