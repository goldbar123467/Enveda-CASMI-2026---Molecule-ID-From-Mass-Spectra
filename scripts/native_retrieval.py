"""Evaluate fixed NP holdout models on native timsTOF and untrained formula decoys.

Formula-oracle diagnostics are not competition MRR. Every target identity and
every unseen-decoy identity is excluded from the evaluated model training sets.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import local_seeds as L
from tail_fusion import paired_summary
from view_balance_replay import records_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local/professional'
MODELS = {
    'e16_s1': L.OUT / 'holdout_np/casmi26_holdout_1.pt',
    'e16_s2': L.OUT / 'holdout_np/casmi26_holdout_2.pt',
    'e32_s6': L.OUT / 'holdout_np/casmi26_holdout_6_e32.pt',
    'e64_s7': L.OUT / 'holdout_np/casmi26_holdout_7_e64.pt',
    'e64_s8': L.OUT / 'holdout_np/casmi26_holdout_8_e64.pt',
    'e64_s9': L.OUT / 'holdout_np/casmi26_holdout_9_e64.pt',
    'e128_s8': ROOT / 'outputs/forks/n5-e128-curves/seeds_out/holdout_np/casmi26_holdout_8_e128.pt',
    'e256_s10': ROOT / 'outputs/forks/n9-e256-curves/seeds_out/holdout_np/casmi26_holdout_10_e256.pt',
    'e64_w384_s12': ROOT / 'outputs/forks/n15-capacity2/seeds_out/holdout_np/casmi26_holdout_12_e64_heads8-layers4-width384.pt',
    'e128_w256_s12': ROOT / 'outputs/forks/n15-capacity2/seeds_out/holdout_np/casmi26_holdout_12_e128_heads8-layers4-width256.pt',
}


def load_holdout(cfg, path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['stage'] == 'holdout'
    settings = dict(cfg, **checkpoint.get('cfg_overrides', {}))
    model = L.SpectrumCandidateModel(settings).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), settings, checkpoint['epoch']


@torch.inference_mode()
def embedding(model, data, device):
    zsum, psum = 0, 0
    for start in range(0, len(data), 64):
        batch = data[start:start + 64]
        z, p = model.encode_spectra(torch.tensor(np.stack([r[0] for r in batch]), device=device),
            torch.tensor([r[1] for r in batch], device=device, dtype=torch.long),
            torch.tensor(np.stack([r[2] for r in batch]), device=device),
            torch.tensor([r[3] for r in batch], device=device, dtype=torch.long))
        zsum = zsum + z.sum(0)
        psum = psum + p.sum(0)
    return zsum / len(data), psum / len(data)


@torch.inference_mode()
def score(model, cfg, fp, z, p, device):
    f = torch.tensor(np.asarray(fp), device=device, dtype=torch.float32)
    return (model.encode_molecules(f) @ z / cfg['temperature'] + cfg['fp_score_weight'] * (f @ p) / cfg['fp_bits']).cpu().numpy()


def rank(score):
    return 1 + int((score[1:] >= score[0]).sum())


def summarize(ranks):
    ranks = np.asarray(ranks)
    return dict(n=len(ranks), mrr25=float(np.where(ranks <= 25, 1 / ranks, 0).mean()),
                top1=float((ranks == 1).mean()), top5=float((ranks <= 5).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unseen-dir', type=Path, default=ROOT / 'outputs/forks/p04-native-unseen')
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--checkpoint', action='append', default=[], help='NAME=holdout_np/path.pt')
    parser.add_argument('--out', type=Path, default=OUT / 'native_retrieval.json')
    args = parser.parse_args()
    start = time.time()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    cfg = json.loads((L.KIT / 'config.json').read_text())
    cfg['work'] = str(L.CACHE)
    data, device = L.T.Data(cfg), torch.device('cuda')
    raw = pd.read_parquet(OUT / 'np_raw_canonical.parquet')
    raw = raw[raw.ingest_lib.eq('enveda-np-examples') & raw.instrument_type.eq('timsTOF')]
    assert (raw.canonical_key == raw.molecule_id).all()
    queries = {str(key): records_for(group, cfg, True) for key, group in raw.groupby('molecule_id', sort=False)}
    cands = L.candidates(cfg, data, 'np')
    heldout = set((L.OUT / 'np_holdout_keys.txt').read_text().split())
    known = set(data.mol.key.astype(str))
    unseen = np.load(args.unseen_dir / 'np_unseen_fingerprints.npz', allow_pickle=False)
    unseen_keys = np.load(args.unseen_dir / 'np_unseen_candidate_keys.npz', allow_pickle=False)
    counts = {}
    for i, ids in cands.items():
        key = str(data.mol.key.iloc[i])
        assert key in heldout and ids[0] == i
        assert unseen_keys[key][0] == key
        assert all(k not in known for k in unseen_keys[key][1:])
        assert np.array_equal(unseen[key][0], data.fp[i]), 'Fingerprint contract mismatch'
        counts[key] = len(unseen[key])
    assert len(cands) == len(unseen.files) == 234
    provenance, summaries, rows, scores = {}, {}, [], {}
    custom = dict(s.split('=', 1) for s in args.checkpoint)
    models = dict(MODELS, **{name:Path(path) for name,path in custom.items()})
    selected = (args.models or list(custom)) if custom else (args.models or list(MODELS))
    for name in selected:
        path = models[name]
        assert path.is_file(), path
        assert 'holdout_np' in path.parts
        if name in custom:
            exclusion_file = path.parent.parent / 'np_holdout_keys.txt'
            assert exclusion_file.is_file(), 'Custom checkpoint requires its actual training-exclusion list'
            assert heldout <= set(exclusion_file.read_text().split()), 'Every target must be excluded across sources'
        model, settings, epoch = load_holdout(cfg, path, device)
        provenance[name] = dict(path=str(path.relative_to(ROOT)), epoch=epoch,
                                overrides={k: settings[k] for k in ['width', 'heads', 'layers']},
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        model_rows = []
        for i, ids in cands.items():
            key = str(data.mol.key.iloc[i])
            records = queries.get(key, [])
            row = dict(model=name, key=key, spectra=len(records), unseen_candidates=counts[key])
            if records:
                z, p = embedding(model, records, device)
                for kind, fp in [('familiar', data.fp[ids]), ('unseen', unseen[key])]:
                    value = score(model, settings, fp, z, p, device)
                    row[f'{kind}_rank'] = rank(value)
                    scores[f'{key}__{name}__{kind}'] = value
            else:
                row['familiar_rank'] = row['unseen_rank'] = 10**6
            model_rows.append(row)
        frame = pd.DataFrame(model_rows)
        summaries[name] = dict(familiar=summarize(frame.familiar_rank), unseen=summarize(frame.unseen_rank),
                               unseen_ambiguous8=summarize(frame.loc[frame.unseen_candidates >= 8, 'unseen_rank']))
        rows.extend(model_rows)
        print('MODEL', name, json.dumps(summaries[name]), round(time.time() - start), 's', flush=True)
        del model
        torch.cuda.empty_cache()
    output = dict(protocol='Native Enveda timsTOF NP queries; familiar mass-window diagnostic and oracle-formula untrained COCONUT diagnostic',
                  competition_mrr=False, fresh_targets=False, n=234, spectra=sum(map(len, queries.values())),
                  candidate_count_quantiles={str(q):float(np.quantile(list(counts.values()),q)) for q in [0,.25,.5,.75,1]},
                  provenance=provenance, models=summaries, seconds=time.time() - start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out.with_suffix('.parquet'), index=False)
    np.savez_compressed(args.out.with_suffix('.npz'), **scores)
    args.out.write_text(json.dumps(output, indent=2))
    print(f'[{time.time() - start:.0f}s] DONE native retrieval model audit', flush=True)


if __name__ == '__main__':
    main()
