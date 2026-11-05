# ---- n3 (= n1 with seeds 9-16, no holdout runs): extra seeds of Saner Turhaner's spectrum-structure scorer (GPU), trained from n0's prepared cache ----
# Kit: sanpier/casmi-spectrum-model-v1 (config, model.py, train.py unchanged). Cache: clarkkitchen/casmi26-n0-scorer-prep
# (same molecules/fingerprints as the public retrain; one extra spectrum out of 2.43 M).
# 1. REFIT_SEEDS: full-data refits exactly like final_model.pt (16 epochs), different seeds -> seeds/casmi26_seed_<k>.pt.
#    Not named final_model.pt and no config.json beside them, so the pipeline's find_bundle still sees one bundle.
# 2. HOLDOUT_SEEDS: train-split runs with the kit's validation; the 1000-molecule diagnostic then compares the kit's
#    holdout model alone with the average of 2 and 3 holdout models (same averaging as the submission variant).
import json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)

REFIT_SEEDS = [9, 10, 11, 12, 13, 14, 15, 16]
HOLDOUT_SEEDS = []
BASE_SEED = 20261023
W = Path('/kaggle/working')
INP = Path('/kaggle/input')

kits = [p.parent for p in INP.rglob('train.py') if (p.parent / 'final_model.pt').is_file() and (p.parent / 'model.py').is_file()]
caches = [p.parent for p in INP.rglob('peaks.bin') if (p.parent / 'fingerprints.npy').is_file()]
assert len(kits) == 1 and len(caches) == 1, (kits, caches)
KIT, CACHE = kits[0], caches[0]
log('KIT', KIT, 'CACHE', CACHE, sorted(p.name for p in CACHE.iterdir()))
sys.path.insert(0, str(KIT))
import train as T
from model import SpectrumCandidateModel

# hard_pools is deterministic in (allowed ids, count) and only read during training; build each distinct pool set
# once instead of once per run (a CPU loop over 274k molecules while the GPU idles).
_pool_cache, _hard_pools = {}, T.hard_pools
def cached_pools(mol, allowed, count):
    a = np.asarray(allowed); key = (len(a), int(a.sum()), int(a[0]), count)
    if key not in _pool_cache:
        t = time.time(); _pool_cache[key] = _hard_pools(mol, a, count); log('POOLS', key, round(time.time() - t), 's')
    return _pool_cache[key]
T.hard_pools = cached_pools

assert torch.cuda.is_available(), 'GPU required'
device = torch.device('cuda')
torch.set_num_threads(4)
cfg = json.loads((KIT / 'config.json').read_text())
cfg['work'] = str(CACHE)
data = T.Data(cfg)
t = time.time(); data.x = np.array(data.x); data.meta = np.array(data.meta); data.fp = np.array(data.fp)
log('PRELOADED', round(time.time() - t), 's', data.x.nbytes >> 20, 'MB peaks', data.fp.nbytes >> 20, 'MB fingerprints')
log('DATA', len(data.mol), 'molecules', len(data.spec), 'spectra', len(data.train), 'train', len(data.val), 'validation')
report = dict(kit=str(KIT), cache=str(CACHE), refit={}, holdout={}, diagnostic={})


def load(path):
    m = SpectrumCandidateModel(cfg).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True)['state_dict'])
    return m.eval()


@torch.inference_mode()
def validate_ens(models, candidates):
    """T.validate with each candidate's score averaged over models (pessimistic ties, gold at index 0)."""
    for m in models:
        m.eval()
    rows = []
    for i, ids in candidates.items():
        sr = data.by_mol[i]; sr = sr[data.spec.source.values[sr] == 0]
        if not len(sr):
            continue
        fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
        total = 0
        for m in models:
            zsum = 0; psum = 0; n = 0
            for start in range(0, len(sr), cfg['batch_size']):
                batch = sr[start:start + cfg['batch_size']]
                z, p = m.encode_spectra(*data.batch(batch, device)); zsum = zsum + z.sum(0); psum = psum + p.sum(0); n += len(batch)
            total = total + ((m.encode_molecules(fp) @ (zsum / n)) / cfg['temperature']
                             + cfg['fp_score_weight'] * (fp @ (psum / n)) / cfg['fp_bits'])
        scores = (total / len(models)).cpu().numpy()
        rank = 1 + int(np.sum(scores[1:] >= scores[0]))
        rows.append(dict(rank=rank, mrr25=1 / rank if rank <= 25 else 0., top1=int(rank == 1)))
    ranks = np.array([r['rank'] for r in rows])
    return dict(mrr25=float(np.mean([r['mrr25'] for r in rows])), top1=float(np.mean(ranks == 1)),
                top5=float(np.mean(ranks <= 5)), n=len(rows))


# Check the averaging code against the kit's own validate() on one model before spending GPU time on training.
candidates = {}
log('CANDIDATES', len(candidates))
kit_holdout = load(KIT / 'best_validation_model.pt')

SEEDS = W / 'seeds'; SEEDS.mkdir(exist_ok=True)
for k in REFIT_SEEDS:
    out = W / 'runs' / f'refit_{k}'; out.mkdir(parents=True, exist_ok=True)
    c = dict(cfg, seed=BASE_SEED + 1000 * k)
    t = time.time()
    res = T.train_stage(data, c, out, device, refit=True, epochs=16, deadline=time.monotonic() + 1800)
    shutil.move(str(out / 'final_model.pt'), str(SEEDS / f'casmi26_seed_{k}.pt'))
    report['refit'][k] = dict(res, seconds=round(time.time() - t), seed=c['seed'],
                              history=json.loads((out / 'refit_history.json').read_text()))
    log('REFIT_DONE', k, report['refit'][k]['seconds'], 's', res)
    (W / 'n1_report.json').write_text(json.dumps(report, indent=2, default=str))

try:
    holdout_models = [kit_holdout]
    for k in HOLDOUT_SEEDS:
        out = W / 'runs' / f'holdout_{k}'; out.mkdir(parents=True, exist_ok=True)
        c = dict(cfg, seed=BASE_SEED + 1000 * k)
        t = time.time()
        res = T.train_stage(data, c, out, device, deadline=time.monotonic() + 2400)
        report['holdout'][k] = dict(res, seconds=round(time.time() - t), seed=c['seed'],
                                    history=json.loads((out / 'training_history.json').read_text()))
        shutil.move(str(out / 'best_validation_model.pt'), str(SEEDS / f'casmi26_holdout_{k}.pt'))
        for p in out.glob('*.pt'):
            p.unlink()
        holdout_models.append(load(SEEDS / f'casmi26_holdout_{k}.pt'))
        log('HOLDOUT_DONE', k, report['holdout'][k]['seconds'], 's', res)
    diag = report['diagnostic']
    for n in range(2, len(holdout_models) + 1):
        diag[f'ensemble_{n}'] = validate_ens(holdout_models[:n], candidates)
    log('DIAGNOSTIC', json.dumps(diag))
except Exception:
    report['diagnostic_error'] = traceback.format_exc()[-3000:]
    log('DIAGNOSTIC FAILED', report['diagnostic_error'])

shutil.rmtree(W / 'runs', ignore_errors=True)
report['seconds'] = round(time.time() - T0)
(W / 'n1_report.json').write_text(json.dumps(report, indent=2, default=str))
log('N1_DONE', sorted(p.name for p in SEEDS.iterdir()))
