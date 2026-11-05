# ---- CPU test of fpvote.py on the labelled replay panel (inference only, no training) ----
import glob, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(CODE))
for name in ('ranking.py', 'consensus.py'):
    (CODE / name).write_text(MODULES[name])
from ranking import stable_rank


def find(pat):
    return sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))


print('fold checkpoints', find('fpnet_fold*.pt'))
print('fp_bits', find('casmi26-fold-safe-fpnet/**/fp_bits.npy') or find('fp_bits.npy')[:3])
feat = pd.read_parquet(find('catalogue_replay_checkpoint/all_features.parquet')[0])
spec = pd.concat([pd.read_parquet(p) for p in find('catalogue_replay_checkpoint/batch_*/effective_query_spectra.parquet')])
feat.to_parquet(ROOT / 'panel_features.parquet', index=False)
spec.to_parquet(ROOT / 'panel_spectra.parquet', index=False)
r = subprocess.run([sys.executable, str(CODE / 'fpvote.py'), ROOT / 'panel_features.parquet', ROOT / 'panel_spectra.parquet',
                    ROOT / 'panel_fp.parquet', 'cpu'], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-5000:])
r.check_returncode()
fp = pd.read_parquet(ROOT / 'panel_fp.parquet')
lab = pd.read_parquet(find('catalogue_experiment/labelled_candidate_scores.parquet')[0])
label = {(str(m), s): int(l) for m, s, l in zip(lab.molecule_id, lab.smiles, lab.label)}


def mrr(order_fn):
    out = []
    for mid, g in fp.groupby('molecule_id', sort=False):
        g = g.sort_values('baseline_rank', kind='stable')
        order = order_fn(g)
        smi = g.smiles.values[order]
        hit = [i for i, s in enumerate(smi[:25], 1) if label.get((str(mid), s))]
        out.append(1 / hit[0] if hit else 0.)
    return round(float(np.mean(out)), 4)


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / max(v.std(), 1e-6)


def rrf(b_scores, w, k=3.0):
    n = len(b_scores)
    return np.argsort(-(1 / (k + np.arange(1, n + 1)) + w / (k + stable_rank(b_scores))), kind='stable')


def safe(g, lam, w_mid=.75):
    first = rrf(g.neural_score.values, .25)[0]
    mid = zs(g.neural_score.values) + lam * g.fp_vote_z.values
    rest = [i for i in rrf(mid, w_mid) if i != first]
    return np.array([first] + rest)


print('coverage', fp.fp_vote.notna().mean())
print('baseline', mrr(lambda g: np.arange(len(g))))
print('fp_vote alone', mrr(lambda g: np.argsort(-g.fp_vote_z.values, kind='stable')))
print('neural alone', mrr(lambda g: np.argsort(-g.neural_score.values, kind='stable')))
for lam in (0., .25, .5, 1.):
    print('custom 0.25/0.75 + fp lambda', lam, mrr(lambda g: safe(g, lam)))
print('corr neural vs fp', fp[['neural_score', 'fp_vote']].corr().iloc[0, 1])
