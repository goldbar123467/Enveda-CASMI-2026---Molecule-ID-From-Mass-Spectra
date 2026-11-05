# ---- Offline replay of the fragment check's rank-1 rule on Saner Turhaner's labelled panel (CPU, no training) ----
import glob, importlib.util, json, math, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CODE))
for name in ('ranking.py', 'consensus.py'):
    (CODE / name).write_text(MODULES[name])
from consensus import leader_safe_order
import frag_rescore as frag
POST = (CODE / 'parallax_physical.py').read_text()


def find(pat):
    hits = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    assert hits, pat
    return hits


feat = pd.read_parquet(find('catalogue_replay_checkpoint/all_features.parquet')[0])
lab = pd.read_parquet(find('catalogue_experiment/labelled_candidate_scores.parquet')[0])
spec = pd.concat([pd.read_parquet(p) for p in find('catalogue_replay_checkpoint/batch_*/effective_query_spectra.parquet')])
spec['molecule_id'] = spec.molecule_id.astype(str)
print('features', feat.shape, 'labels', lab.shape, 'spectra', spec.shape, spec.molecule_id.nunique())
label = {(str(m), s): int(l) for m, s, l in zip(lab.molecule_id, lab.smiles, lab.label)}
role = lab.drop_duplicates('molecule_id').set_index('molecule_id').role.to_dict()

GATE = 'lib<.87 and len(spec)>=3'
CONFIGS = {
    'base': [],
    'lib092': [(GATE, 'lib<.92 and len(spec)>=3')],
    'lib095': [(GATE, 'lib<.95 and len(spec)>=3')],
    'nolib': [(GATE, 'lib<9 and len(spec)>=3')],
    'spec2': [(GATE, 'lib<.87 and len(spec)>=2')],
    'win8': [('control[1:min(6,n)]', 'control[1:min(8,n)]')],
    'nm015': [('nmargin<.23', 'nmargin<.15')],
    'nm000': [('nmargin<.23', 'nmargin<0')],
    'fm007': [('if margin<.095 or oldmargin<.028', 'if margin<.07 or oldmargin<.02')],
    'votes060': [('if votes<.70', 'if votes<.60')],
    'support0': [('if supports<1 and simulator_vote<1:continue', 'pass')],
    'loose': [(GATE, 'lib<.92 and len(spec)>=2'), ('control[1:min(6,n)]', 'control[1:min(8,n)]'),
              ('nmargin<.23', 'nmargin<.15')],
}


def module_for(name, edits):
    src = POST
    for old, new in edits:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    path = CODE / f'pp_{name}.py'
    path.write_text(src)
    spec_ = importlib.util.spec_from_file_location(f'pp_{name}', path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def rr(order, mid):
    for i, s in enumerate(order[:25], 1):
        if label.get((mid, s)):
            return 1.0 / i
    return 0.0


frag.warmup()
groups = {str(m): g.sort_values('baseline_rank', kind='stable') for m, g in feat.groupby('molecule_id', sort=False)}
qmap = {m: g for m, g in spec.groupby('molecule_id', sort=False)}
orders = {}
for mid, g in groups.items():
    base = g.smiles.tolist()
    ok = bool(g.processed.all()) and bool(g.neural_ok.all())
    v2 = [base[i] for i in leader_safe_order(g.neural_score.values, .25, .50)] if ok else base
    cu = [base[i] for i in leader_safe_order(g.neural_score.values, .25, .75)] if ok else base
    orders[mid] = (base, v2, cu)
print('molecules', len(groups), 'with spectra', sum(m in qmap for m in groups))
summary, cache = {}, {}
for name, edits in CONFIGS.items():
    mod = module_for(name, edits)
    t0, rows = time.time(), []
    for mid, g in groups.items():
        base, v2, cu = orders[mid]
        if mid not in qmap:
            continue
        g2 = g.copy()
        if 'metric_key' not in g2:
            g2['metric_key'] = None
        new, diag = mod.decision(mid, v2, g2, None, qmap[mid], {}, frag, cache, allow_tail=False, allow_top=True)
        moved = new[0] != v2[0]
        merged = ([new[0], v2[0]] + [s for s in cu if s not in (new[0], v2[0])]) if moved else cu
        rows.append(dict(mid=mid, role=role.get(mid), moved=moved, good=moved and bool(label.get((mid, new[0]))),
                         bad=moved and bool(label.get((mid, v2[0]))), rr_v2=rr(v2, mid), rr_cu=rr(cu, mid),
                         rr_merged=rr(merged, mid), reason=diag.get('reason'), n_spec=diag.get('n_spectra'),
                         lib=float(g.library_max.iloc[0])))
    d = pd.DataFrame(rows)
    summary[name] = dict(n=len(d), moves=int(d.moved.sum()), good=int(d.good.sum()), bad=int(d.bad.sum()),
                         neutral=int((d.moved & ~d.good & ~d.bad).sum()), mrr_v2=round(d.rr_v2.mean(), 4),
                         mrr_custom=round(d.rr_cu.mean(), 4), mrr_merged=round(d.rr_merged.mean(), 4),
                         seconds=round(time.time() - t0, 1))
    print(name, summary[name], flush=True)
    if name in ('base', 'loose', 'nolib'):
        print(d[d.moved].to_string()[:3000])
        print(d.reason.value_counts().to_string())
ceil = 0
for mid, g in groups.items():
    base, v2, cu = orders[mid]
    if not label.get((mid, v2[0])) and any(label.get((mid, s)) for s in v2[1:8]):
        ceil += 1
print('molecules with wrong V2 leader and the answer at V2 ranks 2-8:', ceil)
print('library_max quantiles', feat.drop_duplicates('molecule_id').library_max.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
json.dump(summary, open(ROOT / 'replay_summary.json', 'w'), indent=2)
