"""Replay the fixed final in-list recipe using exact cached physical evidence.

This reused 120-molecule panel is not a fresh test. Candidate generation and
outside-list retrieval are not replayed. Production has no duplicate metric
identities within these lists, which is asserted before evaluating.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cached_physical import CachedFragment, physical_module
from panel_replay import leader_safe
from tail_fusion import paired_summary, reciprocal_rank, standardize
from metric_identity import unique_metric_order

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/local/professional'


def final_order(group, neural, cos, fps, query, post, frag, annotation, document):
    group = group.copy()
    group['neural_score'], group['neural_cos'], group['neural_fp'] = neural, cos, fps
    v2 = leader_safe(neural, .25, .50).tolist()
    custom = leader_safe(neural, .25, .75).tolist()
    smiles = group.smiles.tolist()
    mid = str(group.molecule_id.iloc[0])
    frag.mid = mid
    physical, diag = post.decision(mid, [smiles[i] for i in v2], group, None, query, {}, frag, {},
                                   allow_tail=False, allow_top=True)
    leader = smiles.index(physical[0])
    moved = leader != v2[0]
    if not moved:
        leader = custom[0]
    rest = [i for i in custom if i != leader]
    if document:
        nranks = np.empty(len(neural), dtype=int)
        nranks[np.argsort(-neural, kind='stable')] = np.arange(1, len(neural)+1)
        a = annotation.reindex(group.metric_key.astype(str)).fillna(0)
        npflag = (a.np_coconut.astype(bool) | a.np_lotus.astype(bool) | a.src_natural_product.astype(bool)).to_numpy()
        score = 1/(3+group.baseline_rank.to_numpy()) + .75/(3+nranks)
        score += .10*a['pop'].to_numpy()/20 + (.05 if group.library_max.max() >= .87 else 0)*a.src_spectral_library.to_numpy() + .02*npflag
        rest.sort(key=lambda i: float(score[i]), reverse=True)
        if moved and v2[0] in rest:
            rest = [v2[0]] + [i for i in rest if i != v2[0]]
    return unique_metric_order([leader]+rest, group.metric_key.tolist()), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terms', nargs='*', default=[], help='NAME=path.npz; each is blended 50/50 with original scores')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT / 'physical_recipe_replay.json')
    args = ap.parse_args()
    started = time.time()
    folder = ROOT / 'outputs/forks/p06-fragment-cache'
    audit = json.loads((folder/'fragment_cache_audit.json').read_text())
    post = physical_module(ROOT/'kaggle/forks/p06-fragment-cache/notebook.ipynb', audit['physical_source_sha256'])
    frag = CachedFragment(folder)
    post.select_spectra = frag.selected_spectra
    panel = pd.read_parquet(ROOT/'outputs/local/saner_panel/labelled_candidate_scores.parquet')
    query_files = list((ROOT/'outputs/research/raw-panel').rglob('*.parquet'))
    assert sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in query_files) == audit['query_file_sha256']
    queries = pd.concat([pd.read_parquet(p) for p in query_files])
    qmap = dict(tuple(queries.groupby('molecule_id', sort=False)))
    annotation = pd.read_parquet(ROOT/'outputs/research/pop-prior/saner_annotations.parquet').set_index('inchikey14')
    terms = {name: np.load(path, allow_pickle=False) for name,path in (s.split('=',1) for s in args.terms)}
    rows = []
    for mid,g in panel.groupby('molecule_id',sort=True):
        g = g.sort_values('baseline_rank',kind='stable').reset_index(drop=True)
        assert g.metric_key.nunique() == len(g), 'Outside-list dedup filling would need a full replay'
        public = g.neural_score.to_numpy()
        base, diag = final_order(g,public,g.neural_cos.to_numpy(),g.neural_fp.to_numpy(),qmap[mid],post,frag,annotation,False)
        row = dict(molecule_id=mid, baseline=reciprocal_rank(g.label,base), baseline_move=int(diag['top1_changed']))
        control,d = final_order(g,public,g.neural_cos.to_numpy(),g.neural_fp.to_numpy(),qmap[mid],post,frag,annotation,True)
        row['public479'] = reciprocal_rank(g.label,control)
        for name,values in terms.items():
            blend = .5*standardize(public) + .5*standardize(values[mid+'__score'])
            cos = .5*g.neural_cos.to_numpy() + .5*values[mid+'__cos']
            fps = .5*g.neural_fp.to_numpy() + .5*values[mid+'__fp']
            order,d = final_order(g,blend,cos,fps,qmap[mid],post,frag,annotation,True)
            row[name] = reciprocal_rank(g.label,order)
            row[name+'_move'] = int(d['top1_changed'])
        rows.append(row)
    frame = pd.DataFrame(rows)
    comparisons = {name:paired_summary(frame.baseline,frame[name]) for name in ['public479']+list(terms)}
    report = dict(n=len(frame),fresh_test=False,competition_mrr=False,full_pipeline=False,
                  exact_in_list_physical_merge=True,candidate_generation_replayed=False,
                  physical_source_sha256=audit['physical_source_sha256'],
                  baseline_leader_moves=int(frame.baseline_move.sum()),
                  leader_moves={name:int(frame[name+'_move'].sum()) for name in terms},
                  comparisons=comparisons,seconds=time.time()-started)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(args.out.with_suffix('.parquet'),index=False)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
