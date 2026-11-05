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


def final_order(group, neural, cos, fps, query, post, frag, annotation, document, policy='479'):
    assert policy in ('479','481')
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
        spec_weight, np_weight, lib_gate = (.10,.12,.78) if policy == '481' else (.05,.02,.87)
        score += .10*a['pop'].to_numpy()/20 + (spec_weight if group.library_max.max() >= lib_gate else 0)*a.src_spectral_library.to_numpy() + np_weight*npflag
        rest.sort(key=lambda i: float(score[i]), reverse=True)
        promoted = False
        if policy == '481' and not moved and rest and group.library_max.max() < .80:
            runner = rest[0]
            if nranks[runner] < nranks[leader] and npflag[runner] and not npflag[leader] and score[runner] > score[leader]+.055:
                old = leader
                leader = runner
                rest = [old]+[i for i in rest if i != runner]
                promoted = True
        diag['np_promoted'] = promoted
        if moved and v2[0] in rest:
            rest = [v2[0]] + [i for i in rest if i != v2[0]]
    return unique_metric_order([leader]+rest, group.metric_key.tolist()), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terms', nargs='*', default=[], help='NAME=path.npz; each is blended 50/50 with original scores')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT / 'physical_recipe_replay.json')
    ap.add_argument('--public481', action='store_true', help='Also test fixed current public weights, LOTUS membership and NP promotion')
    ap.add_argument('--pair', action='append', default=[], help='NAME1,NAME2: exact .50 public + .25 each per-query z-score blend')
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
    variants = {name:[name] for name in terms}
    variants.update({pair.replace(',','_plus_'):pair.split(',') for pair in args.pair})
    assert all(len(names) in (1,2) and set(names)<=terms.keys() for names in variants.values())
    annotations = {'479':annotation}
    if args.public481:
        lotus = set(pd.read_parquet(ROOT/'outputs/research/lotus/candidates_lotus.parquet',columns=['inchikey14']).inchikey14.astype(str))
        augmented = annotation.reindex(sorted(set(panel.metric_key.astype(str))|set(annotation.index))).fillna(0)
        augmented['np_lotus'] = augmented.np_lotus.astype(bool) | augmented.index.isin(lotus)
        annotations['481'] = augmented
    rows = []
    for mid,g in panel.groupby('molecule_id',sort=True):
        g = g.sort_values('baseline_rank',kind='stable').reset_index(drop=True)
        assert g.metric_key.nunique() == len(g), 'Outside-list dedup filling would need a full replay'
        public = g.neural_score.to_numpy()
        base, diag = final_order(g,public,g.neural_cos.to_numpy(),g.neural_fp.to_numpy(),qmap[mid],post,frag,annotation,False)
        row = dict(molecule_id=mid, baseline=reciprocal_rank(g.label,base), baseline_move=int(diag['top1_changed']))
        control,d = final_order(g,public,g.neural_cos.to_numpy(),g.neural_fp.to_numpy(),qmap[mid],post,frag,annotation,True)
        row['public479'] = reciprocal_rank(g.label,control)
        if args.public481:
            order,d = final_order(g,public,g.neural_cos.to_numpy(),g.neural_fp.to_numpy(),qmap[mid],post,frag,annotations['481'],True,'481')
            row['public481_control'] = reciprocal_rank(g.label,order)
            row['public481_control_promotion'] = int(d.get('np_promoted',False))
        for name,names in variants.items():
            weight = .5/len(names)
            blend = .5*standardize(public) + sum(weight*standardize(terms[n][mid+'__score']) for n in names)
            cos = .5*g.neural_cos.to_numpy() + sum(weight*terms[n][mid+'__cos'] for n in names)
            fps = .5*g.neural_fp.to_numpy() + sum(weight*terms[n][mid+'__fp'] for n in names)
            for policy,a in annotations.items():
                tag = name if policy == '479' else name+'_public481'
                order,d = final_order(g,blend,cos,fps,qmap[mid],post,frag,a,True,policy)
                row[tag] = reciprocal_rank(g.label,order)
                row[tag+'_move'] = int(d['top1_changed'])
                row[tag+'_promotion'] = int(d.get('np_promoted',False))
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = [c for c in frame if c not in ('molecule_id','baseline') and not c.endswith(('_move','_promotion'))]
    comparisons = {name:paired_summary(frame.baseline,frame[name]) for name in columns}
    report = dict(n=len(frame),fresh_test=False,competition_mrr=False,full_pipeline=False,
                  exact_in_list_physical_merge=True,candidate_generation_replayed=False,
                  physical_source_sha256=audit['physical_source_sha256'],
                  baseline_leader_moves=int(frame.baseline_move.sum()),
                  leader_moves={c.removesuffix('_move'):int(frame[c].sum()) for c in frame if c.endswith('_move')},
                  np_promotions={c.removesuffix('_promotion'):int(frame[c].sum()) for c in frame if c.endswith('_promotion')},
                  comparisons=comparisons,seconds=time.time()-started)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(args.out.with_suffix('.parquet'),index=False)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
