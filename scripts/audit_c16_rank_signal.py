"""Rank-signal diagnosis on already-used160 only; no new recipe selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import pyarrow as pa

from precision_cache import ROOT,sha,write


def main():
    pa.set_cpu_count(2)
    out=ROOT/'outputs/local/c16-learned-pool-20261110';target=out/'rank-signal.json';assert not target.exists()
    assert not json.loads((out/'development.json').read_text())['new_candidate_selected']
    pool=ROOT/'outputs/local/c12-deep-candidates-20261110/all'
    frame=pd.read_parquet(pool/'pool.parquet');parent=frame[frame.is_parent.astype(bool)]
    labels=pd.read_csv(ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv',dtype=str).set_index('molecule_id')
    original={mid:set(g.metric_key) for mid,g in parent.groupby('molecule_id',sort=False)}
    channels={};sources={}
    for name in ['cosine-control','C15NP']:
        path=out/(name+'-scores.parquet');score=pd.read_parquet(path)
        score=frame[['molecule_id','smiles','metric_key','prior_rank']].merge(score,on=['molecule_id','smiles'],validate='one_to_one')
        for column in ['c08_score','c08_raw_fold0','c08_raw_fold1']:channels[name+':'+column]=(score,column)
        sources[str(path.relative_to(ROOT))]=sha(path)
    path=pool/'scores.parquet';score=pd.read_parquet(path);channels['P01pair:ensemble']=(score,'neural_score');sources[str(path.relative_to(ROOT))]=sha(path)
    summaries={};rows=[]
    for name,(scores,column) in channels.items():
        values=[];outside=[];inside=[]
        for mid,g in scores.groupby('molecule_id',sort=False):
            g=g.sort_values(['prior_rank'],kind='stable');ranked=g.iloc[np.argsort(-g[column].to_numpy(),kind='stable')]
            keys=ranked.metric_key.tolist();truth=labels.loc[mid,'key'];rank=keys.index(truth)+1 if truth in keys else None
            values.append(1/rank if rank is not None and rank<=25 else 0.)
            if rank is not None:
                (inside if truth in original[mid] else outside).append(rank)
                rows.append(dict(channel=name,molecule_id=mid,truth_rank=rank,absent_parent=truth not in original[mid]))
        assert len(values)==160
        summaries[name]=dict(queries=160,direct_raw_model_mrr25=float(np.mean(values)),
            parent_truths_present=len(inside),additional_truths_present=len(outside),
            additional_truth_rank_median=float(np.median(outside)),
            additional_truths_at_or_above={str(k):int((np.array(outside)<=k).sum()) for k in [1,3,5,10,25,50,128]},
            existing_truth_rank_median=float(np.median(inside)))
    pd.DataFrame(rows).to_csv(out/'rank-signal-rows.csv',index=False)
    write(target,dict(completed_utc=datetime.now(timezone.utc).isoformat(),queries=160,
        auditor_sha256=sha(__file__),sources_sha256=sources,channels=summaries,
        all_development_previously_used=True,fresh_C13_prediction_metrics_read=False,
        model_trained=False,recipe_selected=False,automatic_submission=False,
        limitation='Oracle truth-rank diagnosis on useddevelopment; comparing individual/ensemble channels here is not model selection or independent confirmation.'))
    print('DONE C16 USED-PANEL RANK DIAGNOSIS',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
