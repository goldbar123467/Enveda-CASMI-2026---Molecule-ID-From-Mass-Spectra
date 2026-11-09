"""Read-only identity/acquisition/source audit of the already-used C08 panel."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import json,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2);pa.set_cpu_count(2)
    base=pd.read_parquet(ROOT/'outputs/local/n0/cache/molecules.parquet',columns=['key'])
    external=pd.read_parquet(ROOT/'outputs/external/cache-20261106/molecules.parquet',columns=['key'])
    combined=pd.concat([base,external[~external.key.isin(set(base.key))]],ignore_index=True)
    groups=np.load(ROOT/'outputs/local/long-20261106/spectrum_groups.npz',allow_pickle=False)['groups']
    assert len(groups)==len(combined) and combined.key.is_unique
    combined['group']=groups;lookup=combined.set_index('key').group
    labels=pd.read_csv(ROOT/'outputs/publish/c08-validation-20261108/labels.csv',dtype={'key':str,'molecule_id':str})
    labels['group']=labels.key.map(lookup);assert labels.group.notna().all()
    labels['group_overlaps_base']=labels.group.isin(set(groups[:len(base)]))
    nf=pd.read_parquet(ROOT/'outputs/forks/c08-external-validation/neural_candidate_features.parquet')
    metrics=pd.read_csv(ROOT/'outputs/forks/c08-external-validation/c08_validation_p01_metrics.csv')
    labels['maximum_library_similarity']=labels.molecule_id.map(nf.groupby('molecule_id').library_max.max())
    labels=labels.merge(metrics[['molecule_id','truth_present','mrr25']],on='molecule_id',validate='one_to_one')
    query=pd.read_parquet(ROOT/'outputs/forks/c08-external-validation/c08_validation_queries.parquet')
    instruments=query.groupby('molecule_id').instrument_type.agg(lambda x:';'.join(sorted(set(x.astype(str)))))
    labels['query_instruments']=labels.molecule_id.map(instruments)
    out=ROOT/'outputs/local/c09-20261109/panel-audit';out.mkdir(parents=True,exist_ok=True)
    labels.to_csv(out/'identity-acquisition-cohorts.csv',index=False)
    report=dict(queries=len(labels),identity_overlap_with_base=int(labels.key.isin(set(base.key)).sum()),
        cached_model_input_group_overlap_with_base=int(labels.group_overlaps_base.sum()),
        library_similarity_at_least_999=int((labels.maximum_library_similarity>=.999).sum()),
        high_library_truths_present=int(((labels.maximum_library_similarity>=.999)&labels.truth_present).sum()),
        query_instruments=query.instrument_type.value_counts().to_dict(),
        by_source=labels.groupby('source').agg(queries=('key','size'),truths_present=('truth_present','sum'),mrr25=('mrr25','mean'),group_overlaps=('group_overlaps_base','sum')).to_dict('index'),
        group_artifact_sha256=hashlib.sha256((ROOT/'outputs/local/long-20261106/spectrum_groups.npz').read_bytes()).hexdigest(),
        scope='C08 panel already used; no model/weight selection. Cached input groups audit quantized96-peak model inputs including metadata, not every reference-library preprocessing representation.',
        limitation='High library similarity is a warning requiring reference-spectrum audit, not by itself proof of an identical query or label leakage. Public encoders/rankers have not universally excluded this panel.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
