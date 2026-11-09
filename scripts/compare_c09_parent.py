"""Compare frozen C09 against already trained C08 on the same development rows."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
import json
from pathlib import Path
from compatible_tail import apply
from tail_fusion import paired_summary

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=ROOT/'outputs/local/c09-20261109';selection=json.loads((run/'development/selection.json').read_text())
    original=json.loads((ROOT/'outputs/local/long-compatible-20261108/qualification/selection.json').read_text())
    ids=selection['development_ids'];assert selection['winner']['weight']==.2
    baseline=pd.read_csv(ROOT/'outputs/forks/c08-external-validation/c08_validation_p01.csv',dtype=str,keep_default_na=False)
    baseline=baseline.set_index('molecule_id').loc[ids].reset_index()
    parent=pd.read_parquet(ROOT/'outputs/local/long-compatible-20261108/qualification/legacy16-hard.parquet')
    parent=parent[parent.molecule_id.isin(set(ids))]
    new=pd.read_parquet(run/'development/scores.parquet')
    after_parent,pc=apply(baseline,parent,original['winner']['weight'])
    after_new,nc=apply(baseline,new,selection['winner']['weight'])
    labels=pd.read_csv(ROOT/'outputs/publish/c08-validation-20261108/labels.csv').set_index('molecule_id')
    assert labels.loc[ids,'partition'].eq('development').all()
    mapping={}
    for name in ['neural_candidate_features.parquet','outsider_neural_features.parquet']:
        frame=pd.read_parquet(ROOT/'outputs/forks/c08-external-validation'/name,columns=['smiles','metric_key']).dropna()
        mapping.update(zip(frame.smiles.astype(str),frame.metric_key.astype(str)))
    def rr(frame):
        values=[]
        for row in frame.itertuples(index=False):
            truth=labels.loc[row.molecule_id,'key'];keys=[mapping[s] for s in row.smiles.split(';')]
            rank=keys.index(truth)+1 if truth in keys else None;values.append(1/rank if rank else 0.)
        return np.array(values)
    a,b=rr(after_parent),rr(after_new)
    report=dict(kind='same_development_parent_control',queries=len(ids),p01_mrr25=float(rr(baseline).mean()),
        c08_parent_mrr25=float(a.mean()),c09_mrr25=float(b.mean()),c09_vs_parent=paired_summary(a,b),
        parent_weight=original['winner']['weight'],c09_weight=selection['winner']['weight'],
        list_changes=int((after_parent.smiles!=after_new.smiles).sum()),rank_one_changes=nc['rank_one_changes'],candidate_set_changes=nc['candidate_set_changes'],
        new80_confirmation_used=False,selection_changed=False,
        limitation='Previously used development rows only. A tie does not establish continuation benefit. No fresh confirmation or hidden-score claim.')
    (run/'development/parent-control.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
