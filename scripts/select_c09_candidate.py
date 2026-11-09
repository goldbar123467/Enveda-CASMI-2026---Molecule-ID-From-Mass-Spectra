"""Freeze C09 tail weight on previously used, acquisition-clean DEVELOPMENT rows.

Never use the new80-query confirmation labels/metrics for coefficient selection.
The old confirmation rows are not reported or queried here. A zero-weight tie
retains champion and does not create another scoring submission.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import pandas as pd
import numpy as np
import importlib.util,json,hashlib
from pathlib import Path
from precision_cache import ROOT,sha,write
from compatible_tail import apply
from tail_fusion import paired_summary


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
    run=ROOT/'outputs/local/c09-20261109';out=run/'development';out.mkdir(exist_ok=True)
    frozen=out/'selection.json';assert not frozen.exists(),'Selection is frozen; do not reselect'
    bundle=ROOT/'outputs/publish/c09-models-20261109'
    module=importlib.util.spec_from_file_location('c09_serving',bundle/'c09_inference.py');scorer=importlib.util.module_from_spec(module);module.loader.exec_module(scorer)
    # Only old development IDs without base-connected acquisition groups.
    audit=pd.read_csv(run/'panel-audit/identity-acquisition-cohorts.csv')
    dev=audit[(audit.partition=='development')&~audit.group_overlaps_base].copy()
    assert len(dev)>10 and dev.molecule_id.is_unique
    baseline=pd.read_csv(ROOT/'outputs/forks/c08-external-validation/c08_validation_p01.csv',dtype=str,keep_default_na=False)
    baseline=baseline[baseline.molecule_id.isin(set(dev.molecule_id))].reset_index(drop=True)
    q=pd.read_parquet(ROOT/'outputs/forks/c08-external-validation/c08_validation_queries.parquet')
    q=q[q.molecule_id.isin(set(dev.molecule_id))];q.to_parquet(out/'queries.parquet',index=False)
    scores_path=out/'scores.parquet';receipt_path=Path(str(scores_path)+'.json')
    if scores_path.exists() and receipt_path.exists():
        receipt=json.loads(receipt_path.read_text());assert receipt['manifest_sha256']==sha(bundle/'c08-manifest.json')
        assert receipt['queries_sha256']==sha(out/'queries.parquet') and receipt['steps']==45000 and not receipt['smoke']
        assert receipt['baseline_content_sha256']==hashlib.sha256(baseline.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
        assert receipt['input_errors']==0 and receipt['scored_molecules']==len(baseline)
        scores=pd.read_parquet(scores_path)
    else:scores,receipt=scorer.score(bundle,'served-score',baseline,out/'queries.parquet',scores_path)
    truth=dev.set_index('molecule_id').key.to_dict();keys={};identity_sources={}
    # Reuse the exact official identities exported by the complete P01 pipeline,
    # already replay-audited in C08. Avoid repeated expensive tautomer searches.
    for name in ['neural_candidate_features.parquet','outsider_neural_features.parquet']:
        path=ROOT/'outputs/forks/c08-external-validation'/name;identity_sources[name]=sha(path)
        frame=pd.read_parquet(path,columns=['smiles','metric_key']).dropna()
        for smiles,key in zip(frame.smiles.astype(str),frame.metric_key.astype(str)):
            assert smiles not in keys or keys[smiles]==key;keys[smiles]=key
    for text in baseline.smiles:
        for s in text.split(';'):
            assert s in keys,'Expected byte-preserved original pipeline identity: '+s
    def evaluate(frame):
        values=[]
        for row in frame.itertuples(index=False):
            candidates=[keys[s] for s in row.smiles.split(';')]
            assert len(set(candidates))==len(candidates)
            rank=candidates.index(truth[row.molecule_id])+1 if truth[row.molecule_id] in candidates else None
            values.append(1/rank if rank else 0.)
        return np.array(values)
    before=evaluate(baseline);trials=[]
    for weight in [0.,.05,.1,.2]:
        after,changes=apply(baseline,scores,weight);values=evaluate(after)
        trials.append(dict(changes,weight=weight,mrr25=float(values.mean()),paired=paired_summary(before,values)))
    winner=max(trials,key=lambda r:(r['mrr25'],-r['weight']))
    winner=dict(winner,arm='served-score')
    selection=dict(campaign='C09',winner=winner,trials=trials,development_ids=baseline.molecule_id.tolist(),
        development_queries=len(baseline),selection='Previously used C08 development only, exclude all cached base-connected input groups; zero-weight tie retains champion',
        manifest_sha256=sha(bundle/'c08-manifest.json'),new80_confirmation_used=False,old_confirmation_used=False,
        official_identity_source_sha256=identity_sources,score_receipt_sha256=sha(receipt_path),
        qualify_for_confirmation=winner['weight']>0 and winner['mrr25']>float(before.mean()),
        promotion=False,causal_limitation='Additional updates and objective change are not separated; no hidden improvement claim')
    write(frozen,selection);write(bundle/'selection.json',selection)
    print('DONE C09 DEVELOPMENT FREEZE',json.dumps(selection),flush=True)


if __name__=='__main__':main()
