"""Choose one C08 tail experiment on DEVELOPMENT only; report reserved once.

The user explicitly authorized one trial. A failed/noisy statistical comparison
does not become a claim of promotion or a reason to invent validation gains.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse,json
from compatible_inference import score,digest,load_modules
from compatible_tail import apply
from tail_fusion import paired_summary
from precision_cache import ROOT,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation',type=Path,default=ROOT/'outputs/forks/c08-external-validation')
    parser.add_argument('--bundle',type=Path,default=ROOT/'outputs/publish/c08-models-20261108');args=parser.parse_args()
    torch.set_num_threads(2);out=ROOT/'outputs/local/long-compatible-20261108/qualification';out.mkdir(exist_ok=True)
    frozen=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else None
    campaign=json.loads((ROOT/'configs/compatible-campaign-20261108.json').read_text());proposal=campaign['candidate']
    report=json.loads((args.validation/'c08_validation_report.json').read_text());assert report['queries']==128 and report['truth_absent_mrr']==0 and not report['truth_injection']
    contract=ROOT/'outputs/publish/c08-validation-20261108/validation-contract.json';assert digest(contract)==report['validation_contract_sha256']
    for name,expected in json.loads(contract.read_text())['files'].items():
        assert digest(contract.parent/name)==expected,'Validation input bytes changed: '+name
    if frozen:
        assert frozen['manifest_sha256']==digest(args.bundle/'c08-manifest.json') and frozen['validation_report_sha256']==digest(args.validation/'c08_validation_report.json')
    baseline=pd.read_csv(args.validation/'c08_validation_p01.csv',dtype=str,keep_default_na=False)
    labels=pd.read_csv(contract.parent/'labels.csv',dtype={'molecule_id':str,'key':str}).set_index('molecule_id')
    assert set(baseline.molecule_id)==set(labels.index) and labels.partition.value_counts().to_dict()==dict(development=64,confirmation=64)
    mapping={}
    for name in ['neural_candidate_features.parquet','outsider_neural_features.parquet']:
        frame=pd.read_parquet(args.validation/name,columns=['smiles','metric_key']).dropna()
        for s,key in zip(frame.smiles.astype(str),frame.metric_key.astype(str)):
            assert s not in mapping or mapping[s]==key;mapping[s]=key
    _,features=load_modules(args.bundle);missing=0
    for text in baseline.smiles:
        for smiles in text.split(';'):
            if smiles not in mapping:mapping[smiles]=features.structure(smiles)[0];missing+=1
    def evaluate(frame):
        rows=[]
        for row in frame.itertuples(index=False):
            truth=labels.loc[row.molecule_id,'key'];keys=[mapping[s] for s in row.smiles.split(';')]
            assert len(set(keys))==len(keys)
            rank=keys.index(truth)+1 if truth in keys else None
            rows.append(dict(molecule_id=row.molecule_id,partition=labels.loc[row.molecule_id,'partition'],truth_present=rank is not None,
                             rank=rank,mrr25=1/rank if rank else 0.))
        return pd.DataFrame(rows).set_index('molecule_id').reindex(labels.index)
    before=evaluate(baseline);dev=labels.partition.eq('development').to_numpy();trials=[];scores={};shared={}
    for arm in ([frozen['winner']['arm']] if frozen else proposal['arms']):
        path=out/(arm+'.parquet')
        if path.exists():
            cached=json.loads(Path(str(path)+'.json').read_text())
            assert cached['manifest_sha256']==digest(args.bundle/'c08-manifest.json') and cached['arm']==arm
            assert cached['steps']==40000 and not cached['smoke'] and cached['input_errors']==0
            assert cached['queries_sha256']==digest(args.validation/'c08_validation_queries.parquet')
            import hashlib
            assert cached['baseline_content_sha256']==hashlib.sha256(baseline.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
            values=pd.read_parquet(path)
        else:values,_=score(args.bundle,arm,baseline,args.validation/'c08_validation_queries.parquet',path,shared_fps=shared)
        scores[arm]=values
        if frozen:continue
        for weight in proposal['weights']:
            candidate,changes=apply(baseline,values,weight);metrics=evaluate(candidate)
            trials.append(dict(arm=arm,development_mrr25=float(metrics.mrr25.to_numpy()[dev].mean()),
                               development_baseline_mrr25=float(before.mrr25.to_numpy()[dev].mean()),**changes))
    winner=frozen['winner'] if frozen else max(trials,key=lambda r:(r['development_mrr25'],-r['weight'],-proposal['arms'].index(r['arm'])))
    selection=frozen or dict(winner=winner,trials=trials,manifest_sha256=digest(args.bundle/'c08-manifest.json'),
        validation_report_sha256=digest(args.validation/'c08_validation_report.json'),development_ids=labels.index[dev].tolist(),
        reserved_ids=labels.index[~dev].tolist(),selection='Development only; weight then arm-order tiebreak',
        integration=proposal['integration'],explicit_user_trial=True,statistical_promotion=False)
    # Freeze the decision BEFORE looking at reserved metric differences.
    if not frozen:write(out/'selection.json',selection)
    candidate,changes=apply(baseline,scores[winner['arm']],winner['weight']);after=evaluate(candidate)
    confirmation=paired_summary(before.mrr25.to_numpy()[~dev],after.mrr25.to_numpy()[~dev])
    result=dict(winner=winner,development=paired_summary(before.mrr25.to_numpy()[dev],after.mrr25.to_numpy()[dev]),confirmation=confirmation,
        baseline_candidate_recall25=float(before.truth_present.mean()),truth_absent_queries=int((~before.truth_present).sum()),queries=128,
        missing_metric_keys_recomputed=missing,rank_one_changes=changes['rank_one_changes'],candidate_set_changes=changes['candidate_set_changes'],
        failed_P14_full_pipeline_negative_control='Not reproduced on this external panel; no statistical promotion claim',
        all_promotion_requirements_passed=False,explicit_user_trial=True,hidden_gain_unproven=True)
    write(out/'report.json',result);write(args.bundle/'selection.json',selection)
    before.add_prefix('p01_').join(after.add_prefix('c08_')).reset_index().to_csv(out/'validation_predictions.csv',index=False)
    candidate.to_csv(out/'validation_candidate.csv',index=False)
    print('DONE C08 DEVELOPMENT SELECTION',json.dumps(result),flush=True)


if __name__=='__main__':main()
