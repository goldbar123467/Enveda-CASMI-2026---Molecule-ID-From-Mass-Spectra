"""Select follow-up model/milestone/weight on old clean DEVELOPMENT only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from followup_models import PACKAGE,CANDIDATES,REFERENCES,score_group
from precision_cache import ROOT,sha,write
from compatible_tail import apply
from tail_fusion import paired_summary
from training_monitor import system_memory


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
    out=ROOT/'outputs/local/c09-followup-evaluation-20261109';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'selection.json').exists(),'Decision is frozen; never reselect on confirmation'
    original=json.loads((ROOT/'outputs/local/c09-20261109/development/selection.json').read_text());ids=original['development_ids']
    baseline=pd.read_csv(ROOT/'outputs/forks/c08-external-validation/c08_validation_p01.csv',dtype=str,keep_default_na=False).set_index('molecule_id').loc[ids].reset_index()
    queries=ROOT/'outputs/local/c09-20261109/development/queries.parquet'
    labels=pd.read_csv(ROOT/'outputs/publish/c08-validation-20261108/labels.csv').set_index('molecule_id').loc[ids]
    assert len(ids)==30 and labels.partition.eq('development').all()
    mapping={}
    for name in ['neural_candidate_features.parquet','outsider_neural_features.parquet']:
        frame=pd.read_parquet(ROOT/'outputs/forks/c08-external-validation'/name,columns=['smiles','metric_key']).dropna()
        for s,key in zip(frame.smiles.astype(str),frame.metric_key.astype(str)):
            assert s not in mapping or mapping[s]==key;mapping[s]=key
    def evaluate(frame):
        result=[]
        for row in frame.itertuples(index=False):
            keys=[mapping[s] for s in row.smiles.split(';')];assert len(keys)==len(set(keys))
            truth=labels.loc[row.molecule_id,'key'];rank=keys.index(truth)+1 if truth in keys else None
            result.append(1/rank if rank else 0.)
        return np.asarray(result)
    before=evaluate(baseline);shared={};trials=[];refs={}
    for name in REFERENCES+CANDIDATES:
        assert system_memory().get('free_gib',0)>=4,'Free RAM below floor'
        values,receipt=score_group(name,baseline,queries,out/(name+'.parquet'),shared,out/'bundles')
        assert receipt['input_errors']==0 and receipt['scored_molecules']==30
        # Persist model-independent molecular features for later replay.
        if shared:
            names=sorted(shared);packed=np.packbits(np.stack([shared[s] for s in names]),axis=1,bitorder='big')
            assert np.array_equal(np.unpackbits(packed,axis=1,bitorder='big'),np.stack([shared[s] for s in names]))
            pd.DataFrame(dict(smiles=names)).to_parquet(out/'cached-fingerprints.parquet',index=False)
            np.save(out/'cached-fingerprints.npy',packed)
            write(out/'cached-fingerprints.json',dict(features_sha256=sha(PACKAGE/'features.py'),names_sha256=sha(out/'cached-fingerprints.parquet'),values_sha256=sha(out/'cached-fingerprints.npy')))
        for weight in ([.2] if name in REFERENCES else [0.,.05,.1,.2]):
            candidate,changes=apply(baseline,values,weight);after=evaluate(candidate)
            row=dict(changes,group=name,mrr25=float(after.mean()),paired=paired_summary(before,after))
            if name in REFERENCES:refs[name]=row
            else:trials.append(row)
        print('FOLLOWUP DEVELOPMENT',name,json.dumps(refs.get(name,trials[-4:])),flush=True)
    winner=max(trials,key=lambda row:(row['mrr25'],-row['weight'],-CANDIDATES.index(row['group'])))
    selection=dict(campaign='C09',winner=winner,trials=trials,references=refs,development_queries=30,development_ids=ids,
        model_manifest_sha256=sha(PACKAGE/'followup-manifest.json'),queries_sha256=sha(queries),selection='Old acquisition-clean DEVELOPMENT only; smaller weight then predeclared group order on ties',
        fresh80_confirmation_used=False,old_confirmation_used=False,
        improves_p01_development=winner['mrr25']>float(before.mean()),improves_p15_development=winner['mrr25']>refs['p15-parent']['mrr25'],
        qualification=False,automatic_submission=False)
    write(out/'selection.json',selection);write(PACKAGE/'development-selection.json',selection)
    print('DONE FOLLOWUP DEVELOPMENT FREEZE',json.dumps(selection),flush=True)


if __name__=='__main__':main()
