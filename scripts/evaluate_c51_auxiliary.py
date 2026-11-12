"""Seal both complete candidates before the single new-source comparison."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from evaluate_c45_complete import point
from validation_repair import clustered_summary
OUT=ROOT/'outputs/local/c51-complete-auxiliary-20261112'
PANEL=ROOT/'outputs/local/c48-auxiliary-holdout-20261112'
FAMILY=ROOT/'outputs/local/c49-consensus-pair-20261112/complete-candidate-freeze.json'
POLICY=ROOT/'configs/c49-consensus-pair-20261112.json'


def paths():return dict(P01=OUT/'P01.csv',P16=OUT/'P16.csv',cpu=OUT/'cpu-predictions/cpu.csv',gpu=OUT/'gpu-predictions/gpu.csv')


def freeze():
    assert not (OUT/'evaluation-freeze.json').exists() and not (OUT/'evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    family=read(FAMILY);assert family['policy_sha256']==sha(POLICY)
    assert read(FAMILY.parent/'independent-used-development-audit.json')['passed_pair']
    parent=read(OUT/'complete-parent-freeze.json');assert parent['candidate_family_sha256']==sha(OUT/'candidate-family-freeze.json')==sha(FAMILY)
    assert parent['policy_sha256']==sha(POLICY)
    for name,pin in parent['files_sha256'].items():assert sha(OUT/name)==pin
    mask=OUT/'auxiliary-masked-reference/independent-mask-audit.json';assert read(mask)['forbidden_rows_retained']==0
    files=[Path(__file__),ROOT/'scripts/audit_c51_quality.py',POLICY,FAMILY,OUT/'complete-parent-freeze.json',OUT/'candidate-family-freeze.json',mask,
        OUT/'auxiliary-masked-reference/mask-manifest.json',OUT/'auxiliary-parent/independent-parent-audit.json',
        ROOT/'outputs/forks/c50-full-masked-validation-v1/collection-audit.json',ROOT/'outputs/forks/c50-full-masked-validation-v1/acquisition-fidelity-audit.json',
        ROOT/'outputs/local/c48-clean-residual-heads-20261112/independent-fitted-model-audit.json',FAMILY.parent/'independent-used-development-audit.json']
    identities={}
    for trial,entry in family['candidates'].items():
        bundle=ROOT/entry['bundle'];assert sha(bundle/'head-manifest.json')==entry['head_manifest_sha256']
        folder=ROOT/entry['path'];assert sha(folder/'notebook.ipynb')==entry['notebook_sha256'] and sha(folder/'production-pin.json')==entry['pin_sha256']
        files.extend([folder/'notebook.ipynb',folder/'production-pin.json',bundle/'head-manifest.json'])
        scores=OUT/(trial+'-predictions');receipt=read(scores/'score-receipt.json');audit=read(scores/'independent-ranking-audit.json')
        assert receipt['baseline_sha256']==sha(OUT/'P16.csv') and receipt['query_sha256']==sha(OUT/'queries.parquet')
        assert receipt['head_manifest_sha256']==entry['head_manifest_sha256'] and receipt['queries']==receipt['acquisitions']==35
        assert audit['score_receipt_sha256']==sha(scores/'score-receipt.json') and set(receipt['trials'])=={trial}
        assert not receipt['quality_labels_read'] and not receipt['feature_cache_used']
        for name,pin in receipt['files_sha256'].items():assert sha(scores/name)==pin;files.append(scores/name)
        for row in pd.read_parquet(scores/'candidate-identities.parquet').itertuples(index=False):
            if row.smiles in identities:assert identities[row.smiles]==row.key
            identities[row.smiles]=row.key
        files.extend([scores/'score-receipt.json',scores/'independent-ranking-audit.json'])
    features=module('c51_freeze_identity','c09e_features.py')
    for path in paths().values():
        frame=pd.read_csv(path,dtype=str,keep_default_na=False);assert len(frame)==frame.molecule_id.nunique()==35;files.append(path)
        for text in frame.smiles:
            for s in text.split(';'):
                if s not in identities:identities[s]=features.structure(s)[0]
    pd.DataFrame([dict(smiles=s,key=k) for s,k in identities.items()]).to_parquet(OUT/'quality-identity-map.parquet',index=False)
    files.extend([OUT/'quality-identity-map.parquet',OUT/'queries.parquet',PANEL/'validation-freeze.json'])
    write(OUT/'evaluation-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},all_predictions_frozen=True,
        source_quality_outcomes_read=False,C13_read=False,maximum_scoring_entries=2))
    print('C51 BOTH COMPLETE PREDICTIONS SEALED',flush=True)


def evaluate():
    assert not (OUT/'source-results.json').exists() and not (OUT/'evaluation-consumption.json').exists()
    frozen=read(OUT/'evaluation-freeze.json');assert frozen['source_sha256']==sha(__file__) and frozen['policy_sha256']==sha(POLICY)
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
    label_path=PANEL/'labels.csv';panel=read(PANEL/'validation-freeze.json')
    assert sha(label_path)==panel['files_sha256'][str(label_path.relative_to(ROOT))]
    # Durable marker precedes any prediction/truth matching.
    with (OUT/'evaluation-consumption.json').open('x',encoding='utf8') as stream:
        import json
        json.dump(dict(started_utc=stamp(),evaluation_freeze_sha256=sha(OUT/'evaluation-freeze.json'),labels_sha256=sha(label_path),
            candidates=['cpu','gpu'],source='entire frozen35 auxiliary panel',C13_read=False),stream,indent=2)
    frames={name:pd.read_csv(path,dtype=str,keep_default_na=False) for name,path in paths().items()}
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==35
    labels=pd.read_csv(label_path,dtype=str).set_index('molecule_id').loc[mids]
    identities=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict()
    lists={};ranks={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[identities[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in lists[name])
        ranks[name]=np.asarray([row.index(truth)+1 if truth in row else 0 for row,truth in zip(lists[name],labels.key)],np.int16)
    q=pd.read_parquet(OUT/'queries.parquet');assert len(q)==35
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy()
    assert set(modes)=={'positive'} and len(set(labels.group))==21
    groups=labels.group.to_numpy();all_mask=np.ones(35,bool)
    def summary(base,trial,mask):
        before=np.divide(1.,ranks[base],out=np.zeros(35),where=ranks[base]>0);after=np.divide(1.,ranks[trial],out=np.zeros(35),where=ranks[trial]>0)
        value=clustered_summary(before[mask],after[mask],groups[mask],10000,20261151);value['point']=point(ranks[base],ranks[trial],mask);return value
    def compare(base,trial):
        return dict(all=summary(base,trial,all_mask),source={s:summary(base,trial,labels.source.eq(s).to_numpy()) for s in sorted(set(labels.source))},
            query_mode={'positive':summary(base,trial,all_mask)},recovered_truths=int(((ranks[base]==0)&(ranks[trial]>0)).sum()),
            displaced_truths=int(((ranks[base]>0)&(ranks[trial]==0)).sum()),truth_absent_queries=int((ranks[trial]==0).sum()),
            rank1_changes=sum(a[0]!=b[0] for a,b in zip(lists[base],lists[trial])),candidate_membership_changes=sum(set(a)!=set(b) for a,b in zip(lists[base],lists[trial])))
    results={}
    for trial in ['cpu','gpu']:
        assert all(a[0]==b[0] and set(a)==set(b) for a,b in zip(lists['P16'],lists[trial]))
        complete=compare('P01',trial);incremental=compare('P16',trial)
        passed=all(v['all']['point']['nonnegative'] and all(s['point']['nonnegative'] for s in v['source'].values()) for v in [complete,incremental])
        results[trial]=dict(complete=complete,incremental=incremental,admission=dict(passed=passed,statistical_promotion=False))
    np.savez(OUT/'source-ranks.npz',**ranks)
    record=dict(completed_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),evaluation_freeze_sha256=sha(OUT/'evaluation-freeze.json'),
        consumption_sha256=sha(OUT/'evaluation-consumption.json'),queries=35,acquisitions=35,chemistry_groups=21,results=results,
        parent_overlay=compare('P01','P16'),passed_pair=all(v['admission']['passed'] for v in results.values()),
        all_queries_retained=True,absent_truths_zero=True,actual_query_modes={'positive':35},C13_read=False,
        limitation='Small positive-only panel, mostly APCI consensus exports. Not target mixed-ESI calibration; public encoder/parent pretraining exposure is not erased. No cohort/recipe revision after these outcomes.')
    write(OUT/'source-results.json',record)
    print('C51 AUXILIARY SOURCE RESULTS', {trial:dict(admission=v['admission'],all=v['complete']['all']['point'],sources={s:x['point'] for s,x in v['complete']['source'].items()}) for trial,v in results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);a=p.parse_args()
    if a.phase=='freeze':freeze()
    else:evaluate()
