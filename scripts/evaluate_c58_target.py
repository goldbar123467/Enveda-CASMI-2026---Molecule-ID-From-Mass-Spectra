"""Freeze complete predictions; consume the entire new C53 panel once.

Evaluation requires BOTH actual production audits and all387USED guards first.
No cohort/recipe selection or C13 evaluation is implemented here.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from evaluate_c45_complete import point
from validation_repair import clustered_summary
OUT=ROOT/'outputs/local/c58-complete-target-20261112'
PANEL=ROOT/'outputs/local/c53-target-holdout-20261112'
MODELS=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
POLICY=ROOT/'configs/c56-bounded-semantic-pair-20261112.json'


def paths():return dict(P01=OUT/'P01.csv',P16=OUT/'P16.csv',cpu=OUT/'head-predictions/cpu.csv',gpu=OUT/'head-predictions/gpu.csv')


def verify_production(family):
    files=[]
    for trial,entry in family['candidates'].items():
        folder=ROOT/'outputs/forks'/(__import__('pathlib').Path(entry['path']).name+'-v1');path=folder/'independent-production-audit.json';audit=read(path)
        assert audit['execution_qualified'] and audit['local_cloud_final_lists_exact'] and audit['remote_source_exact']
        assert audit['notebook_sha256']==entry['notebook_sha256'] and audit['head_manifest_sha256']==entry['head_manifest_sha256']
        assert audit['remote_scientific_sha256']==entry['scientific_sha256'] and audit['source_sha256']==sha(ROOT/'scripts/audit_c56_production.py')
        for name,pin in audit['artifact_sha256'].items():assert sha(ROOT/name)==pin
        files.append(path)
    return files


def freeze():
    assert not (OUT/'evaluation-freeze.json').exists() and not (PANEL/'evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    family=read(MODELS/'complete-candidate-freeze.json');assert family['policy_sha256']==sha(POLICY)
    cloud=ROOT/'outputs/forks/c57-full-masked-validation-v1';health=read(cloud/'independent-core-health-audit.json')
    assert health['passed'] and health['queries']==482 and health['source_sha256']==sha(ROOT/'scripts/audit_complete_core_health.py')
    for name,pin in health['files_sha256'].items():assert sha(ROOT/name)==pin
    used=read(MODELS/'independent-used-quality-audit.json');assert used['passed_pair']
    parent=read(OUT/'complete-parent-freeze.json');assert parent['candidate_family_sha256']==sha(OUT/'candidate-family-freeze.json')==sha(MODELS/'complete-candidate-freeze.json')
    assert parent['policy_sha256']==sha(POLICY)
    for name,pin in parent['files_sha256'].items():assert sha(OUT/name)==pin
    mask=OUT/'target-masked-reference/independent-mask-audit.json';assert read(mask)['forbidden_rows_retained']==0
    folder=OUT/'head-predictions';receipt=read(folder/'score-receipt.json');audit=read(folder/'independent-ranking-audit.json')
    assert set(receipt['trials'])=={'cpu','gpu'} and receipt['baseline_sha256']==sha(OUT/'P16.csv')
    assert receipt['query_sha256']==sha(OUT/'queries.parquet') and receipt['head_manifest_sha256']==family['head_manifest_sha256']
    assert receipt['queries']==482 and receipt['acquisitions']==2125 and not receipt['quality_labels_read']
    assert audit['score_receipt_sha256']==sha(folder/'score-receipt.json')
    for trial in ['cpu','gpu']:
        assert audit['trials'][trial]['independent_insertion_order_exact'] and audit['trials'][trial]['protected_parent_prefix']==2
        assert audit['trials'][trial]['last_reorderable_parent_rank']==10
    files=[Path(__file__),ROOT/'scripts/audit_c58_quality.py',POLICY,MODELS/'complete-candidate-freeze.json',MODELS/'independent-used-quality-audit.json',MODELS/'independent-fitted-model-audit.json',
        OUT/'complete-parent-freeze.json',OUT/'candidate-family-freeze.json',mask,OUT/'target-masked-reference/mask-manifest.json',OUT/'target-parent/independent-parent-audit.json',
        folder/'score-receipt.json',folder/'independent-ranking-audit.json',PANEL/'validation-freeze.json',PANEL/'independent-acquisition-audit.json',PANEL/'raw-source-alias-audit.json',
        ROOT/'outputs/forks/c57-full-masked-validation-v1/collection-audit.json',ROOT/'outputs/forks/c57-full-masked-validation-v1/acquisition-fidelity-audit.json']
    files.extend(verify_production(family))
    files.extend([cloud/'independent-core-health-audit.json',ROOT/'scripts/audit_complete_core_health.py'])
    files.extend(ROOT/name for name in health['files_sha256'])
    for trial,entry in family['candidates'].items():
        bundle=ROOT/entry['bundle'];assert sha(bundle/'head-manifest.json')==entry['head_manifest_sha256']
        notebook=ROOT/entry['path'];assert sha(notebook/'notebook.ipynb')==entry['notebook_sha256'] and sha(notebook/'production-pin.json')==entry['pin_sha256']
        files.extend([bundle/'head-manifest.json',notebook/'notebook.ipynb',notebook/'production-pin.json'])
    for name,pin in receipt['files_sha256'].items():assert sha(folder/name)==pin;files.append(folder/name)
    identities={row.smiles:row.key for row in pd.read_parquet(folder/'candidate-identities.parquet').itertuples(index=False)}
    features=module('c58_freeze_identity','c09e_features.py')
    for path in paths().values():
        frame=pd.read_csv(path,dtype=str,keep_default_na=False);assert len(frame)==frame.molecule_id.nunique()==482;files.append(path)
        for text in frame.smiles:
            for s in text.split(';'):
                if s not in identities:identities[s]=features.structure(s)[0]
    pd.DataFrame([dict(smiles=s,key=k) for s,k in identities.items()]).to_parquet(OUT/'quality-identity-map.parquet',index=False)
    files.extend([OUT/'quality-identity-map.parquet',OUT/'queries.parquet'])
    write(OUT/'evaluation-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},all_predictions_frozen=True,
        source_quality_outcomes_read=False,C13_read=False,maximum_scoring_entries=2))
    print('C58 BOTH COMPLETE FRESH PREDICTIONS SEALED',flush=True)


def evaluate(resume=False):
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    frozen=read(OUT/'evaluation-freeze.json');assert frozen['source_sha256']==sha(__file__) and frozen['policy_sha256']==sha(POLICY)
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
    family=read(MODELS/'complete-candidate-freeze.json');verify_production(family)
    label_path=PANEL/'labels.csv';panel=read(PANEL/'validation-freeze.json');assert sha(label_path)==panel['files_sha256'][str(label_path.relative_to(ROOT))]
    marker=dict(started_utc=stamp(),evaluation_freeze_sha256=sha(OUT/'evaluation-freeze.json'),labels_sha256=sha(label_path),candidates=['cpu','gpu'],
        source='Entire frozen482C53queries/2125acquisitions',C13_read=False)
    # One global source marker, before any prediction/truth matching. Never
    # recover by removing it or changing the cohort/recipe after outcomes.
    if (PANEL/'evaluation-consumption.json').exists():
        prior=read(PANEL/'evaluation-consumption.json')
        assert resume and all(prior[k]==v for k,v in marker.items() if k!='started_utc'),'Already consumed; only the exact same committed calculation may resume'
        if (OUT/'evaluation-consumption.json').exists():assert read(OUT/'evaluation-consumption.json')==prior
        else:write(OUT/'evaluation-consumption.json',prior)
        if (OUT/'source-results.json').exists():
            result=read(OUT/'source-results.json')
            assert result['evaluation_freeze_sha256']==sha(OUT/'evaluation-freeze.json') and result['consumption_sha256']==sha(PANEL/'evaluation-consumption.json')
            print('C58 EXACT COMMITTED SOURCE RESULTS ALREADY COMPLETE',flush=True);return
    else:
        assert not resume and not (OUT/'source-results.json').exists() and not (OUT/'evaluation-consumption.json').exists()
        with (PANEL/'evaluation-consumption.json').open('x',encoding='utf8') as stream:json.dump(marker,stream,indent=2)
        write(OUT/'evaluation-consumption.json',marker)
    frames={name:pd.read_csv(path,dtype=str,keep_default_na=False) for name,path in paths().items()};mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==482
    labels=pd.read_csv(label_path,dtype=str).set_index('molecule_id').loc[mids]
    identity=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict();lists={};ranks={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[identity[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in lists[name])
        ranks[name]=np.asarray([row.index(truth)+1 if truth in row else 0 for row,truth in zip(lists[name],labels.key)],np.int16)
    q=pd.read_parquet(OUT/'queries.parquet');assert len(q)==2125
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict()=={'positive':307,'mixed':141,'negative':34} and labels.group.nunique()==447
    groups=labels.group.to_numpy();all_mask=np.ones(482,bool)
    def summary(base,trial,mask):
        a=np.divide(1.,ranks[base],out=np.zeros(482),where=ranks[base]>0);b=np.divide(1.,ranks[trial],out=np.zeros(482),where=ranks[trial]>0)
        value=clustered_summary(a[mask],b[mask],groups[mask],10000,20261158);value['point']=point(ranks[base],ranks[trial],mask);return value
    def compare(base,trial):
        return dict(all=summary(base,trial,all_mask),source={s:summary(base,trial,labels.source.eq(s).to_numpy()) for s in sorted(set(labels.source))},
            query_mode={m:summary(base,trial,modes==m) for m in sorted(set(modes))},
            source_mode={s+' / '+m:summary(base,trial,labels.source.eq(s).to_numpy()&(modes==m)) for s in sorted(set(labels.source)) for m in sorted(set(modes)) if (labels.source.eq(s).to_numpy()&(modes==m)).any()},
            recovered_truths=int(((ranks[base]==0)&(ranks[trial]>0)).sum()),displaced_truths=int(((ranks[base]>0)&(ranks[trial]==0)).sum()),
            truth_absent_queries=int((ranks[trial]==0).sum()),rank1_changes=sum(a[0]!=b[0] for a,b in zip(lists[base],lists[trial])),
            candidate_membership_changes=sum(set(a)!=set(b) for a,b in zip(lists[base],lists[trial])))
    results={}
    for trial in ['cpu','gpu']:
        assert all(a[:2]==b[:2] and a[10:]==b[10:] and set(a)==set(b) for a,b in zip(lists['P16'],lists[trial]))
        complete=compare('P01',trial);incremental=compare('P16',trial)
        passed=all(v['all']['point']['strictly_positive'] and all(s['point']['nonnegative'] for field in ['source','query_mode'] for s in v[field].values()) for v in [complete,incremental])
        results[trial]=dict(complete=complete,incremental=incremental,admission=dict(passed=passed,statistical_promotion=False))
    np.savez(OUT/'source-ranks.npz',**ranks)
    write(OUT/'source-results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),evaluation_freeze_sha256=sha(OUT/'evaluation-freeze.json'),
        consumption_sha256=sha(PANEL/'evaluation-consumption.json'),queries=482,acquisitions=2125,chemistry_groups=447,results=results,parent_overlay=compare('P01','P16'),
        passed_pair=all(v['admission']['passed'] for v in results.values()),all_queries_retained=True,absent_truths_zero=True,
        actual_query_modes=pd.Series(modes).value_counts().to_dict(),C13_read=False,
        limitation='Controlled target-domain in-corpus confirmation, not an independent laboratory or leaderboard-calibrated population. Public parent/encoder historic exposure persists. No recipe/cohort rescue after these outcomes.'))
    print('C58 FRESH TARGET ADMISSION', {t:dict(admission=v['admission'],all=v['complete']['all']['point']) for t,v in results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);p.add_argument('--resume-same-evaluation',action='store_true');a=p.parse_args()
    if a.phase=='freeze':freeze()
    else:evaluate(a.resume_same_evaluation)
