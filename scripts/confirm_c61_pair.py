"""Exactly one joint full/incremental C13confirmation for the frozen C56pair."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import RDLogger
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from evaluate_c45_complete import point
from validation_repair import clustered_summary
OUT=ROOT/'outputs/local/c61-paired-confirmation-20261112'
BASE=ROOT/'outputs/local/c60-complete-c13-20261112'
MODELS=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
C13=ROOT/'outputs/local/c13-confirmation-20261110'
POLICY=ROOT/'configs/c56-bounded-semantic-pair-20261112.json'
CONSUMPTION=C13/'evaluation-consumption.json'


def freeze():
    from release_c59_c13 import verified
    verified();OUT.mkdir(exist_ok=True);assert not (OUT/'prediction-freeze.json').exists() and not CONSUMPTION.exists()
    family=read(MODELS/'complete-candidate-freeze.json');assert family['policy_sha256']==sha(POLICY)
    cloud=ROOT/'outputs/forks/c59-c13-main-compatibility-v1';health=read(cloud/'independent-core-health-audit.json')
    assert health['passed'] and health['queries']==120 and health['source_sha256']==sha(ROOT/'scripts/audit_c59_core_health.py')
    assert health['pubchem_supported']==119 and health['pubchem_unsupported']==1 and health['all_original_queries_retained']
    for name,pin in health['files_sha256'].items():assert sha(ROOT/name)==pin
    parent=read(BASE/'complete-parent-freeze.json');assert parent['candidate_family_sha256']==sha(BASE/'candidate-family-freeze.json')==sha(MODELS/'complete-candidate-freeze.json')
    for name,pin in parent['files_sha256'].items():assert sha(BASE/name)==pin
    folder=BASE/'head-predictions';receipt=read(folder/'score-receipt.json');audit=read(folder/'independent-ranking-audit.json')
    assert receipt['head_manifest_sha256']==family['head_manifest_sha256'] and set(receipt['trials'])=={'cpu','gpu'}
    assert receipt['baseline_sha256']==sha(BASE/'P16.csv') and receipt['query_sha256']==sha(BASE/'queries.parquet') and receipt['queries']==receipt['acquisitions']==120
    assert not receipt['quality_labels_read'] and audit['score_receipt_sha256']==sha(folder/'score-receipt.json')
    for trial in ['cpu','gpu']:
        assert audit['trials'][trial]['independent_insertion_order_exact'] and audit['trials'][trial]['protected_parent_prefix']==2 and audit['trials'][trial]['last_reorderable_parent_rank']==10
    labels_pin=read(C13/'preparation-report.json')['labels_sha256'];assert sha(C13/'labels.csv')==labels_pin
    files=[Path(__file__),ROOT/'scripts/c61_exposure.py',ROOT/'scripts/audit_c61_confirmation.py',POLICY,MODELS/'complete-candidate-freeze.json',
        BASE/'complete-parent-freeze.json',BASE/'candidate-family-freeze.json',BASE/'C13-masked-reference/mask-manifest.json',
        BASE/'C13-masked-reference/independent-mask-audit.json',BASE/'C13-parent/independent-parent-audit.json',
        ROOT/'outputs/forks/c59-c13-main-compatibility-v1/collection-audit.json',ROOT/'outputs/forks/c59-c13-main-compatibility-v1/acquisition-fidelity-audit.json',
        ROOT/'kaggle/forks/c59-c13-main-compatibility/compatibility-pin.json',C13/'preparation-report.json',C13/'identity-inventory.parquet',
        ROOT/'outputs/local/c58-complete-target-20261112/source-results.json',ROOT/'outputs/local/c58-complete-target-20261112/independent-quality-audit.json',
        MODELS/'independent-fitted-model-audit.json',MODELS/'independent-used-quality-audit.json',folder/'score-receipt.json',folder/'independent-ranking-audit.json']
    identities={r.smiles:r.key for r in pd.read_parquet(folder/'candidate-identities.parquet').itertuples(index=False)}
    features=module('c61_freeze_official_identity','c09e_features.py')
    for name in ['P01','P16']:
        path=BASE/(name+'.csv');files.append(path);frame=pd.read_csv(path,dtype=str,keep_default_na=False)
        assert len(frame)==frame.molecule_id.nunique()==120
        for text in frame.smiles:
            for s in text.split(';'):
                if s not in identities:identities[s]=features.structure(s)[0]
    for name,pin in receipt['files_sha256'].items():assert sha(folder/name)==pin;files.append(folder/name)
    files.append(BASE/'queries.parquet')
    files.extend([cloud/'independent-core-health-audit.json',ROOT/'scripts/audit_c59_core_health.py'])
    files.extend(ROOT/name for name in health['files_sha256'])
    pd.DataFrame([dict(smiles=s,key=k) for s,k in identities.items()]).to_parquet(OUT/'quality-identity-map.parquet',index=False);files.append(OUT/'quality-identity-map.parquet')
    for entry in family['candidates'].values():
        for name,pin in [('notebook.ipynb',entry['notebook_sha256']),('production-pin.json',entry['pin_sha256'])]:
            path=ROOT/entry['path']/name;assert sha(path)==pin;files.append(path)
        bundle=ROOT/entry['bundle'];assert sha(bundle/'head-manifest.json')==entry['head_manifest_sha256'];files.append(bundle/'head-manifest.json')
        files.append(ROOT/'outputs/forks'/(Path(entry['path']).name+'-v1')/'independent-production-audit.json')
    boundary=read(MODELS/'fit-boundary-freeze.json');feature=read(MODELS/'head-feature-freeze.json')
    manifest=read(ROOT/family['candidates']['cpu']['bundle']/'head-manifest.json')
    assert manifest['fit_boundary_sha256']==sha(MODELS/'fit-boundary-freeze.json') and manifest['feature_freeze_sha256']==sha(MODELS/'head-feature-freeze.json')
    files.extend([MODELS/'fit-boundary-freeze.json',MODELS/'head-feature-freeze.json',ROOT/'outputs/local/long-frozen-20261107-r1/molecules.parquet',ROOT/'outputs/local/long-frozen-20261107-r1/selected.parquet'])
    for name,pin in {**boundary['files_sha256'],**feature['files_sha256']}.items():assert sha(MODELS/name)==pin;files.append(MODELS/name)
    for name,pin in manifest['files'].items():
        path=ROOT/family['candidates']['cpu']['bundle']/name;assert sha(path)==pin;files.append(path)
    write(OUT/'prediction-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),C13_labels_sha256=labels_pin,
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},paired_complete_predictions_frozen=True,C13_quality_read=False,
        full_baseline='Faithful main/fragment-compatible C59P01, not the obsolete PV-onlyC13v2',incremental_baseline='Exact frozen P16 with whole231identity/input-alias quarantine',
        confirmation_gate='Nonnegative exact full AND incremental all120/positive60/negative60; fitted/prior-used official identity overlaps zero. All absent truths count zero. No cohort/recipe selection.'))
    print('C61 BOTH COMPLETE C13PREDICTIONS SEALED; QUALITY UNREAD',flush=True)


def evaluate(consume,resume):
    assert consume,'Explicit --consume selects the already authorized single joint confirmation'
    import msvcrt
    lock=(C13/'paired-consumption.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    frozen=read(OUT/'prediction-freeze.json');assert frozen['source_sha256']==sha(__file__) and frozen['policy_sha256']==sha(POLICY)
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
    assert sha(C13/'labels.csv')==frozen['C13_labels_sha256']
    commitment=dict(cycle='C61_C56_COMPLETE_PAIRED',source_sha256=sha(__file__),policy_sha256=sha(POLICY),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'))
    if CONSUMPTION.exists():
        marker=read(CONSUMPTION);assert resume and all(marker.get(k)==v for k,v in commitment.items()),'Reserved confirmation already consumed or a different seal'
        if (OUT/'confirmation-results.json').exists():
            old=read(OUT/'confirmation-results.json');assert old['commitment']==commitment
            marker.update(state='completed',completed_utc=stamp(),results_sha256=sha(OUT/'confirmation-results.json'));write(CONSUMPTION,marker);return
    else:
        assert not resume
        from release_c59_c13 import verified
        verified()
        with CONSUMPTION.open('x',encoding='utf8') as stream:json.dump(dict(started_utc=stamp(),state='one_joint_evaluation_started',**commitment),stream,indent=2)
    RDLogger.DisableLog('rdApp.*')
    frames={n:pd.read_csv(BASE/(n+'.csv'),dtype=str,keep_default_na=False) for n in ['P01','P16']}
    for trial in ['cpu','gpu']:frames[trial]=pd.read_csv(BASE/'head-predictions'/(trial+'.csv'),dtype=str,keep_default_na=False)
    mids=frames['P01'].molecule_id.tolist();assert len(mids)==len(set(mids))==120
    # Sole joint hypothesis evaluation. Structures now report all exposures;
    # no experimental fitting, recipe selection or cohort revision follows.
    labels=pd.read_csv(C13/'labels.csv',dtype=str).set_index('molecule_id').loc[mids]
    assert labels['mode'].value_counts().to_dict()=={'positive':60,'negative':60}
    q=pd.read_parquet(BASE/'queries.parquet');actual_modes=q.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids]
    assert actual_modes.tolist()==labels['mode'].tolist() and len(q)==120
    identity=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict();lists={};ranks={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[identity[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in lists[name])
        ranks[name]=np.asarray([row.index(truth)+1 if truth in row else 0 for row,truth in zip(lists[name],labels.key)],np.int16)
    np.savez(OUT/'confirmation-ranks.npz',**ranks)
    from c61_exposure import chemistry_and_fit_exposure
    groups,exposure=chemistry_and_fit_exposure(labels);np.save(OUT/'confirmation-groups.npy',groups)
    def summary(base,trial,mask):
        a=np.divide(1.,ranks[base],out=np.zeros(120),where=ranks[base]>0);b=np.divide(1.,ranks[trial],out=np.zeros(120),where=ranks[trial]>0)
        value=clustered_summary(a[mask],b[mask],groups[mask],10000,20261161);value['point']=point(ranks[base],ranks[trial],mask);return value
    def compare(base,trial):
        return dict(all120=summary(base,trial,np.ones(120,bool)),mode={m:summary(base,trial,labels['mode'].eq(m).to_numpy()) for m in ['positive','negative']},
            recovered_truths=int(((ranks[base]==0)&(ranks[trial]>0)).sum()),displaced_truths=int(((ranks[base]>0)&(ranks[trial]==0)).sum()),truth_absent_queries=int((ranks[trial]==0).sum()),
            rank1_changes=sum(a[0]!=b[0] for a,b in zip(lists[base],lists[trial])),candidate_membership_changes=sum(set(a)!=set(b) for a,b in zip(lists[base],lists[trial])))
    results={}
    for trial in ['cpu','gpu']:
        assert all(a[:2]==b[:2] and a[10:]==b[10:] and set(a)==set(b) for a,b in zip(lists['P16'],lists[trial]))
        negative=labels['mode'].eq('negative').to_numpy();assert frames[trial].loc[negative].equals(frames['P16'].loc[negative])
        full=compare('P01',trial);incremental=compare('P16',trial)
        passed=all(v['all120']['point']['nonnegative'] and all(s['point']['nonnegative'] for s in v['mode'].values()) for v in [full,incremental])
        passed=passed and exposure['new_head_identity_overlaps']==exposure['prior_used_identity_overlaps']==0
        results[trial]=dict(complete=full,incremental=incremental,admission=dict(passed=passed,statistical_promotion=False))
    record=dict(completed_utc=stamp(),commitment=commitment,queries=120,acquisitions=120,actual_query_modes=actual_modes.value_counts().to_dict(),results=results,
        parent_overlay=compare('P01','P16'),fitting_exposure=exposure,passed_pair=all(v['admission']['passed'] for v in results.values()),
        one_joint_reserved_evaluation=True,no_recipe_or_cohort_selection=True,all_absent_truths_zero=True,leaderboard_calibration_established=False,
        statistical_promotion=False,automatic_submission=False,
        limitation='Original single-acquisition FT/HCD confirmation is not target timsTOF calibration. Near fitting/prior chemistry and public pretraining exposure remain disclosed. Point admission and uncertain intervals do not prove public improvement.')
    write(OUT/'confirmation-results.json',record);marker=read(CONSUMPTION);marker.update(state='completed',completed_utc=stamp(),results_sha256=sha(OUT/'confirmation-results.json'));write(CONSUMPTION,marker)
    print('C61 SINGLE PAIRED CONFIRMATION RESULT',{t:dict(admission=v['admission'],full=v['complete']['all120']['point'],incremental=v['incremental']['all120']['point']) for t,v in results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);p.add_argument('--consume',action='store_true');p.add_argument('--resume-same-evaluation',action='store_true');a=p.parse_args()
    if a.phase=='freeze':freeze()
    else:evaluate(a.consume,a.resume_same_evaluation)
