"""Freeze all predictions, select parents on development, then check sources.

No source-outcome rescue, head/weight search or C13 access is implemented here.
"""
import os
for _name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(_name,'2')
import argparse,json
from fractions import Fraction
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from validation_repair import clustered_summary
from c45_complete_parent import ROOT,OUT,CODE,C39,POLICY,read,write,sha,module


def stamp():return datetime.now(timezone.utc).isoformat()


def point(before,after,mask):
    positions=np.flatnonzero(mask);assert len(positions)
    base=sum((Fraction(1,int(before[i])) if before[i] else Fraction(0) for i in positions),Fraction(0))/len(positions)
    value=sum((Fraction(1,int(after[i])) if after[i] else Fraction(0) for i in positions),Fraction(0))/len(positions)
    delta=value-base
    return dict(n=len(positions),baseline_MRR=float(base),candidate_MRR=float(value),delta=float(delta),
        numerator=delta.numerator,denominator=delta.denominator,strictly_positive=delta>0,nonnegative=delta>=0)


def select_variants(development_points):
    choices={}
    for trial in ['cpu','gpu']:
        eligible=[parent for parent in ['P01','P16'] if all(development_points[trial+'_'+parent][comparison]['strictly_positive'] for comparison in ['complete','incremental'])]
        selected=None
        for parent in eligible:
            value=development_points[trial+'_'+parent]['complete']
            score=Fraction(value['numerator'],value['denominator'])
            if selected is None or score>best:selected=parent;best=score
        choices[trial]=selected
    return choices


def sources_admit(summaries):
    return summaries['development']['point']['strictly_positive'] and all(value['point']['nonnegative'] for field in ['source','query_mode'] for value in summaries[field].values())


def paths():
    result={name:OUT/(name+'.csv') for name in ['P01','P16','P14']}
    for parent in ['P01','P16']:
        for trial in ['cpu','gpu']:result[trial+'_'+parent]=OUT/('heads-'+parent)/(trial+'.csv')
    return result


def freeze():
    assert not (OUT/'evaluation-freeze.json').exists() and not (OUT/'development-results.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    parent=read(OUT/'complete-parent-freeze.json');family=read(OUT/'candidate-family-freeze.json')
    assert parent['policy_sha256']==sha(POLICY) and parent['candidate_family_sha256']==sha(OUT/'candidate-family-freeze.json')
    for name,pin in parent['files_sha256'].items():assert sha(OUT/name)==pin,name
    mask=OUT/'C39-masked-reference/independent-mask-audit.json';mask_audit=read(mask)
    assert mask_audit['mask_manifest_sha256']==sha(OUT/'C39-masked-reference/mask-manifest.json') and mask_audit['forbidden_rows_retained']==0
    assert mask_audit['source_sha256']==sha(ROOT/'scripts/audit_c45_reference_mask.py')
    files=[OUT/'complete-parent-freeze.json',OUT/'candidate-family-freeze.json',POLICY,Path(__file__),mask,
        OUT/'C39-masked-reference/mask-manifest.json',ROOT/'scripts/audit_c45_reference_mask.py',ROOT/'scripts/audit_c45_quality.py']
    for trial,variants in family['families'].items():
        for value in variants.values():
            folder=ROOT/value['path'];assert sha(folder/'notebook.ipynb')==value['notebook_sha256']
            assert sha(folder/'production-pin.json')==value['pin_sha256']
            files.extend([folder/'notebook.ipynb',folder/'production-pin.json'])
    identities={}
    for name,path in paths().items():assert path.exists(),name;files.append(path)
    for variant in ['P01','P16']:
        folder=OUT/('heads-'+variant);receipt=read(folder/'score-receipt.json');audit=read(folder/'independent-ranking-audit.json')
        assert receipt['baseline_sha256']==sha(OUT/(variant+'.csv')) and receipt['queries']==352 and not receipt['quality_labels_read']
        assert receipt['head_manifest_sha256']==family['head_manifest_sha256'] and audit['score_receipt_sha256']==sha(folder/'score-receipt.json')
        assert receipt['query_sha256']==sha(OUT/'queries.parquet')
        if receipt.get('execution_kind')=='derived_from_exact_equivalent_parent':
            proof=read(folder/'exact-parent-reuse.json')
            assert receipt['exact_parent_reuse_sha256']==sha(folder/'exact-parent-reuse.json')
            assert proof['source_sha256']==sha(ROOT/'scripts/reuse_c45_equivalent_parent_scores.py')
            assert proof['original_score_receipt_sha256']==sha(OUT/'heads-P01/score-receipt.json')
            assert proof['complete_parent_freeze_sha256']==sha(OUT/'complete-parent-freeze.json')
            assert proof['P01_sha256']==sha(OUT/'P01.csv') and proof['P16_sha256']==sha(OUT/'P16.csv')
            assert proof['all_parent_input_rows_and_strings_exact'] and not proof['second_GPU_execution_performed']
            pd.testing.assert_frame_equal(pd.read_csv(OUT/'P01.csv',dtype=str),pd.read_csv(OUT/'P16.csv',dtype=str),check_exact=True)
            files.extend([folder/'exact-parent-reuse.json',ROOT/'scripts/reuse_c45_equivalent_parent_scores.py'])
        for name,pin in receipt['files_sha256'].items():assert sha(folder/name)==pin,name;files.append(folder/name)
        for trial in ['cpu','gpu']:assert audit['trials'][trial]['independent_RRF_exact']
        records=pd.read_parquet(folder/'candidate-identities.parquet')
        for row in records.itertuples(index=False):
            if row.smiles in identities:assert identities[row.smiles]==row.key
            identities[row.smiles]=row.key
        files.extend([folder/'score-receipt.json',folder/'independent-ranking-audit.json'])
    features=module('c45_quality_features','c09e_features.py')
    for path in paths().values():
        frame=pd.read_csv(path,dtype=str,keep_default_na=False);assert frame.molecule_id.nunique()==len(frame)==352
        for text in frame.smiles:
            for smiles in text.split(';'):
                if smiles not in identities:identities[smiles]=features.structure(smiles)[0]
    pd.DataFrame([dict(smiles=s,key=k) for s,k in identities.items()]).to_parquet(OUT/'quality-identity-map.parquet',index=False)
    files.append(OUT/'quality-identity-map.parquet')
    write(OUT/'evaluation-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},all_predictions_frozen=True,
        source_outcomes_read=False,development_outcomes_read=False,C13_read=False,maximum_scoring_entries=2))


def verified():
    frozen=read(OUT/'evaluation-freeze.json');assert frozen['source_sha256']==sha(__file__) and frozen['policy_sha256']==sha(POLICY)
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
    labels=pd.read_csv(C39/'labels.csv',dtype=str).set_index('molecule_id')
    original=read(C39/'validation-freeze.json');assert sha(C39/'labels.csv')==original['files_sha256'][str((C39/'labels.csv').relative_to(ROOT))]
    frames={name:pd.read_csv(path,dtype=str,keep_default_na=False) for name,path in paths().items()}
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==352
    assert all(frame.molecule_id.tolist()==mids for frame in frames.values());labels=labels.loc[mids]
    assert labels.partition.eq('development').sum()==192 and len(set(labels.group))==273
    assert not set(labels.loc[labels.partition.eq('development'),'group'])&set(labels.loc[~labels.partition.eq('development'),'group'])
    mapping=pd.read_parquet(OUT/'quality-identity-map.parquet');keys=dict(zip(mapping.smiles,mapping.key))
    candidate_keys={name:[[keys[s] for s in text.split(';')] for text in frame.smiles] for name,frame in frames.items()}
    for lists in candidate_keys.values():assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in lists)
    for parent in ['P01','P16']:
        for trial in ['cpu','gpu']:
            assert all(a[0]==b[0] and set(a)==set(b) for a,b in zip(candidate_keys[parent],candidate_keys[trial+'_'+parent]))
    queries=pd.read_parquet(OUT/'queries.parquet');assert len(queries)==1292 and queries.molecule_id.nunique()==352
    modes=queries.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict()=={'positive':176,'mixed':126,'negative':50}
    return labels,candidate_keys,modes


def ranks(keys,labels,positions,names):
    result={name:np.zeros(len(labels),np.int16) for name in names}
    for i in positions:
        truth=labels.key.iloc[i]
        for name in names:
            row=keys[name][i];result[name][i]=row.index(truth)+1 if truth in row else 0
    return result


def development():
    assert not (OUT/'development-results.json').exists() and not (OUT/'source-results.json').exists()
    labels,keys,modes=verified();mask=labels.partition.eq('development').to_numpy();positions=np.flatnonzero(mask)
    rr=ranks(keys,labels,positions,list(keys));points={}
    for trial in ['cpu','gpu']:
        for parent in ['P01','P16']:
            name=trial+'_'+parent;points[name]=dict(complete=point(rr['P01'],rr[name],mask),incremental=point(rr[parent],rr[name],mask))
    choices=select_variants(points);family=read(OUT/'candidate-family-freeze.json')
    record=dict(completed_utc=stamp(),evaluation_freeze_sha256=sha(OUT/'evaluation-freeze.json'),queries=192,
        candidate_family_size=4,results=points,choices=choices,source_outcomes_read=False,C13_read=False,
        exact_rule='Eligible positive complete and incremental development deltas; highest complete delta within each head; exact ties prefer P01.')
    write(OUT/'development-results.json',record);np.savez(OUT/'development-ranks.npz',**rr)
    write(OUT/'selected-pair-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        development_results_sha256=sha(OUT/'development-results.json'),development_ranks_sha256=sha(OUT/'development-ranks.npz'),
        evaluation_freeze_sha256=sha(OUT/'evaluation-freeze.json'),choices=choices,
        candidates={trial:None if parent is None else family['families'][trial][parent] for trial,parent in choices.items()},
        both_selected=all(value is not None for value in choices.values()),source_outcomes_read=False,C13_read=False))
    print('C45 DEVELOPMENT CHOICES',json.dumps(record),flush=True)


def source():
    assert not (OUT/'source-results.json').exists()
    chosen=read(OUT/'selected-pair-freeze.json');assert chosen['both_selected'],'No eligible development pair; sources remain unevaluated'
    assert chosen['source_sha256']==sha(__file__) and chosen['policy_sha256']==sha(POLICY)
    assert chosen['development_results_sha256']==sha(OUT/'development-results.json') and chosen['development_ranks_sha256']==sha(OUT/'development-ranks.npz')
    assert chosen['evaluation_freeze_sha256']==sha(OUT/'evaluation-freeze.json')
    labels,keys,modes=verified();choices=chosen['choices'];names=['P01','P16','P14']+[trial+'_'+parent for trial,parent in choices.items()]
    rr=ranks(keys,labels,np.arange(352),names);dev=labels.partition.eq('development').to_numpy();old=np.load(OUT/'development-ranks.npz')
    for name in names:assert np.array_equal(old[name][dev],rr[name][dev])
    groups=labels.group.to_numpy();full=np.ones(352,bool)
    def summary(base,name,mask):
        before=np.asarray([1./v if v else 0. for v in rr[base]])
        after=np.asarray([1./v if v else 0. for v in rr[name]])
        value=clustered_summary(before[mask],after[mask],groups[mask],10000,20261145)
        value['point']=point(rr[base],rr[name],mask);return value
    def compare(base,name):
        return dict(all=summary(base,name,full),development=summary(base,name,dev),
            source={v:summary(base,name,labels.source.eq(v).to_numpy()) for v in sorted(set(labels.source))},
            query_mode={v:summary(base,name,modes==v) for v in sorted(set(modes))},
            source_mode={s+' / '+m:summary(base,name,labels.source.eq(s).to_numpy()&(modes==m)) for s in sorted(set(labels.source)) for m in sorted(set(modes)) if (labels.source.eq(s).to_numpy()&(modes==m)).any()},
            recovered_truths=int(((rr[base]==0)&(rr[name]>0)).sum()),displaced_truths=int(((rr[base]>0)&(rr[name]==0)).sum()),
            truth_absent_queries=int((rr[name]==0).sum()),rank1_changes=sum(a[0]!=b[0] for a,b in zip(keys[base],keys[name])),
            candidate_membership_changes=sum(set(a)!=set(b) for a,b in zip(keys[base],keys[name])))
    results={}
    for trial,parent in choices.items():
        name=trial+'_'+parent;complete=compare('P01',name);incremental=compare(parent,name)
        assert incremental['recovered_truths']==incremental['displaced_truths']==incremental['rank1_changes']==incremental['candidate_membership_changes']==0
        reasons=[]
        if not sources_admit(complete):reasons.append('complete_candidate_directional_gate_failed')
        if not sources_admit(incremental):reasons.append('incremental_head_directional_gate_failed')
        results[trial]=dict(parent=parent,complete=complete,incremental=incremental,admission=dict(passed=not reasons,reasons=reasons,statistical_promotion=False))
    record=dict(completed_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),queries=352,acquisitions=1292,
        selected_pair_sha256=sha(OUT/'selected-pair-freeze.json'),results=results,parent_overlay=compare('P01','P16'),P14_control=compare('P01','P14'),
        all_queries_retained=True,absent_truths_zero=True,actual_query_modes=pd.Series(modes).value_counts().to_dict(),
        choices_not_revised_using_source_outcomes=True,C13_read=False,leaderboard_calibration_established=False,
        passed_pair=all(value['admission']['passed'] for value in results.values()),
        limitation='Controlled in-corpus reference-masked source study; public parent and encoder pretraining exposure remains known or unknown. Chemistry-grouped uncertainty does not prove public transfer.')
    np.savez(OUT/'source-ranks.npz',**rr);write(OUT/'source-results.json',record)
    print('C45 SOURCE GATES',json.dumps({trial:value['admission'] for trial,value in results.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','development','source']);args=parser.parse_args()
    {'freeze':freeze,'development':development,'source':source}[args.phase]()
