"""One predeclared paired C39 evaluation; no search or candidate modification."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write,features
from validation_repair import clustered_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('scores');a=p.parse_args();out=ROOT/a.scores
    cloud=ROOT/'outputs/forks/c41r2-full-masked-validation-v1'
    assert not (out/'quality-results.json').exists() and not (out/'evaluation-freeze.json').exists()
    collection=json.loads((cloud/'collection-audit.json').read_text())
    assert collection['queries']==352 and collection['forbidden_reference_rows']==0
    for n,h in collection['files_sha256'].items():assert sha(cloud/n)==h,n
    receipt=json.loads((out/'score-receipt.json').read_text())
    for n,h in receipt['files_sha256'].items():assert sha(out/n)==h,n
    assert receipt['baseline_sha256']==sha(cloud/'c41_p01.csv')
    bundle=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111'
    assert receipt['head_manifest_sha256']==sha(bundle/'head-manifest.json')
    assert receipt['source_sha256']==json.loads((bundle/'head-manifest.json').read_text())['files']['score_c42_crossmodal.py']
    plan=json.loads((bundle/'trial-plan.json').read_text())
    candidates=pd.read_parquet(out/'candidate-identities.parquet')
    keys=dict(zip(candidates.smiles,candidates.key));formulas=dict(zip(candidates.smiles,candidates.formula))
    frames={'P01':pd.read_csv(cloud/'c41_p01.csv',dtype=str),'P14':pd.read_csv(cloud/'c41_p14.csv',dtype=str)}
    for trial in ['cpu','gpu']:frames[trial]=pd.read_csv(out/(trial+'.csv'),dtype=str)
    mids=frames['P01'].molecule_id.tolist()
    for name,f in frames.items():assert f.molecule_id.tolist()==mids,name
    for text in frames['P14'].smiles:
        for s in text.split(';'):
            if s not in keys:key,_,formula,*_=features.structure(s);keys[s]=key;formulas[s]=formula
    for trial in ['cpu','gpu']:
        for b,c in zip(frames['P01'].smiles,frames[trial].smiles):
            before,after=b.split(';'),c.split(';')
            assert before[0]==after[0] and set(before)==set(after) and len(before)==len(after)
            assert [formulas[s] for s in before]==[formulas[s] for s in after]
    # Persist complete predictions and evaluation implementation before matching
    # any query label to a prediction. No threshold or recipe depends on results.
    write(out/'evaluation-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        trial_plan_sha256=sha(bundle/'trial-plan.json'),score_receipt_sha256=sha(out/'score-receipt.json'),
        collection_audit_sha256=sha(cloud/'collection-audit.json'),
        prediction_sha256={n:sha(cloud/('c41_'+n.lower()+'.csv')) if n in ['P01','P14'] else sha(out/(n+'.csv')) for n in frames},
        all_queries_retained=True,absent_truths_zero=True,quality_metrics_read=False,C13_read=False))
    labels=ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv'
    lab=pd.read_csv(labels,dtype={'molecule_id':str}).set_index('molecule_id').loc[mids]
    freeze=json.loads((labels.parent/'validation-freeze.json').read_text())
    assert sha(labels)==freeze['files_sha256'][str(labels.relative_to(ROOT))]
    q=pd.read_parquet(cloud/'c41_queries.parquet')
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda x:'mixed' if x.nunique()>1 else x.iloc[0]).loc[mids].to_numpy()
    rr={};present={}
    for name,f in frames.items():
        values=[];coverage=[]
        for text,truth in zip(f.smiles,lab.key):
            kk=[keys[s] for s in text.split(';')];assert len(set(kk))==len(kk)
            values.append(1./(1+kk.index(truth)) if truth in kk else 0.);coverage.append(truth in kk)
        rr[name]=np.asarray(values);present[name]=np.asarray(coverage)
    groups=lab.group.to_numpy();results={}
    def summary(name,mask):return clustered_summary(rr['P01'][mask],rr[name][mask],groups[mask],draws=10000,seed=20261142)
    for name in ['cpu','gpu','P14']:
        report=dict(all=summary(name,np.ones(len(mids),bool)),partition={},source={},query_mode={})
        for field,array in [('partition',lab.partition.to_numpy()),('source',lab.source.to_numpy()),('query_mode',modes)]:
            for value in sorted(set(array)):report[field][value]=summary(name,array==value)
        report['source_mode']={}
        for source in sorted(set(lab.source)):
            for mode in sorted(set(modes)):
                mask=(lab.source.to_numpy()==source)&(modes==mode)
                if mask.any():report['source_mode'][source+' / '+mode]=summary(name,mask)
        report.update(recovered_truths=int((~present['P01']&present[name]).sum()),displaced_truths=int((present['P01']&~present[name]).sum()),
            truth_absent_queries=int((~present[name]).sum()))
        if name!='P14':
            assert np.array_equal(present['P01'],present[name])
            reasons=[]
            if report['partition']['development']['delta']<=0:reasons.append('no_strict_development_gain')
            if any(x['delta']< -1e-12 for field in ['source','query_mode'] for x in report[field].values()):reasons.append('source_or_query_mode_regression')
            report['admission']=dict(passed=not reasons,reasons=reasons,intervals_are_uncertainty_not_proof=True,
                qualification_tier=plan['admission']['qualification_tier'],submission_qualified=False)
        results[name]=report
    record=dict(completed_utc=datetime.now(timezone.utc).isoformat(),evaluation_freeze_sha256=sha(out/'evaluation-freeze.json'),
        queries=len(mids),results=results,known_P14_local_sign=float(rr['P14'].mean()-rr['P01'].mean()),
        leaderboard_calibration_established=False,all_queries_retained=True,missing_truths_score_zero=True,
        C13_read=False,recipe_selection=False,automatic_submission=False,
        limitation='Controlled in-corpus source/acquisition study. Newly fitted heads exclude heldout chemistry; frozen public encoder and parent pretraining exposure remains known or unknown.')
    np.savez(out/'quality-vectors.npz',**rr)
    write(out/'quality-results.json',record)
    print('C42 FIXED QUALITY',json.dumps({n:dict(all=v['all'],admission=v.get('admission')) for n,v in results.items()}),flush=True)


if __name__=='__main__':main()
