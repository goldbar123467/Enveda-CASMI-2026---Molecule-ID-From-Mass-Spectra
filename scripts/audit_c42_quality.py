"""Independently recompute all C39 ranks and exact rational admission signs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from fractions import Fraction
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write,features


def main():
    out=ROOT/'outputs/local/c42-fixed-c39-validation-20261111'
    target=out/'independent-quality-audit.json';assert not target.exists()
    quality=json.loads((out/'quality-results.json').read_text())
    score=json.loads((out/'score-receipt.json').read_text())
    for n,h in score['files_sha256'].items():assert sha(out/n)==h,n
    cloud=ROOT/'outputs/forks/c41r3-full-masked-validation-v1'
    frames={'P01':pd.read_csv(cloud/'c41_p01.csv',dtype=str),'P14':pd.read_csv(cloud/'c41_p14.csv',dtype=str)}
    for t in ['cpu','gpu']:frames[t]=pd.read_csv(out/(t+'.csv'),dtype=str)
    mids=frames['P01'].molecule_id.tolist()
    assert len(mids)==len(set(mids))==352 and all(f.molecule_id.tolist()==mids for f in frames.values())
    labels_path=ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv'
    freeze=json.loads((labels_path.parent/'validation-freeze.json').read_text())
    assert sha(labels_path)==freeze['files_sha256'][str(labels_path.relative_to(ROOT))]
    labels=pd.read_csv(labels_path,dtype=str).set_index('molecule_id').loc[mids]
    ids=pd.read_parquet(out/'candidate-identities.parquet');keys=dict(zip(ids.smiles,ids.key))
    vectors=np.load(out/'quality-vectors.npz',allow_pickle=False);ranks={};fractions={}
    for name,f in frames.items():
        rr=[];rank=[]
        for text,truth in zip(f.smiles,labels.key):
            kk=[]
            for s in text.split(';'):
                if s not in keys:keys[s]=features.structure(s)[0]
                kk.append(keys[s])
            assert 1<=len(kk)<=25 and len(set(kk))==len(kk)
            r=kk.index(truth)+1 if truth in kk else 0
            rank.append(r);rr.append(Fraction(1,r) if r else Fraction(0))
        assert np.array_equal(np.asarray([float(v) for v in rr]),vectors[name]),name
        fractions[name]=rr;ranks[name]=rank
    queries=pd.read_parquet(cloud/'c41_queries.parquet')
    mode=queries.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy()
    masks={'all':np.ones(352,bool),'development':labels.partition.eq('development').to_numpy()}
    for field,array in [('source',labels.source.to_numpy()),('query_mode',mode)]:
        for value in sorted(set(array)):masks[field+':'+value]=array==value
    def exact_delta(name,mask):
        n=int(mask.sum());assert n
        value=sum((fractions[name][i]-fractions['P01'][i] for i in np.flatnonzero(mask)),Fraction(0))/n
        return dict(n=n,numerator=value.numerator,denominator=value.denominator,value=float(value),strictly_positive=value>0,nonnegative=value>=0)
    results={}
    for name in ['cpu','gpu','P14']:
        points={key:exact_delta(name,mask) for key,mask in masks.items()}
        original=quality['results'][name]
        assert abs(points['all']['value']-original['all']['delta'])<1e-14
        assert abs(points['development']['value']-original['partition']['development']['delta'])<1e-14
        for field in ['source','query_mode']:
            for value,summary in original[field].items():assert abs(points[field+':'+value]['value']-summary['delta'])<1e-14
        item=dict(points=points,all352_rank_vectors_exact=True)
        if name!='P14':
            exact_pass=points['development']['strictly_positive'] and all(p['nonnegative'] for key,p in points.items() if key.startswith(('source:','query_mode:')))
            item.update(exact_predeclared_gates_passed=exact_pass,original_float_gate_passed=original['admission']['passed'],
                admission_agrees=exact_pass==original['admission']['passed'],passed=exact_pass and original['admission']['passed'])
        results[name]=item
    record=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        quality_results_sha256=sha(out/'quality-results.json'),score_receipt_sha256=sha(out/'score-receipt.json'),
        labels_sha256=sha(labels_path),queries=352,results=results,passed_pair=all(results[t]['passed'] for t in ['cpu','gpu']),
        C13_read=False,posthoc_threshold_selection=False,
        purpose='Exact reciprocal-rank arithmetic independently verifies the original zero-threshold signs, including true zero versus floating cancellation noise. No new model or cohort is selected here.')
    write(target,record);print('C42 INDEPENDENT QUALITY AUDIT',json.dumps({t:dict(passed=results[t]['passed'],development=results[t]['points']['development']) for t in ['cpu','gpu']}),flush=True)


if __name__=='__main__':main()
