"""Independent rank/denominator/selection/directional audit of C45 outcomes.

Only run after source-results exists. Does not evaluate unselected source arms.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import json,hashlib
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c45-complete-parent-validation-20261112'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def mean(ranks,mask):
    counts=np.bincount(np.asarray(ranks)[mask],minlength=26)
    return sum((Fraction(int(counts[i]),i) for i in range(1,26)),Fraction(0))/int(np.sum(mask))


def check_point(record,before,after,mask):
    a,b=mean(before,mask),mean(after,mask);delta=b-a
    assert record['n']==int(np.sum(mask))
    assert (record['numerator'],record['denominator'])==(delta.numerator,delta.denominator)
    assert record['strictly_positive']==(delta>0) and record['nonnegative']==(delta>=0)
    for field,value in [('baseline_MRR',a),('candidate_MRR',b),('delta',delta)]:assert abs(record[field]-float(value))<1e-14
    return delta


def main():
    target=OUT/'independent-quality-audit.json';assert not target.exists()
    report=read(OUT/'source-results.json');selected=read(OUT/'selected-pair-freeze.json')
    development=read(OUT/'development-results.json');frozen=read(OUT/'evaluation-freeze.json')
    assert selected['both_selected'] and report['selected_pair_sha256']==sha(OUT/'selected-pair-freeze.json')
    assert report['source_sha256']==selected['source_sha256']==frozen['source_sha256']==sha(ROOT/'scripts/evaluate_c45_complete.py')
    assert report['policy_sha256']==selected['policy_sha256']==frozen['policy_sha256']==sha(ROOT/'configs/c45-complete-parent-validation-20261112.json')
    assert selected['development_results_sha256']==sha(OUT/'development-results.json')
    assert selected['development_ranks_sha256']==sha(OUT/'development-ranks.npz')
    assert selected['evaluation_freeze_sha256']==sha(OUT/'evaluation-freeze.json')
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
    panel=ROOT/'outputs/local/c39-source-supported-validation-20261111';labels=pd.read_csv(panel/'labels.csv',dtype=str).set_index('molecule_id')
    label_freeze=read(panel/'validation-freeze.json');assert sha(panel/'labels.csv')==label_freeze['files_sha256'][str((panel/'labels.csv').relative_to(ROOT))]
    parent=pd.read_csv(OUT/'P01.csv',dtype=str,keep_default_na=False);mids=parent.molecule_id.tolist();labels=labels.loc[mids]
    assert len(mids)==len(set(mids))==352
    development_mask=labels.partition.eq('development').to_numpy();assert development_mask.sum()==192
    identities=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict()
    frames={'P01':parent,'P16':pd.read_csv(OUT/'P16.csv',dtype=str,keep_default_na=False),'P14':pd.read_csv(OUT/'P14.csv',dtype=str,keep_default_na=False)}
    for trial in ['cpu','gpu']:
        for variant in ['P01','P16']:frames[trial+'_'+variant]=pd.read_csv(OUT/('heads-'+variant)/(trial+'.csv'),dtype=str,keep_default_na=False)
    lists={};dev_ranks={};source_ranks={}
    chosen=selected['choices'];names=['P01','P16','P14']+[trial+'_'+variant for trial,variant in chosen.items()]
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids
        lists[name]=[[identities[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in lists[name])
        dev_ranks[name]=np.zeros(352,np.int16)
        if name in names:source_ranks[name]=np.zeros(352,np.int16)
        for i in range(352):
            # Reserved source truth is never matched to an unselected arm.
            if development_mask[i] or name in names:
                positions={key:rank for rank,key in enumerate(lists[name][i],1)};rank=positions.get(labels.key.iloc[i],0)
                if development_mask[i]:dev_ranks[name][i]=rank
                if name in names:source_ranks[name][i]=rank
    stored_dev=np.load(OUT/'development-ranks.npz');stored_source=np.load(OUT/'source-ranks.npz')
    for name,values in dev_ranks.items():np.testing.assert_array_equal(values,stored_dev[name])
    for name,values in source_ranks.items():np.testing.assert_array_equal(values,stored_source[name])
    expected={}
    for trial in ['cpu','gpu']:
        eligible=[]
        for variant in ['P01','P16']:
            name=trial+'_'+variant;entry=development['results'][name]
            full=check_point(entry['complete'],dev_ranks['P01'],dev_ranks[name],development_mask)
            incremental=check_point(entry['incremental'],dev_ranks[variant],dev_ranks[name],development_mask)
            if full>0 and incremental>0:eligible.append((full,variant))
        # Reversed enumeration followed by a stable max implements P01 tie priority.
        expected[trial]=max(eligible,key=lambda v:(v[0],v[1]=='P01'))[1] if eligible else None
    assert expected==chosen==development['choices']
    q=pd.read_parquet(OUT/'queries.parquet');assert len(q)==1292
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda x:'mixed' if len(set(x))>1 else x.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict()=={'positive':176,'mixed':126,'negative':50}
    all_mask=np.ones(352,bool)
    def check_comparison(summary,base,name):
        masks={'all':all_mask,'development':development_mask}
        for field in ['all','development']:check_point(summary[field]['point'],source_ranks[base],source_ranks[name],masks[field])
        signs=[]
        assert set(summary['source'])==set(labels.source) and set(summary['query_mode'])==set(modes)
        for source,record in summary['source'].items():signs.append(check_point(record['point'],source_ranks[base],source_ranks[name],labels.source.eq(source).to_numpy()))
        for mode,record in summary['query_mode'].items():signs.append(check_point(record['point'],source_ranks[base],source_ranks[name],modes==mode))
        intersections={s+' / '+m:labels.source.eq(s).to_numpy()&(modes==m) for s in set(labels.source) for m in set(modes)}
        intersections={k:v for k,v in intersections.items() if v.any()};assert set(intersections)==set(summary['source_mode'])
        for key,mask in intersections.items():check_point(summary['source_mode'][key]['point'],source_ranks[base],source_ranks[name],mask)
        a,b=source_ranks[base],source_ranks[name]
        assert summary['recovered_truths']==int(((a==0)&(b>0)).sum()) and summary['displaced_truths']==int(((a>0)&(b==0)).sum())
        assert summary['truth_absent_queries']==int((b==0).sum())
        assert summary['rank1_changes']==sum(a[0]!=b[0] for a,b in zip(lists[base],lists[name]))
        assert summary['candidate_membership_changes']==sum(set(a)!=set(b) for a,b in zip(lists[base],lists[name]))
        return check_point(summary['development']['point'],source_ranks[base],source_ranks[name],development_mask)>0 and all(v>=0 for v in signs)
    admissions={}
    for trial,variant in chosen.items():
        entry=report['results'][trial];name=trial+'_'+variant
        full=check_comparison(entry['complete'],'P01',name);incremental=check_comparison(entry['incremental'],variant,name)
        assert entry['admission']['passed']==(full and incremental)
        admissions[trial]=full and incremental
        assert all(a[0]==b[0] and set(a)==set(b) for a,b in zip(lists[variant],lists[name]))
    check_comparison(report['parent_overlay'],'P01','P16');check_comparison(report['P14_control'],'P01','P14')
    assert report['passed_pair']==all(admissions.values()) and report['all_queries_retained'] and report['absent_truths_zero'] and not report['C13_read']
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        quality_results_sha256=sha(OUT/'source-results.json'),selected_pair_sha256=sha(OUT/'selected-pair-freeze.json'),
        queries=352,acquisitions=1292,exact_rank_vectors=True,exact_fraction_points=True,
        development_only_choices_exact=True,absent_truths_zero=True,all_source_mode_denominators_exact=True,
        recovered_displaced_counts_exact=True,admissions=admissions,passed_pair=all(admissions.values()),C13_read=False)
    target.write_text(json.dumps(result,indent=2),encoding='utf8');print('C45 INDEPENDENT QUALITY AUDIT',json.dumps(result))


if __name__=='__main__':main()
