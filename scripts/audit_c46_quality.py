"""Independently verify C46 exact ranks, zero accounting and source gates."""
from fractions import Fraction
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from audit_c45_quality import ROOT,read,sha,check_point
OUT=ROOT/'outputs/local/c46-top2-protected-pair-20261112'


def main():
    target=OUT/'independent-quality-audit.json';assert not target.exists()
    frozen=read(OUT/'prediction-freeze.json');selected=read(OUT/'selected-pair-freeze.json');report=read(OUT/'source-results.json')
    assert frozen['source_sha256']==report['source_sha256']==sha(ROOT/'scripts/evaluate_c46_top2.py')
    assert frozen['policy_sha256']==selected['policy_sha256']==report['policy_sha256']==sha(ROOT/'configs/c46-top2-protected-pair-20261112.json')
    assert report['selected_pair_sha256']==sha(OUT/'selected-pair-freeze.json') and report['prediction_freeze_sha256']==sha(OUT/'prediction-freeze.json')
    assert selected['development_results_sha256']==sha(OUT/'development-results.json') and selected['development_ranks_sha256']==sha(OUT/'development-ranks.npz')
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
    panel=ROOT/'outputs/local/c39-source-supported-validation-20261111';original=read(panel/'validation-freeze.json')
    assert sha(panel/'labels.csv')==original['files_sha256'][str((panel/'labels.csv').relative_to(ROOT))]
    frames={name:pd.read_csv(OUT/(name+'.csv'),dtype=str,keep_default_na=False) for name in ['P01','P14','cpu','gpu']}
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==352
    labels=pd.read_csv(panel/'labels.csv',dtype=str).set_index('molecule_id').loc[mids]
    mapping=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict();ranks={};lists={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[mapping[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in lists[name])
        ranks[name]=np.asarray([{key:i for i,key in enumerate(row,1)}.get(truth,0) for row,truth in zip(lists[name],labels.key)],np.int16)
    stored=np.load(OUT/'source-ranks.npz');devstored=np.load(OUT/'development-ranks.npz');dev=labels.partition.eq('development').to_numpy()
    assert dev.sum()==192 and len(set(labels.group))==273 and not set(labels.group[dev])&set(labels.group[~dev])
    for name in ranks:
        np.testing.assert_array_equal(stored[name],ranks[name]);np.testing.assert_array_equal(devstored[name][dev],ranks[name][dev])
        assert not devstored[name][~dev].any()
    development=read(OUT/'development-results.json');all_mask=np.ones(352,bool)
    q=pd.read_parquet(OUT/'queries.parquet');assert len(q)==1292
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda x:'mixed' if len(set(x))>1 else x.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict()=={'positive':176,'mixed':126,'negative':50}
    admissions={}
    for name,entry in report['results'].items():
        assert name in ranks
        check_point(entry['all']['point'],ranks['P01'],ranks[name],all_mask)
        gain=check_point(entry['development']['point'],ranks['P01'],ranks[name],dev);signs=[]
        assert set(entry['source'])==set(labels.source) and set(entry['query_mode'])==set(modes)
        for source,value in entry['source'].items():signs.append(check_point(value['point'],ranks['P01'],ranks[name],labels.source.eq(source).to_numpy()))
        for mode,value in entry['query_mode'].items():signs.append(check_point(value['point'],ranks['P01'],ranks[name],modes==mode))
        intersections={s+' / '+m:labels.source.eq(s).to_numpy()&(modes==m) for s in set(labels.source) for m in set(modes)}
        intersections={k:v for k,v in intersections.items() if v.any()};assert set(intersections)==set(entry['source_mode'])
        for key,mask in intersections.items():check_point(entry['source_mode'][key]['point'],ranks['P01'],ranks[name],mask)
        a,b=ranks['P01'],ranks[name]
        assert entry['recovered_truths']==int(((a==0)&(b>0)).sum()) and entry['displaced_truths']==int(((a>0)&(b==0)).sum())
        assert entry['truth_absent_queries']==int((b==0).sum())
        if name in ['cpu','gpu']:
            check_point(development['results'][name],a,b,dev);admissions[name]=gain>0 and all(v>=0 for v in signs)
            assert admissions[name]==entry['admission']['passed']
            assert all(a[0]==b[0] and set(a)==set(b) for a,b in zip(lists['P01'],lists[name]))
    assert all(a[:2]==b[:2] for a,b in zip(lists['P01'],lists['gpu']))
    assert report['passed_pair']==all(admissions.values()) and selected['both_selected'] and not report['C13_read']
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),quality_results_sha256=sha(OUT/'source-results.json'),
        selected_pair_sha256=sha(OUT/'selected-pair-freeze.json'),queries=352,acquisitions=1292,exact_rank_vectors=True,
        exact_fraction_points=True,all_source_mode_denominators_exact=True,absent_truths_zero=True,GPU_top2_exact=True,
        candidate_membership_changes=0,recovered_displaced_accounting_exact=True,admissions=admissions,passed_pair=all(admissions.values()),C13_read=False)
    import json
    target.write_text(json.dumps(result,indent=2),encoding='utf8');print('C46 INDEPENDENT QUALITY AUDIT',json.dumps(result))


if __name__=='__main__':main()
