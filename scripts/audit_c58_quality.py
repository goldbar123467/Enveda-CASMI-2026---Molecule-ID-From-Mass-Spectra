"""Independent exact rank, cohort, zero and gate audit after C51 consumption."""
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp
from audit_c45_quality import check_point
OUT=ROOT/'outputs/local/c58-complete-target-20261112'


def main():
    target=OUT/'independent-quality-audit.json';assert not target.exists()
    result=read(OUT/'source-results.json');frozen=read(OUT/'evaluation-freeze.json')
    assert result['evaluation_freeze_sha256']==sha(OUT/'evaluation-freeze.json')
    assert result['source_sha256']==frozen['source_sha256']==sha(ROOT/'scripts/evaluate_c58_target.py')
    assert result['consumption_sha256']==sha(ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json')
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin
    panel=ROOT/'outputs/local/c53-target-holdout-20261112';labels_path=panel/'labels.csv'
    assert read(panel/'validation-freeze.json')['files_sha256'][str(labels_path.relative_to(ROOT))]==sha(labels_path)
    paths=dict(P01=OUT/'P01.csv',P16=OUT/'P16.csv',cpu=OUT/'head-predictions/cpu.csv',gpu=OUT/'head-predictions/gpu.csv')
    frames={n:pd.read_csv(p,dtype=str,keep_default_na=False) for n,p in paths.items()};mids=frames['P01'].molecule_id.tolist()
    assert len(mids)==len(set(mids))==482;labels=pd.read_csv(labels_path,dtype=str).set_index('molecule_id').loc[mids]
    assert labels.source.value_counts().to_dict()=={'enveda-180':384,'enveda-np-examples':98}
    assert len(set(labels.group))==447
    identities=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict();lists={};ranks={}
    stored=np.load(OUT/'source-ranks.npz')
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[identities[s] for s in text.split(';')] for text in frame.smiles]
        ranks[name]=np.asarray([{k:i for i,k in enumerate(row,1)}.get(truth,0) for row,truth in zip(lists[name],labels.key)],np.int16)
        np.testing.assert_array_equal(ranks[name],stored[name])
    q=pd.read_parquet(OUT/'queries.parquet');assert len(q)==2125
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda a:'mixed' if a.nunique()>1 else a.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict()=={'positive':307,'mixed':141,'negative':34}
    all_mask=np.ones(482,bool)
    def check(value,base,name):
        signs=[check_point(value['all']['point'],ranks[base],ranks[name],all_mask)]
        assert set(value['source'])==set(labels.source)
        for source,entry in value['source'].items():signs.append(check_point(entry['point'],ranks[base],ranks[name],labels.source.eq(source).to_numpy()))
        assert set(value['query_mode'])==set(modes)
        for mode,entry in value['query_mode'].items():signs.append(check_point(entry['point'],ranks[base],ranks[name],modes==mode))
        assert set(value['source_mode'])=={s+' / '+m for s in set(labels.source) for m in set(modes) if (labels.source.eq(s).to_numpy()&(modes==m)).any()}
        for key,entry in value['source_mode'].items():
            source,mode=key.split(' / ');check_point(entry['point'],ranks[base],ranks[name],labels.source.eq(source).to_numpy()&(modes==mode))
        assert value['recovered_truths']==int(((ranks[base]==0)&(ranks[name]>0)).sum())
        assert value['displaced_truths']==int(((ranks[base]>0)&(ranks[name]==0)).sum())
        assert value['truth_absent_queries']==int((ranks[name]==0).sum())
        assert value['rank1_changes']==sum(a[0]!=b[0] for a,b in zip(lists[base],lists[name]))
        assert value['candidate_membership_changes']==sum(set(a)!=set(b) for a,b in zip(lists[base],lists[name]))
        return signs[0]>0 and all(s>=0 for s in signs)
    admissions={}
    for trial,value in result['results'].items():
        admissions[trial]=check(value['complete'],'P01',trial)&check(value['incremental'],'P16',trial)
        assert admissions[trial]==value['admission']['passed']
        assert all(a[:2]==b[:2] and a[10:]==b[10:] and set(a)==set(b) for a,b in zip(lists['P16'],lists[trial]))
    check(result['parent_overlay'],'P01','P16')
    assert result['passed_pair']==all(admissions.values()) and result['all_queries_retained'] and result['absent_truths_zero']
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),quality_results_sha256=sha(OUT/'source-results.json'),
        evaluation_freeze_sha256=sha(OUT/'evaluation-freeze.json'),queries=482,acquisitions=2125,chemistry_groups=447,
        exact_official_rank_vectors=True,exact_fraction_points=True,all_source_denominators_exact=True,
        absent_truths_zero=True,recovered_displaced_counts_exact=True,admissions=admissions,passed_pair=all(admissions.values()),C13_read=False))
    print('C58 INDEPENDENT QUALITY AUDIT',admissions,flush=True)


if __name__=='__main__':main()
