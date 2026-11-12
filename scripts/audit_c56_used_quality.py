"""Independent exact official ranks and every unchanged USED-panel guard."""
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp
from audit_c45_quality import check_point
OUT=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'


def main():
    target=OUT/'independent-used-quality-audit.json';assert not target.exists()
    family=read(OUT/'complete-candidate-freeze.json');assert family['policy_sha256']==sha(ROOT/'configs/c56-bounded-semantic-pair-20261112.json')
    admissions={};reports={}
    for panel,n,base,label_file,result_file,rank_file,freeze_file in [
        ('C39',352,ROOT/'outputs/local/c45-complete-parent-validation-20261112',ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv','used-development-results.json','used-development-ranks.npz','used-development-freeze.json'),
        ('C51',35,ROOT/'outputs/local/c51-complete-auxiliary-20261112',ROOT/'outputs/local/c48-auxiliary-holdout-20261112/labels.csv','used-auxiliary-results.json','used-auxiliary-ranks.npz','used-auxiliary-freeze.json')]:
        result=read(OUT/result_file);frozen=read(OUT/freeze_file)
        assert result['evaluation_freeze_sha256']==sha(OUT/freeze_file)
        for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
        folder=OUT/(panel+'-predictions');receipt=read(folder/'score-receipt.json');audit=read(folder/'independent-ranking-audit.json')
        assert receipt['head_manifest_sha256']==family['head_manifest_sha256'] and audit['score_receipt_sha256']==sha(folder/'score-receipt.json')
        identities=pd.read_parquet(folder/'candidate-identities.parquet').set_index('smiles').key.to_dict()
        frames={name:pd.read_csv(base/(name+'.csv'),dtype=str,keep_default_na=False) for name in ['P01','P16']}
        for trial in ['cpu','gpu']:frames[trial]=pd.read_csv(folder/(trial+'.csv'),dtype=str,keep_default_na=False)
        mids=frames['P01'].molecule_id.tolist();assert len(mids)==len(set(mids))==n
        labels=pd.read_csv(label_file,dtype=str).set_index('molecule_id').loc[mids];ranks={};lists={};stored=np.load(OUT/rank_file)
        for name,frame in frames.items():
            assert frame.molecule_id.tolist()==mids;lists[name]=[[identities[s] for s in text.split(';')] for text in frame.smiles]
            ranks[name]=np.asarray([{key:i for i,key in enumerate(row,1)}.get(truth,0) for row,truth in zip(lists[name],labels.key)],np.int16)
            np.testing.assert_array_equal(ranks[name],stored[name])
        queries=pd.read_parquet(base/'queries.parquet');modes=queries.groupby('molecule_id').ionization_mode.agg(lambda a:'mixed' if a.nunique()>1 else a.iloc[0]).loc[mids].to_numpy()
        assert len(queries)==(1292 if panel=='C39' else 35)
        def verify(value,base_name,trial):
            all_mask=np.ones(n,bool);signs=[check_point(value['all']['point'],ranks[base_name],ranks[trial],all_mask)]
            assert set(value['source'])==set(labels.source)
            for source,entry in value['source'].items():signs.append(check_point(entry['point'],ranks[base_name],ranks[trial],labels.source.eq(source).to_numpy()))
            if panel=='C39':
                assert set(value['query_mode'])==set(modes)
                for mode,entry in value['query_mode'].items():signs.append(check_point(entry['point'],ranks[base_name],ranks[trial],modes==mode))
                dev=check_point(value['original_development192']['point'],ranks[base_name],ranks[trial],labels.partition.eq('development').to_numpy())
                assert labels.partition.eq('development').sum()==192
                assert set(value['source_mode'])=={s+' / '+m for s in set(labels.source) for m in set(modes) if (labels.source.eq(s).to_numpy()&(modes==m)).any()}
                for name,entry in value['source_mode'].items():
                    source,mode=name.split(' / ');check_point(entry['point'],ranks[base_name],ranks[trial],labels.source.eq(source).to_numpy()&(modes==mode))
            assert value['recovered_truths']==int(((ranks[base_name]==0)&(ranks[trial]>0)).sum())
            assert value['displaced_truths']==int(((ranks[base_name]>0)&(ranks[trial]==0)).sum())
            assert value['truth_absent_queries']==int((ranks[trial]==0).sum())
            return all(s>=0 for s in signs) and (panel!='C39' or (signs[0]>0 and dev>0))
        reports[panel]={}
        for trial,entry in result['results'].items():
            passed=verify(entry['complete'],'P01',trial)&verify(entry['incremental'],'P16',trial)
            assert passed==entry['admission']['passed']
            for a,b in zip(lists['P16'],lists[trial]):assert a[:2]==b[:2] and a[10:]==b[10:] and set(a)==set(b)
            reports[panel][trial]=bool(passed)
        assert result['passed_pair']==all(reports[panel].values());admissions[panel]=all(reports[panel].values())
    combined=all(admissions.values());assert combined==read(OUT/'used-auxiliary-results.json')['combined_used_development_pair_passed']
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),complete_candidate_freeze_sha256=sha(OUT/'complete-candidate-freeze.json'),
        old352_result_sha256=sha(OUT/'used-development-results.json'),used35_result_sha256=sha(OUT/'used-auxiliary-results.json'),
        used_queries=387,used_acquisitions=1327,exact_official_rank_vectors=True,exact_fraction_arithmetic=True,
        all_original_source_and_actual_mode_guards_retained=True,absent_truths_zero=True,recovered_displaced_counts_exact=True,
        first2_and_beyond10_exact=True,admissions=reports,passed_pair=combined,C53_quality_read=False,C13_read=False,statistical_promotion=False))
    print('C56 INDEPENDENT ALL387USED QUALITY AUDIT',reports,'PAIR',combined,flush=True)


if __name__=='__main__':main()
