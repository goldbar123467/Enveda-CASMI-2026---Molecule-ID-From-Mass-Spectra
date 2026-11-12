"""Check the now-USED C51 panel; never access new C53 or C13 outcomes."""
import numpy as np
import pandas as pd
from pathlib import Path
from c45_complete_parent import ROOT,read,write,sha,stamp
from evaluate_c45_complete import point
from audit_c45_quality import check_point
from validation_repair import clustered_summary
OUT=ROOT/'outputs/local/c54-isomer-fingerprint-heads-20261112'
BASE=ROOT/'outputs/local/c51-complete-auxiliary-20261112'


def main():
    assert not (OUT/'used-auxiliary-results.json').exists()
    assert not (ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    family=read(OUT/'complete-candidate-freeze.json');assert family['policy_sha256']==sha(ROOT/'configs/c54-isomer-fingerprint-pair-20261112.json')
    folder=OUT/'C51-predictions';receipt=read(folder/'score-receipt.json');audit=read(folder/'independent-ranking-audit.json')
    assert receipt['head_manifest_sha256']==family['head_manifest_sha256'] and set(receipt['trials'])=={'cpu','gpu'}
    assert receipt['baseline_sha256']==sha(BASE/'P16.csv') and receipt['query_sha256']==sha(BASE/'queries.parquet')
    assert audit['score_receipt_sha256']==sha(folder/'score-receipt.json')
    files=[Path(__file__),OUT/'complete-candidate-freeze.json',BASE/'P01.csv',BASE/'P16.csv',BASE/'queries.parquet',folder/'score-receipt.json',folder/'independent-ranking-audit.json']
    for name,pin in receipt['files_sha256'].items():assert sha(folder/name)==pin;files.append(folder/name)
    write(OUT/'used-auxiliary-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        panel_role='All35C51queries are now USED development; not a new source confirmation',C53_quality_read=False,C13_read=False))
    frames={n:pd.read_csv(BASE/(n+'.csv'),dtype=str,keep_default_na=False) for n in ['P01','P16']}
    for trial in ['cpu','gpu']:frames[trial]=pd.read_csv(folder/(trial+'.csv'),dtype=str,keep_default_na=False)
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==35
    panel=ROOT/'outputs/local/c48-auxiliary-holdout-20261112';label_path=panel/'labels.csv'
    assert read(panel/'validation-freeze.json')['files_sha256'][str(label_path.relative_to(ROOT))]==sha(label_path)
    labels=pd.read_csv(label_path,dtype=str).set_index('molecule_id').loc[mids]
    identity=pd.read_parquet(folder/'candidate-identities.parquet').set_index('smiles').key.to_dict();lists={};ranks={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[identity[s] for s in text.split(';')] for text in frame.smiles]
        ranks[name]=np.asarray([row.index(truth)+1 if truth in row else 0 for row,truth in zip(lists[name],labels.key)],np.int16)
        # Independent lookup reconstruction catches off-by-one/absent-zero errors.
        rebuilt=np.asarray([{key:j for j,key in enumerate(row,1)}.get(truth,0) for row,truth in zip(lists[name],labels.key)])
        np.testing.assert_array_equal(ranks[name],rebuilt)
    def summary(base,trial,mask):
        a=np.divide(1.,ranks[base],out=np.zeros(35),where=ranks[base]>0);b=np.divide(1.,ranks[trial],out=np.zeros(35),where=ranks[trial]>0)
        v=clustered_summary(a[mask],b[mask],labels.group.to_numpy()[mask],10000,20261154);v['point']=point(ranks[base],ranks[trial],mask)
        check_point(v['point'],ranks[base],ranks[trial],mask);return v
    all_mask=np.ones(35,bool);results={}
    for trial in ['cpu','gpu']:
        comparisons={}
        for name,base in [('complete','P01'),('incremental','P16')]:
            comparisons[name]=dict(all=summary(base,trial,all_mask),source={s:summary(base,trial,labels.source.eq(s).to_numpy()) for s in sorted(set(labels.source))},
                recovered_truths=int(((ranks[base]==0)&(ranks[trial]>0)).sum()),displaced_truths=int(((ranks[base]>0)&(ranks[trial]==0)).sum()),truth_absent_queries=int((ranks[trial]==0).sum()))
        assert all(a[0]==b[0] and set(a)==set(b) for a,b in zip(lists['P16'],lists[trial]))
        passed=all(v['all']['point']['nonnegative'] and all(s['point']['nonnegative'] for s in v['source'].values()) for v in comparisons.values())
        results[trial]=dict(**comparisons,admission=dict(passed=passed))
    np.savez(OUT/'used-auxiliary-ranks.npz',**ranks)
    old=read(OUT/'used-development-results.json');passed=old['passed_pair'] and all(v['admission']['passed'] for v in results.values())
    record=dict(completed_utc=stamp(),source_sha256=sha(__file__),evaluation_freeze_sha256=sha(OUT/'used-auxiliary-freeze.json'),
        queries=35,results=results,passed_pair=all(v['admission']['passed'] for v in results.values()),
        old352_result_sha256=sha(OUT/'used-development-results.json'),combined_used_development_pair_passed=passed,
        all35_retained=True,absent_truths_zero=True,C51_is_used_development=True,C53_quality_read=False,C13_read=False,statistical_promotion=False)
    write(OUT/'used-auxiliary-results.json',record)
    print('C54 ALL USED DEVELOPMENT DECISION',dict(original352=old['passed_pair'],used35=record['passed_pair'],combined=passed,
        delta={trial:v['complete']['all']['point']['delta'] for trial,v in results.items()}),flush=True)


if __name__=='__main__':main()
