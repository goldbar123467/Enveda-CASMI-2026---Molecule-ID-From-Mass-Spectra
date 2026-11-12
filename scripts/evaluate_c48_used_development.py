"""Evaluate all original C39 queries as USED development, before fresh outcomes.

No source cohort is filtered and none of the prior target/source/mixed guards
is removed. C48 auxiliary predictions/outcomes and C13 are not accessed here.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha
from evaluate_c45_complete import point
from validation_repair import clustered_summary
OUT=ROOT/'outputs/local/c48-clean-residual-heads-20261112'
C45=ROOT/'outputs/local/c45-complete-parent-validation-20261112'
C39=ROOT/'outputs/local/c39-source-supported-validation-20261111'
CONFIG=ROOT/'configs/c48-clean-residual-pair-20261112.json'


def main():
    assert not (OUT/'used-development-results.json').exists() and not (OUT/'used-development-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    family=read(OUT/'complete-candidate-freeze.json');assert family['config_sha256']==sha(CONFIG)
    head=ROOT/'outputs/publish/c48-clean-residual-heads-20261112';assert family['head_manifest_sha256']==sha(head/'head-manifest.json')
    manifest=read(head/'head-manifest.json')
    for name,pin in manifest['files'].items():assert sha(head/name)==pin
    files=[CONFIG,OUT/'complete-candidate-freeze.json',head/'head-manifest.json',__import__('pathlib').Path(__file__)]
    for entry in family['candidates'].values():
        folder=ROOT/entry['path'];assert sha(folder/'notebook.ipynb')==entry['notebook_sha256'] and sha(folder/'production-pin.json')==entry['pin_sha256']
        files.extend([folder/'notebook.ipynb',folder/'production-pin.json'])
    folder=OUT/'C39-predictions';receipt=read(folder/'score-receipt.json');audit=read(folder/'independent-ranking-audit.json')
    assert receipt['baseline_sha256']==sha(C45/'P16.csv') and receipt['query_sha256']==sha(C45/'queries.parquet')
    assert receipt['head_manifest_sha256']==family['head_manifest_sha256'] and audit['score_receipt_sha256']==sha(folder/'score-receipt.json')
    for name,pin in receipt['files_sha256'].items():assert sha(folder/name)==pin;files.append(folder/name)
    for trial in ['cpu','gpu']:assert audit['trials'][trial]['independent_RRF_exact']
    original=read(C39/'validation-freeze.json');assert original['files_sha256'][str((C39/'labels.csv').relative_to(ROOT))]==sha(C39/'labels.csv')
    parent_freeze=read(C45/'complete-parent-freeze.json')
    for name,pin in parent_freeze['files_sha256'].items():assert sha(C45/name)==pin
    files.extend([C45/'complete-parent-freeze.json',C45/'P01.csv',C45/'P16.csv',C45/'P14.csv',C45/'queries.parquet',C45/'quality-identity-map.parquet',folder/'score-receipt.json',folder/'independent-ranking-audit.json'])
    write(OUT/'used-development-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        files_sha256={str(path.relative_to(ROOT)):sha(path) for path in files},all_predictions_frozen=True,C39_is_used_development=True,
        new_source_outcomes_read=False,C13_read=False,maximum_scored_entries=2))
    frames={name:pd.read_csv(C45/(name+'.csv'),dtype=str,keep_default_na=False) for name in ['P01','P16','P14']}
    for trial in ['cpu','gpu']:frames[trial]=pd.read_csv(folder/(trial+'.csv'),dtype=str,keep_default_na=False)
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==352
    labels=pd.read_csv(C39/'labels.csv',dtype=str).set_index('molecule_id').loc[mids]
    mapping=pd.read_parquet(C45/'quality-identity-map.parquet').set_index('smiles').key.to_dict()
    current=pd.read_parquet(folder/'candidate-identities.parquet')
    for row in current.itertuples(index=False):assert mapping[row.smiles]==row.key
    lists={};ranks={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[[mapping[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in lists[name])
        ranks[name]=np.asarray([row.index(truth)+1 if truth in row else 0 for row,truth in zip(lists[name],labels.key)],np.int16)
    for trial in ['cpu','gpu']:
        prefix=1 if trial=='cpu' else 2
        assert all(a[:prefix]==b[:prefix] and set(a)==set(b) for a,b in zip(lists['P16'],lists[trial]))
    q=pd.read_parquet(C45/'queries.parquet');assert len(q)==1292 and not {'smiles','normalized_smiles','key','formula','inchikey14'}&set(q.columns)
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict()=={'positive':176,'mixed':126,'negative':50}
    all_mask=np.ones(352,bool);old_dev=labels.partition.eq('development').to_numpy();assert old_dev.sum()==192
    groups=labels.group.to_numpy();assert len(set(groups))==273
    def summary(base,name,mask):
        a=np.divide(1.,ranks[base],out=np.zeros(352),where=ranks[base]>0);b=np.divide(1.,ranks[name],out=np.zeros(352),where=ranks[name]>0)
        item=clustered_summary(a[mask],b[mask],groups[mask],10000,20261148);item['point']=point(ranks[base],ranks[name],mask);return item
    def compare(base,name):
        return dict(all=summary(base,name,all_mask),original_development192=summary(base,name,old_dev),
            source={s:summary(base,name,labels.source.eq(s).to_numpy()) for s in sorted(set(labels.source))},
            query_mode={m:summary(base,name,modes==m) for m in sorted(set(modes))},
            source_mode={s+' / '+m:summary(base,name,labels.source.eq(s).to_numpy()&(modes==m)) for s in sorted(set(labels.source)) for m in sorted(set(modes)) if (labels.source.eq(s).to_numpy()&(modes==m)).any()},
            recovered_truths=int(((ranks[base]==0)&(ranks[name]>0)).sum()),displaced_truths=int(((ranks[base]>0)&(ranks[name]==0)).sum()),truth_absent_queries=int((ranks[name]==0).sum()),
            rank1_changes=sum(a[0]!=b[0] for a,b in zip(lists[base],lists[name])),membership_changes=sum(set(a)!=set(b) for a,b in zip(lists[base],lists[name])))
    def admit(value):
        # Preserve the original strict192 gate as well as the prospectively
        # required strict352 gate. Adding this stronger check precedes outcomes.
        return value['all']['point']['strictly_positive'] and value['original_development192']['point']['strictly_positive'] and all(v['point']['nonnegative'] for field in ['source','query_mode'] for v in value[field].values())
    results={}
    for trial in ['cpu','gpu']:
        complete=compare('P01',trial);incremental=compare('P16',trial)
        assert incremental['recovered_truths']==incremental['displaced_truths']==incremental['rank1_changes']==incremental['membership_changes']==0
        results[trial]=dict(complete=complete,incremental=incremental,admission=dict(passed=admit(complete) and admit(incremental),statistical_promotion=False))
    np.savez(OUT/'used-development-ranks.npz',**ranks)
    record=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),config_sha256=sha(CONFIG),
        evaluation_freeze_sha256=sha(OUT/'used-development-freeze.json'),queries=352,acquisitions=1292,results=results,P14_control=compare('P01','P14'),
        passed_pair=all(v['admission']['passed'] for v in results.values()),all_queries_retained=True,absent_truths_zero=True,
        actual_query_modes=pd.Series(modes).value_counts().to_dict(),C39_is_used_development=True,new_source_outcomes_read=False,C13_read=False,
        limitation='Original C39 outcomes were known before this new design. These are adaptive development guards, not fresh source confirmation. All old source and mixed-mode gates are still required. Small positive-only auxiliary data will not establish target calibration.')
    write(OUT/'used-development-results.json',record)
    print('C48 USED DEVELOPMENT GUARDS',json.dumps({trial:dict(admission=result['admission'],complete_all=result['complete']['all']['point'],
        sources={s:v['point']['delta'] for s,v in result['complete']['source'].items()},modes={m:v['point']['delta'] for m,v in result['complete']['query_mode'].items()}) for trial,result in results.items()}))


if __name__=='__main__':main()
