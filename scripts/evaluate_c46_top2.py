"""Freeze one top2 GPU alternative, test development, then sealed source gates."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import argparse,importlib.util,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,OUT as C45,C39,read,write,sha
from evaluate_c45_complete import point
from validation_repair import clustered_summary
OUT=ROOT/'outputs/local/c46-top2-protected-pair-20261112'
POLICY=ROOT/'configs/c46-top2-protected-pair-20261112.json'


def stamp():return datetime.now(timezone.utc).isoformat()


def prepare():
    assert not (OUT/'prediction-freeze.json').exists() and not (OUT/'development-results.json').exists()
    assert not (C45/'source-results.json').exists() and not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    pair=read(OUT/'candidate-pair-freeze.json');assert pair['policy_sha256']==sha(POLICY)
    old=read(C45/'evaluation-freeze.json')
    for name,pin in old['files_sha256'].items():assert sha(ROOT/name)==pin
    origin=C45/'heads-P01';receipt=read(origin/'score-receipt.json')
    for name,pin in receipt['files_sha256'].items():assert sha(origin/name)==pin
    for trial,entry in pair['candidates'].items():
        folder=ROOT/entry['path'];assert sha(folder/'notebook.ipynb')==entry['notebook_sha256'] and sha(folder/'production-pin.json')==entry['pin_sha256']
        bundle=ROOT/entry['bundle'];manifest=read(bundle/'head-manifest.json');assert sha(bundle/'head-manifest.json')==entry['head_manifest_sha256']
        for name,pin in manifest['files'].items():assert sha(bundle/name)==pin
    parent=pd.read_csv(C45/'P01.csv',dtype=str,keep_default_na=False)
    identities=pd.read_parquet(origin/'candidate-identities.parquet').set_index('smiles')
    raw=pd.read_parquet(origin/'candidate-scores.parquet');raw=raw[raw.trial.eq('gpu')]
    coverage=pd.read_csv(origin/'query-coverage.csv').set_index('molecule_id').supported.to_dict()
    bundle=ROOT/pair['candidates']['gpu']['bundle'];spec=importlib.util.spec_from_file_location('c46_exact_deployed_tail',bundle/'crossmodal_tail.py')
    tail=importlib.util.module_from_spec(spec);spec.loader.exec_module(tail)
    grouped={m:g.sort_values('parent_rank') for m,g in raw.groupby('molecule_id')};answer=parent.copy();changed=0
    for i,row in parent.iterrows():
        strings=row.smiles.split(';');group=grouped[row.molecule_id];assert group.smiles.tolist()==strings
        formula=identities.loc[strings,'formula'].tolist();eligible=group.eligible.to_numpy(bool);scores=group.score.to_numpy(float)
        actual=tail.order(formula,scores,eligible,bool(coverage[row.molecule_id]));expected=list(range(len(strings)))
        # Independent direct RRF reconstruction excludes original slots1 AND2.
        if coverage[row.molecule_id]:
            for f in set(formula[2:]):
                pos=np.asarray([j for j in range(2,len(strings)) if formula[j]==f])
                if len(pos)<2 or not eligible[pos].all():continue
                preference=pos[np.lexsort((pos,-scores[pos]))];head_rank={int(j):k for k,j in enumerate(preference,1)}
                utility=np.asarray([1/(3+j+1)+.2/(3+head_rank[int(j)]) for j in pos]);chosen=pos[np.lexsort((pos,-utility))]
                for slot,choice in zip(pos,chosen):expected[int(slot)]=int(choice)
        assert actual==expected and actual[:2]==list(range(min(2,len(strings))))
        result=[strings[j] for j in actual];assert [identities.loc[s,'formula'] for s in result]==formula
        keys=identities.loc[result,'key'].tolist();assert len(set(keys))==len(keys)
        answer.at[i,'smiles']=';'.join(result);changed+=result!=strings
    for dest,source in [('P01.csv',C45/'P01.csv'),('P14.csv',C45/'P14.csv'),('queries.parquet',C45/'queries.parquet'),
        ('quality-identity-map.parquet',C45/'quality-identity-map.parquet'),('cpu.csv',origin/'cpu.csv')]:shutil.copy2(source,OUT/dest)
    answer.to_csv(OUT/'gpu.csv',index=False)
    write(OUT/'ranking-audit.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),origin_score_receipt_sha256=sha(origin/'score-receipt.json'),
        deployed_rule_equals_independent_RRF=True,GPU_top2_exact=True,changed_GPU_lists=int(changed),queries=352,
        candidate_membership_changes=0,formula_slot_changes=0,official_identity_duplicates=0,
        encoder_or_model_refit=False,raw_scores_reused_exactly=True,quality_outcomes_read=False))
    files=[OUT/n for n in ['candidate-pair-freeze.json','P01.csv','P14.csv','queries.parquet','quality-identity-map.parquet','cpu.csv','gpu.csv','ranking-audit.json']]
    files += [Path(__file__),POLICY,C45/'evaluation-freeze.json',origin/'score-receipt.json',origin/'candidate-scores.parquet',origin/'candidate-identities.parquet',origin/'query-coverage.csv',ROOT/'scripts/audit_c46_quality.py']
    for entry in pair['candidates'].values():
        folder=ROOT/entry['path'];bundle=ROOT/entry['bundle'];files += [folder/n for n in ['notebook.ipynb','production-pin.json','kernel-metadata.json']]
        files += [bundle/'head-manifest.json']+[bundle/name for name in read(bundle/'head-manifest.json')['files']]
    write(OUT/'prediction-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},all_predictions_frozen=True,source_outcomes_read=False,C13_read=False))
    print('C46 FROZEN NEW GPU LISTS',changed)


def verify():
    frozen=read(OUT/'prediction-freeze.json');assert frozen['source_sha256']==sha(__file__) and frozen['policy_sha256']==sha(POLICY)
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin,name
    pair=read(OUT/'candidate-pair-freeze.json');assert pair['policy_sha256']==sha(POLICY)
    panel=read(C39/'validation-freeze.json');assert sha(C39/'labels.csv')==panel['files_sha256'][str((C39/'labels.csv').relative_to(ROOT))]
    frames={name:pd.read_csv(OUT/(name+'.csv'),dtype=str,keep_default_na=False) for name in ['P01','P14','cpu','gpu']}
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==352
    labels=pd.read_csv(C39/'labels.csv',dtype=str).set_index('molecule_id').loc[mids]
    identities=pd.read_parquet(OUT/'quality-identity-map.parquet').set_index('smiles').key.to_dict();keys={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;keys[name]=[[identities[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1<=len(row)<=25 and len(set(row))==len(row) for row in keys[name])
    for trial in ['cpu','gpu']:
        assert all(a[0]==b[0] and set(a)==set(b) for a,b in zip(keys['P01'],keys[trial]))
    assert all(a[:2]==b[:2] for a,b in zip(keys['P01'],keys['gpu']))
    q=pd.read_parquet(OUT/'queries.parquet');assert len(q)==1292
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict()=={'positive':176,'mixed':126,'negative':50}
    dev=labels.partition.eq('development').to_numpy();assert dev.sum()==192
    assert not set(labels.group[dev])&set(labels.group[~dev])
    return labels,keys,modes,dev


def ranks(labels,keys,positions):
    rr={name:np.zeros(352,np.int16) for name in keys}
    for i in positions:
        truth=labels.key.iloc[i]
        for name,rows in keys.items():rr[name][i]=rows[i].index(truth)+1 if truth in rows[i] else 0
    return rr


def development():
    assert not (OUT/'development-results.json').exists() and not (OUT/'source-results.json').exists()
    labels,keys,modes,dev=verify();rr=ranks(labels,keys,np.flatnonzero(dev))
    result={trial:point(rr['P01'],rr[trial],dev) for trial in ['cpu','gpu']}
    np.savez(OUT/'development-ranks.npz',**rr)
    write(OUT/'development-results.json',dict(completed_utc=stamp(),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),
        queries=192,results=result,both_pass=all(v['strictly_positive'] for v in result.values()),source_outcomes_read=False,C13_read=False))
    write(OUT/'selected-pair-freeze.json',dict(frozen_utc=stamp(),candidate_pair_sha256=sha(OUT/'candidate-pair-freeze.json'),
        prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),development_results_sha256=sha(OUT/'development-results.json'),
        development_ranks_sha256=sha(OUT/'development-ranks.npz'),both_selected=all(v['strictly_positive'] for v in result.values()),
        policy_sha256=sha(POLICY),source_sha256=sha(__file__),source_outcomes_read=False,C13_read=False))
    print('C46 DEVELOPMENT ONLY',json.dumps(result))


def source():
    selected=read(OUT/'selected-pair-freeze.json');assert selected['both_selected'] and not (OUT/'source-results.json').exists()
    for field,name in [('candidate_pair_sha256','candidate-pair-freeze.json'),('prediction_freeze_sha256','prediction-freeze.json'),
        ('development_results_sha256','development-results.json'),('development_ranks_sha256','development-ranks.npz')]:assert selected[field]==sha(OUT/name)
    labels,keys,modes,dev=verify();rr=ranks(labels,keys,np.arange(352));old=np.load(OUT/'development-ranks.npz')
    for name in rr:np.testing.assert_array_equal(rr[name][dev],old[name][dev])
    groups=labels.group.to_numpy();all_mask=np.ones(352,bool)
    def summary(name,mask):
        a=np.divide(1.,rr['P01'],out=np.zeros(352),where=rr['P01']>0);b=np.divide(1.,rr[name],out=np.zeros(352),where=rr[name]>0)
        result=clustered_summary(a[mask],b[mask],groups[mask],10000,20261146);result['point']=point(rr['P01'],rr[name],mask);return result
    results={}
    for name in ['cpu','gpu','P14']:
        record=dict(all=summary(name,all_mask),development=summary(name,dev),
            source={s:summary(name,labels.source.eq(s).to_numpy()) for s in sorted(set(labels.source))},
            query_mode={m:summary(name,modes==m) for m in sorted(set(modes))},
            source_mode={s+' / '+m:summary(name,labels.source.eq(s).to_numpy()&(modes==m)) for s in sorted(set(labels.source)) for m in sorted(set(modes)) if (labels.source.eq(s).to_numpy()&(modes==m)).any()},
            recovered_truths=int(((rr['P01']==0)&(rr[name]>0)).sum()),displaced_truths=int(((rr['P01']>0)&(rr[name]==0)).sum()),truth_absent_queries=int((rr[name]==0).sum()))
        if name in ['cpu','gpu']:
            passed=record['development']['point']['strictly_positive'] and all(v['point']['nonnegative'] for field in ['source','query_mode'] for v in record[field].values())
            assert record['recovered_truths']==record['displaced_truths']==0
            record['admission']=dict(passed=passed,statistical_promotion=False,reason=None if passed else 'predeclared_source_or_mode_directional_gate_failed')
        results[name]=record
    np.savez(OUT/'source-ranks.npz',**rr)
    write(OUT/'source-results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        selected_pair_sha256=sha(OUT/'selected-pair-freeze.json'),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),
        queries=352,acquisitions=1292,results=results,passed_pair=all(results[t]['admission']['passed'] for t in ['cpu','gpu']),
        all_queries_retained=True,absent_truths_zero=True,actual_query_modes=pd.Series(modes).value_counts().to_dict(),
        C13_read=False,source_recipe_selection=False,leaderboard_calibration_established=False,
        limitation='One development-motivated top2 GPU rule; controlled reference-masked in-corpus study, not a calibrated leaderboard proxy. Parent/pretrained exposure remains disclosed.'))
    print('C46 SOURCE GATES',json.dumps({t:results[t]['admission'] for t in ['cpu','gpu']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','development','source']);a=p.parse_args()
    {'prepare':prepare,'development':development,'source':source}[a.phase]()
