"""Trace official-identity membership through retained parallel candidate routes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import functools,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import RDLogger
from prepare_c36_validation import ROOT,OUT,sha,write,verify,features


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');contract,_=verify()
    assert not (OUT/'candidate-losses.json').exists()
    metadata=pd.read_csv(OUT/'queries.csv',dtype={'group':str});metadata=metadata[metadata.cohort.eq('primary')]
    evidence=pd.read_parquet(OUT/'features.parquet');evidence=evidence[evidence.cohort.eq('primary')]
    cache=pd.read_parquet(OUT/'identity-cache.parquet');known=dict(zip(cache.smiles,cache.metric_key));invalid=set()
    @functools.lru_cache(maxsize=100000)
    def key(s):
        if s in known:return known[s]
        try:known[s]=features.structure(s)[0]
        except (ValueError,TypeError,RuntimeError):invalid.add(s);return None
        return known[s]
    routes={};finite_scores={}
    for mode in ['positive','negative']:
        def read(name):return json.loads((ROOT/contract['routes'][mode+':'+name]['path']).read_text())
        engine=read('eng_lists.json')
        if mode=='negative':engine.update(read('recovered_engine'))
        for name,data,field in [('engine2',engine,'smiles'),('pubchem',read('pc_lists.json'),'pc')]:
            for mid,row in data.items():
                routes.setdefault(mid,{})[name]={key(s) for s in row.get(field,[]) if key(s) is not None}
        for stage in ['ice','gl']+(['n1'] if mode=='positive' else []):
            for row in read(stage+'_work/'+stage+'_in.json'):
                routes.setdefault(str(row['mid']),{})[stage+'_input']={key(s) for s in row['cands'] if key(s) is not None}
        for stage in ['ice','gl']:
            for mid,row in read(stage+'_work/'+stage+'_out.json').items():
                values={}
                for s,v in row.items():
                    if v is not None and np.isfinite(v) and key(s) is not None:
                        k=key(s);values[k]=max(values.get(k,-np.inf),float(v))
                routes.setdefault(mid,{})[stage+'_scored']=set(values);finite_scores.setdefault(mid,{})[stage]=values
    deep=pd.read_parquet(ROOT/contract['inputs']['deep_pool']['path'])
    deep_keys={mid:set(g.metric_key) for mid,g in deep.groupby('molecule_id',sort=False)}
    parents={mid:g.sort_values('baseline_rank',kind='stable') for mid,g in evidence.groupby('molecule_id',sort=False)}
    names=['engine2','pubchem','ice_input','ice_scored','gl_input','gl_scored','n1_input']
    records=[]
    for r in metadata.itertuples(index=False):
        p=parents[r.molecule_id];keys=p.metric_key.tolist();route=routes.get(r.molecule_id,{})
        union=set(keys).union(*route.values()) if route else set(keys)
        final_rank=keys.index(r.truth)+1 if r.truth in keys else 0
        record=dict(molecule_id=r.molecule_id,mode=r.mode,final_truth_rank=final_rank,
          final_candidates=len(keys),retained_union_truth=r.truth in union,retained_union_size=len(union),
          deep_truth=r.truth in deep_keys.get(r.molecule_id,set()),deep_scope_available=r.molecule_id in deep_keys,
          neutral_mass_error_ppm=abs(r.query_neutral_mass-r.truth_mass)/r.truth_mass*1e6,
          loss='present_final' if final_rank else 'retrieved_but_final_dropped' if r.truth in union else 'absent_from_retained_routes')
        for name in names:
            record[name+'_available']=name in route;record[name+'_truth']=r.truth in route.get(name,set());record[name+'_candidates']=len(route.get(name,set()))
        record['physical_truth_in_input_but_not_scored']=any(record[s+'_input_truth'] and not record[s+'_scored_truth'] for s in ['ice','gl'])
        records.append(record)
    frame=pd.DataFrame(records);frame.to_csv(OUT/'candidate-loss-witnesses.csv',index=False)
    def summarize(f):
        absent=f.final_truth_rank.eq(0)
        return dict(queries=len(f),truths_final=int((~absent).sum()),truths_retained_union=int(f.retained_union_truth.sum()),
          dropped_after_retrieval=int((absent&f.retained_union_truth).sum()),absent_in_all_retained_routes=int((absent&~f.retained_union_truth).sum()),
          final_absent_with_neutral_mass_error_gt30ppm=int((absent&f.neutral_mass_error_ppm.gt(30)).sum()),
          physical_truth_input_without_score=int((absent&f.physical_truth_in_input_but_not_scored).sum()),
          by_route={n:dict(truths=int(f[n+'_truth'].sum()),final_absent_truths=int((absent&f[n+'_truth']).sum()),available_queries=int(f[n+'_available'].sum())) for n in names})
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),contract_sha256=sha(OUT/'input-contract.json'),
      feature_freeze_sha256=sha(OUT/'feature-freeze.json'),witness_sha256=sha(OUT/'candidate-loss-witnesses.csv'),
      groups={name:summarize(f) for name,f in [('all240',frame),('positive160',frame[frame['mode'].eq('positive')]),('negative80',frame[frame['mode'].eq('negative')])]},
      positive_deep_truths=int(frame[frame['mode'].eq('positive')].deep_truth.sum()),
      positive_final_absent_in_deep=int((frame['mode'].eq('positive')&frame.final_truth_rank.eq(0)&frame.deep_truth).sum()),
      invalid_route_structures=len(invalid),official_identity_join=True,reserved_labels_read=False,gpu_work=False,
      inference_coverage_caveat='Raw forward output membership is not evidence of supported negative-mode scoring: production negative final features have no ICE/GL signal. Scores were not newly run or used to modify candidates.',
      limitation='Parallel-route membership witnesses locate observed losses. A missing truth in retained route files does not prove absence from the original full databases; unretained mass filters and candidate budgets cannot be reconstructed from these files alone.')
    assert report['groups']['all240']['truths_final']==150
    assert report['positive_deep_truths']==119 and report['positive_final_absent_in_deep']==19
    write(OUT/'candidate-losses.json',report);print('DONE C36 CANDIDATE LOSS AUDIT',json.dumps(report),flush=True)


if __name__=='__main__':main()
