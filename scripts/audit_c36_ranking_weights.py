"""Check frozen CPU fits with augmented least squares and independent ranks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from prepare_c36_validation import ROOT,OUT,CONFIG,sha,write,load_blocks,verify
from validation_repair import clustered_summary


def independent_ridge(blocks,train_ids,columns,l2):
    deltas=[];weights=[]
    for b in blocks:
        if b['mid'] not in train_ids or b['truth'] not in b['keys']:continue
        truth=b['keys'].index(b['truth'])
        wrong=[j for j in range(len(b['keys'])) if j!=truth and b['formulas'][j]==b['formulas'][truth]]
        for j in wrong:
            deltas.append(b['x'][truth,columns]-b['x'][j,columns]);weights.append(1/len(wrong))
    d=np.asarray(deltas);w=np.asarray(weights);w=w/w.sum()
    matrix=np.concatenate([np.sqrt(w)[:,None]*d,np.sqrt(l2)*np.eye(len(columns))])
    targets=np.r_[np.sqrt(w),np.zeros(len(columns))]
    beta,_,_,_=np.linalg.lstsq(matrix,targets,rcond=None)
    return beta


def independent_order(b,beta,columns,alpha,protect):
    result=list(range(len(b['keys'])))
    if alpha==0 or b['library']>=protect:return result
    score=b['x'][:,columns]@beta
    for formula in set(b['formulas']):
        positions=np.asarray([i for i,f in enumerate(b['formulas']) if f==formula])
        before=1/(positions+4.);new=score[positions]
        before=(before-before.mean())/max(float(before.std()),1e-8)
        new=(new-new.mean())/max(float(new.std()),1e-8)
        utility=(1-alpha)*before+alpha*new
        order=positions[np.lexsort((positions,-utility))]
        for destination,index in zip(positions,order):result[int(destination)]=int(index)
    return result


def main():
    verify();assert not (OUT/'independent-weight-prediction-audit.json').exists()
    blocks,names=load_blocks();primary=[b for b in blocks if b['cohort']=='primary'];stress=[b for b in blocks if b['cohort']=='source_stress']
    cfg=json.loads(CONFIG.read_text());fit=json.loads((OUT/'fit-audits.json').read_text())
    source=json.loads((OUT/'ranking-input-freeze.json').read_text());report=json.loads((OUT/'ranking-results.json').read_text())
    max_weight_error=0.;checked_models=0;checked_predictions=0;vectors={}
    for cohort,data,kind in [('primary',primary,'outer'),('source_stress',stress,'source_stress')]:
        for variant,columns in source['feature_sets'].items():
            frame=pd.read_csv(OUT/(cohort+'-'+variant+'.csv'),dtype=str).set_index('molecule_id')
            assignments=fit[kind][variant] if kind=='outer' else [fit[kind][variant]]
            values=[]
            for receipt in assignments:
                recalculated=independent_ridge(primary,set(receipt['train_ids']),columns,cfg['ridge_l2'])
                beta=np.asarray(receipt['beta']);err=float(np.abs(recalculated-beta).max())
                assert err<1e-10;max_weight_error=max(max_weight_error,err);checked_models+=1
            for b in data:
                receipt=next(r for r in assignments if b['mid'] in r['test_ids']) if kind=='outer' else assignments[0]
                indices=independent_order(b,np.asarray(receipt['beta']),columns,receipt['alpha'],cfg['protect_library_at'])
                expected=[b['smiles'][i] for i in indices];actual=frame.loc[b['mid'],'smiles'].split(';')
                assert expected==actual and set(expected)==set(b['smiles'])
                assert [b['formulas'][i] for i in indices]==b['formulas']
                keys=[b['keys'][i] for i in indices]
                values.append(1/(keys.index(b['truth'])+1) if b['truth'] in keys else 0.)
                checked_predictions+=1
            values=np.asarray(values);target=report['results' if kind=='outer' else 'source_stress'][variant]['all']['candidate_mrr25']
            assert abs(float(values.mean())-target)<1e-12
            vectors[(cohort,variant)]=values
    ablation={}
    for cohort,data in [('primary',primary),('source_stress',stress)]:
        full=vectors[(cohort,'pairwise_ridge_full')];control=vectors[(cohort,'pairwise_ridge_without_physical')]
        masks={'all':np.ones(len(data),bool)}
        if cohort=='source_stress':masks['raw_alias_free']=np.asarray([not b['raw_alias'] for b in data])
        ablation[cohort]={name:clustered_summary(control[mask],full[mask],np.asarray([b['group'] for b in data])[mask],cfg['bootstrap_draws'],cfg['bootstrap_seed']) for name,mask in masks.items()}
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),auditor_source_sha256=sha(__file__),
       fit_audits_sha256=sha(OUT/'fit-audits.json'),ranking_report_sha256=sha(OUT/'ranking-results.json'),
       independently_reconstructed_models=checked_models,maximum_weight_abs_difference=max_weight_error,
       independently_reconstructed_prediction_rows=checked_predictions,all_model_prediction_lists_exact=True,
       independent_metrics_exact=True,formula_slots_and_membership_exact=True,physical_feature_ablation=ablation,
       limitation='Matched feature ablation includes each variant\'s independently inner-selected alpha. This measures the full fixed calibration procedures, not an isolated causal effect of one physical score.',
       model_or_recipe_selection=False,gpu_work=False,fresh_reserved_labels_read=False)
    write(OUT/'independent-weight-prediction-audit.json',result);print('DONE C36 INDEPENDENT WEIGHT AND RANK AUDIT',json.dumps(result),flush=True)


if __name__=='__main__':main()
