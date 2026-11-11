"""Nested chemistry-grouped CPU ranking and fixed source/mode stress checks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from prepare_c36_validation import ROOT,OUT,CONFIG,sha,write,verify,load_blocks
from validation_repair import (crossfit,choose_alpha,fit_pairwise,formula_order,
    evaluate_orders,clustered_summary,validation_gate)


def fixed_physical(block,names,cfg):
    positions=[names.index(s+'_z') for s in ['ice','gl']]
    available=[names.index(s+'_present') for s in ['ice','gl']]
    counts=block['x'][:,available].sum(1)
    score=block['x'][:,positions].sum(1)/np.maximum(counts,1)
    return formula_order(block,score,cfg['fixed_physical_alpha'],cfg['protect_library_at'],eligible=counts>0)


def main():
    pa.set_cpu_count(2);contract,frozen=verify();cfg=json.loads(CONFIG.read_text())
    assert (ROOT/'.local/compute-paused.json').exists()
    assert not (OUT/'ranking-results.json').exists() and not (OUT/'ranking-input-freeze.json').exists()
    blocks,names=load_blocks();primary=[b for b in blocks if b['cohort']=='primary'];external=[b for b in blocks if b['cohort']=='source_stress']
    assert len(primary)==240 and len(external)==80
    schema=json.loads((OUT/'feature-schema.json').read_text())
    feature_sets=dict(pairwise_ridge_full=list(range(len(names))),
        pairwise_ridge_without_physical=[i for i,n in enumerate(names) if n not in schema['physical_features']])
    write(OUT/'ranking-input-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),contract_sha256=sha(OUT/'input-contract.json'),
       feature_freeze_sha256=sha(OUT/'feature-freeze.json'),config_sha256=sha(CONFIG),feature_sets=feature_sets,
       primary_variant=cfg['primary_variant'],outer_metrics_read=False,stress_metrics_read=False,fresh_reserved_labels_read=False,gpu_work=False))
    predictions={'fixed_physical_agreement':[fixed_physical(b,names,cfg) for b in primary]};audits={};stress_predictions={};stress_audits={}
    outer={i:b['outer_fold'] for i,b in enumerate(primary)}
    for variant,columns in feature_sets.items():
        print('C36 NESTED CPU RANKING',variant,flush=True)
        predictions[variant],audits[variant]=crossfit(primary,columns,cfg,outer)
        external_groups={b['group'] for b in external}
        train=[i for i,b in enumerate(primary) if b['group'] not in external_groups]
        assert not (set(primary[i]['group'] for i in train)&external_groups)
        alpha,selection=choose_alpha(primary,train,columns,cfg)
        beta,receipt=fit_pairwise(primary,train,columns,cfg['ridge_l2'])
        receipt.update(alpha=alpha,selection=selection,beta=beta.tolist(),excluded_primary_for_stress=len(primary)-len(train))
        stress_audits[variant]=receipt
        stress_predictions[variant]=[formula_order(b,b['x'][:,columns]@beta,alpha,cfg['protect_library_at']) for b in external]
    stress_predictions['fixed_physical_agreement']=[fixed_physical(b,names,cfg) for b in external]
    # Persist every fixed prediction before reading any outer/stress score.
    files=[]
    for cohort,rows,variants in [('primary',primary,predictions),('source_stress',external,stress_predictions)]:
        for name,orders in variants.items():
            data=[]
            for b,order in zip(rows,orders):
                assert set(order)==set(range(len(b['keys'])))
                data.append(dict(molecule_id=b['mid'],smiles=';'.join(b['smiles'][i] for i in order)))
            path=OUT/(cohort+'-'+name+'.csv');pd.DataFrame(data).to_csv(path,index=False);files.append(path)
    write(OUT/'fit-audits.json',dict(outer=audits,source_stress=stress_audits,outer_labels_used_only_for_final_metrics=True,training_device='CPU',encoder_training=False))
    write(OUT/'prediction-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),input_freeze_sha256=sha(OUT/'ranking-input-freeze.json'),
       files_sha256={p.name:sha(p) for p in files},fit_audits_sha256=sha(OUT/'fit-audits.json'),outer_metrics_read=False,stress_metrics_read=False))
    def summary(rows,orders):
        base=evaluate_orders(rows,[list(range(len(b['keys']))) for b in rows]);after=evaluate_orders(rows,orders)
        def calculate(mask):
            return clustered_summary(base[mask],after[mask],np.asarray([b['group'] for b in rows])[mask],cfg['bootstrap_draws'],cfg['bootstrap_seed'])
        all_mask=np.ones(len(rows),bool);results=dict(all=calculate(all_mask),polarity={})
        for mode in ['positive','negative']:
            mask=np.asarray([b['mode']==mode for b in rows])
            if mask.any():results['polarity'][mode]=calculate(mask)
        if any(b['cohort']=='source_stress' for b in rows):
            mask=np.asarray([not b['raw_alias'] for b in rows]);results['raw_alias_free']=calculate(mask)
        changes=promotions=harm1=help1=0
        for b,o in zip(rows,orders):
            changes+=int(o!=list(range(len(o))));promotions+=int(o[0]!=0)
            was=b['keys'][0]==b['truth'];now=b['keys'][o[0]]==b['truth'];harm1+=int(was and not now);help1+=int(now and not was)
        results.update(changed_lists=changes,rank1_changes=promotions,rank1_new_correct=help1,rank1_correct_lost=harm1,candidate_set_changes=0)
        return results
    results={name:summary(primary,orders) for name,orders in predictions.items()}
    stress_results={name:summary(external,orders) for name,orders in stress_predictions.items()}
    base=evaluate_orders(primary,[list(range(len(b['keys']))) for b in primary]);identities={s:k for b in primary for s,k in zip(b['smiles'],b['keys'])}
    controls={}
    for name in ['P14','P15','P18','P19']:
        frame=pd.read_csv(ROOT/contract['inputs'][name]['path'],dtype=str).set_index('molecule_id')
        orders=[]
        for b in primary:
            indices={k:i for i,k in enumerate(b['keys'])}
            keys=[identities[s] for s in frame.loc[b['mid'],'smiles'].split(';')]
            assert set(keys)==set(b['keys']);orders.append([indices[k] for k in keys])
        controls[name]=summary(primary,orders)['all']
    primary_result=results[cfg['primary_variant']]
    gate=validation_gate(primary_result['all'],primary_result['polarity'],controls,cfg,independent_confirmation=False)
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),config_sha256=sha(CONFIG),
       feature_freeze_sha256=sha(OUT/'feature-freeze.json'),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),
       primary_variant=cfg['primary_variant'],results=results,source_stress=stress_results,controls=controls,gate=gate,
       outer_group_counts=frozen['primary_groups'],largest_outer_group=frozen['largest_primary_group'],
       all_queries_retained=True,missing_truths_score_zero=True,truth_injection=False,formula_positions_preserved=True,
       no_outer_or_stress_parameter_selection=True,no_test_queries_in_fit=True,fresh_reserved_labels_read=False,
       cpu_calibration_only=True,encoder_training=False,gpu_work=False,cloud_runs=False,submission_qualified=False,
       limitation='All panels were already consumed before C36. Nested chemical-group OOF removes new model/alpha fitting leakage, not prior adaptive research reuse or unknown public model pretraining. Source80 is a fixed held-out-source stress check for C36, not a fresh reserved set. A repaired evaluator does not itself establish a leaderboard-predictive validation population.')
    write(OUT/'ranking-results.json',report)
    print('DONE C36 CPU ISOMER RANKING',json.dumps(dict(results=results,source_stress=stress_results,controls=controls,gate=gate)),flush=True)


if __name__=='__main__':main()
