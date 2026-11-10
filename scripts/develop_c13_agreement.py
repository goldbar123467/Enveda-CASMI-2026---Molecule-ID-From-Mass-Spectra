"""Evaluate one fixed C13 rule on the already-consumed160 C10 identities."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import pyarrow as pa

from precision_cache import ROOT,sha,write,features
from prepare_c13_pool import OUT,CONFIG,DEV
from agreement_expansion import expand
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary


def main():
    pa.set_cpu_count(2)
    assert not (OUT/'development.json').exists() and not (OUT/'development-protocol.json').exists()
    config=json.loads(CONFIG.read_text());contract=json.loads((OUT/'source-contract.json').read_text())
    assert contract['config_sha256']==sha(CONFIG)
    for name,pin in contract['source_sha256'].items():assert sha(ROOT/'scripts'/name)==pin
    receipt=json.loads((OUT/'development/scoring-receipt.json').read_text())
    assert receipt['queries']==160 and receipt['input_errors']==0
    assert not receipt['query_labels_read'] and not receipt['prediction_metrics_computed']
    assert receipt['source_contract_sha256']==sha(OUT/'source-contract.json')
    assert receipt['scores_sha256']==sha(OUT/'development/scores.parquet')
    assert all(v['max_raw_score_abs_difference']<=config['local_raw_score_parity_tolerance'] for v in receipt['local_cloud_parity'].values())
    used=ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv'
    write(OUT/'development-protocol.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        evaluator_sha256=sha(__file__),config_sha256=sha(CONFIG),source_contract_sha256=sha(OUT/'source-contract.json'),
        score_receipt_sha256=sha(OUT/'development/scoring-receipt.json'),used_labels_sha256=sha(used),
        selection='One fixed agreement recipe; positive95% lowerbound vs completeP01/P16 plus at least1absenttruth recovered. No parameter search.',
        already_used_development_queries=160,fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    labels=pd.read_csv(used,dtype=str).set_index('molecule_id');assert len(labels)==160
    parent=pd.read_csv(OUT/'development/parent.csv',dtype=str,keep_default_na=False)
    scores=pd.read_parquet(OUT/'development/scores.parquet')
    final,actions=expand(parent,scores,config)
    frames=dict(P16=parent,C13=final)
    for name,path in [('P01',DEV/'c10_p01.csv'),('P14',DEV/'c10_p14.csv'),
        ('P15',ROOT/'outputs/local/c10-candidates-20261110/p15-control.csv')]:
        frames[name]=pd.read_csv(path,dtype=str,keep_default_na=False).set_index('molecule_id').loc[labels.index].reset_index()
    identities=dict(zip(scores.smiles.astype(str),scores.metric_key.astype(str)))
    for frame in frames.values():
        for row in frame.itertuples(index=False):
            for smiles in row.smiles.split(';'):
                if smiles not in identities:identities[smiles]=features.structure(smiles)[0]
    values={name:metric_vector(frame,labels,identities) for name,frame in frames.items()}
    comparisons={name:paired_summary(values[name],values['C13']) for name in ['P01','P16']}
    new=int(((values['P16']==0)&(values['C13']>0)).sum());lost=int(((values['P16']>0)&(values['C13']==0)).sum())
    selected=all(v['delta']>0 and v['ci95'][0]>0 for v in comparisons.values()) and new>0
    controls={name:paired_summary(values['P01'],values[name]) for name in ['P14','P15']}
    final.to_csv(OUT/'development.csv',index=False);actions.to_csv(OUT/'development-actions.csv',index=False)
    pd.DataFrame(dict(molecule_id=labels.index,**values)).to_csv(OUT/'development-predictions.csv',index=False)
    write(OUT/'development.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),queries=160,
        candidate='P16_plus_joint_top_LOTUS',all_development_previously_used=True,
        comparisons=comparisons,truth_coverage={n:int((v>0).sum()) for n,v in values.items()},
        absent_truths_recovered=new,truths_lost=lost,actions={str(k):int(v) for k,v in actions.reason.value_counts().items()},
        new_candidate_selected=bool(selected),controls=controls,
        detects_known_public_regression={n:v['delta']<0 for n,v in controls.items()},
        negative_control_interpretation='P14/P15 signs are retained without filtering; failure to reproduce an actual regression limits source/CE-shift calibration. The modelpair shares its training family; agreement is not independent evidence.',
        config_sha256=sha(CONFIG),source_contract_sha256=sha(OUT/'source-contract.json'),
        protocol_sha256=sha(OUT/'development-protocol.json'),truth_absent_mrr=0,
        fresh_C13_prediction_metrics_read=False,automatic_submission=False,submission_qualified=False,
        limitation='Development only. If selected, freeze and evaluate fresh120 once under a prospective protocol, then require real Kaggle changed-path smoke/independent source/identity/parity/fallback audits and actual P16 resolution.'))
    write(OUT/'selection-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        selected_candidate='P16_plus_joint_top_LOTUS' if selected else 'P16',new_candidate_selected=bool(selected),
        evaluator_sha256=sha(__file__),development_report_sha256=sha(OUT/'development.json'),
        config_sha256=sha(CONFIG),source_contract_sha256=sha(OUT/'source-contract.json'),
        fresh_C13_prediction_metrics_read=False,automatic_submission=False,
        next_action='Prepare prospective fresh120 evaluation only if selected. Otherwise reject this recipe and preserve fresh120; never submit a development failure.'))
    print('DONE C13 FIXED DEVELOPMENT',json.dumps(json.loads((OUT/'development.json').read_text())),flush=True)


if __name__=='__main__':main()
