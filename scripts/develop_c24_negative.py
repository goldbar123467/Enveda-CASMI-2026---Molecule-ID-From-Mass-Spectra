"""One C24 full240 screen and a six-group global selection; no fresh metrics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from precision_cache import ROOT,sha,write
from c24_models import RUN,PACKAGE,EVALUATION,serving_module
import develop_c21_negative as c21
from negative_tail import merge
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary

OUT=ROOT/'outputs/local/c24-evaluation-20261110'


def stamp():return datetime.now(timezone.utc).isoformat()


def freeze():
    c21.setup();OUT.mkdir(exist_ok=True)
    assert not (OUT/'development-protocol.json').exists()
    assert not (c21.OUT/'development.json').exists(),'Freeze before any new80 metric read'
    plan=json.loads(EVALUATION.read_text());assert plan['campaign']=='C24PL' and not plan['automatic_submission']
    manifest,_=serving_module();assert manifest['steps']==50000
    c21.verify()
    sources=['develop_c24_negative.py','c24_models.py','run_c24_evaluation_worker.py',
        'negative_tail.py','compatible_inference.py','compatible_tail.py','develop_c21_negative.py',
        'finish_c09_confirmation.py','tail_fusion.py']
    write(OUT/'development-protocol.json',dict(frozen_utc=stamp(),evaluation_plan_sha256=sha(EVALUATION),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},C21_protocol_sha256=sha(c21.OUT/'development-protocol.json'),
        training_contract_sha256=sha(RUN/'source-contract.json'),training_results_sha256=sha(RUN/'results.json'),
        model_manifest_sha256=sha(PACKAGE/'c08-manifest.json'),queries=240,positive=160,negative=80,
        weight=.2,offset=3.,only_final50000_cumulative_models=True,
        selection='Both full240 paired95%lowerbounds positive; no lost truth versus either parent; both polarity means nonnegative.',
        global_tie_order=['C08-parent','C09-control','C14-CE','C15-NP','C18-peak-drop','C24-polarity-loss'],
        global_selection='Evaluate all six; among passing groups greatest aggregate MRR, ties within1e-12 prefer earlier listed group.',
        C20_prediction_metrics_read=False,C13_confirmation_metrics_read=False,automatic_submission=False))
    print('DONE C24 PROSPECTIVE PROTOCOL',sha(OUT/'development-protocol.json'),flush=True)


def verify():
    p=json.loads((OUT/'development-protocol.json').read_text())
    for n,pin in p['source_sha256'].items():assert sha(ROOT/'scripts'/n)==pin,n
    for path,key in [(EVALUATION,'evaluation_plan_sha256'),(c21.OUT/'development-protocol.json','C21_protocol_sha256'),
            (RUN/'source-contract.json','training_contract_sha256'),(RUN/'results.json','training_results_sha256'),
            (PACKAGE/'c08-manifest.json','model_manifest_sha256')]:assert sha(path)==p[key],key
    c21.verify()
    return p


def score():
    import msvcrt
    c21.setup();verify();prep=c21.prepared()
    assert not (OUT/'scoring-receipt.json').exists()
    assert not (ROOT/'.local/compute-paused.json').exists()
    lock=(RUN/'trainer.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.7)
    path=OUT/'C24-scores.parquet';assert not path.exists(),'Inspect partial inference before recovery'
    index=pd.read_parquet(c21.OUT/'fingerprint-smiles.parquet');fp=np.load(c21.OUT/'fingerprints.npy',mmap_mode='r')
    shared={s:np.asarray(fp[i]) for i,s in enumerate(index.smiles)}
    parent=pd.read_csv(c21.OUT/'negative-parent.csv',dtype=str,keep_default_na=False)
    _,module=serving_module()
    scores,receipt=module.score(PACKAGE,'polarity-loss',parent,c21.OUT/'negative-queries.parquet',path,shared_fps=shared)
    assert receipt['scored_molecules']==80 and receipt['input_errors']==receipt['unprocessed_molecules']==0 and not receipt['unsupported_molecules']
    assert scores.c08_ok.astype(bool).all() and scores.molecule_id.nunique()==80
    write(OUT/'scoring-receipt.json',dict(completed_utc=stamp(),receipt=receipt,scores_sha256=sha(path),
        preparation_sha256=sha(c21.OUT/'preparation.json'),protocol_sha256=sha(OUT/'development-protocol.json'),
        C13_confirmation_metrics_read=False,automatic_submission=False))
    print('DONE C24 NEGATIVE80 INFERENCE',flush=True)


def evaluate():
    c21.setup();protocol=verify();c21.prepared()
    assert not (OUT/'development.json').exists() and not (OUT/'evaluation-input-freeze.json').exists()
    receipt=json.loads((OUT/'scoring-receipt.json').read_text())
    assert receipt['preparation_sha256']==sha(c21.OUT/'preparation.json') and receipt['protocol_sha256']==sha(OUT/'development-protocol.json')
    assert receipt['scores_sha256']==sha(OUT/'C24-scores.parquet')
    frames={n:pd.read_csv(c21.OUT/(n+'.csv'),dtype=str,keep_default_na=False) for n in ['P01']}
    frames['P16']=pd.read_csv(c21.OUT/'parent.csv',dtype=str,keep_default_na=False)
    q=pd.read_parquet(c21.OUT/'queries.parquet');modes=dict(zip(q.molecule_id,q.ionization_mode))
    frames['C24-polarity-loss'],actions=merge(frames['P16'],pd.read_parquet(OUT/'C24-scores.parquet'),modes,protocol['weight'])
    assert actions['positive_exact_passthrough']==160 and actions['scored_molecules']==80 and not actions['passthrough_molecules']
    assert actions['rank_one_changes']==actions['candidate_set_changes']==0
    frames['C24-polarity-loss'].to_csv(OUT/'development.csv',index=False)
    write(OUT/'evaluation-input-freeze.json',dict(frozen_utc=stamp(),protocol_sha256=sha(OUT/'development-protocol.json'),
        scoring_receipt_sha256=sha(OUT/'scoring-receipt.json'),prediction_sha256=sha(OUT/'development.csv'),
        C13_confirmation_metrics_read=False,automatic_submission=False))
    labels=pd.concat([pd.read_csv(ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv',dtype=str),
        pd.read_csv(c21.LOCAL/'labels.csv',dtype=str)],ignore_index=True).set_index('molecule_id')
    ids=frames['P16'].molecule_id.tolist();labels=labels.loc[ids]
    assert len(labels)==labels.key.nunique()==240 and labels.index.is_unique
    annotated=labels['mode'].notna();assert all(labels.loc[mid,'mode']==modes[mid] for mid in labels.index[annotated])
    index=pd.read_parquet(c21.OUT/'fingerprint-smiles.parquet');identity=dict(zip(index.smiles,index.metric_key))
    values={n:metric_vector(frame,labels,identity) for n,frame in frames.items()}
    strata={mode:np.array([modes[mid]==mode for mid in ids]) for mode in ['positive','negative']}
    assert int(strata['positive'].sum())==160 and int(strata['negative'].sum())==80
    candidate=values['C24-polarity-loss']
    comparisons={p:paired_summary(values[p],candidate) for p in ['P01','P16']}
    polarity={mode:{p:paired_summary(values[p][mask],candidate[mask]) for p in ['P01','P16']} for mode,mask in strata.items()}
    losses={p:int(((values[p]>0)&(candidate==0)).sum()) for p in ['P01','P16']}
    passed=all(s['delta']>0 and s['ci95'][0]>0 for s in comparisons.values()) and not any(losses.values()) and all(s['delta']>=0 for v in polarity.values() for s in v.values())
    pd.DataFrame(dict(molecule_id=ids,mode=[modes[mid] for mid in ids],**values)).to_csv(OUT/'development-predictions.csv',index=False)
    write(OUT/'development.json',dict(completed_utc=stamp(),queries=240,positive=160,negative=80,
        comparisons=comparisons,polarity=polarity,truths_lost=losses,truths_present=int((candidate>0).sum()),
        actions=actions,supported_development=bool(passed),protocol_sha256=sha(OUT/'development-protocol.json'),
        input_freeze_sha256=sha(OUT/'evaluation-input-freeze.json'),C13_confirmation_metrics_read=False,
        automatic_submission=False,submission_qualified=False))
    print('DONE C24 FULL240 DEVELOPMENT',json.dumps(json.loads((OUT/'development.json').read_text())),flush=True)


def select():
    verify();c21.prepared();assert not (OUT/'global-selection-freeze.json').exists()
    older=json.loads((c21.OUT/'development.json').read_text());new=json.loads((OUT/'development.json').read_text())
    assert older['protocol_sha256']==sha(c21.OUT/'development-protocol.json') and new['protocol_sha256']==sha(OUT/'development-protocol.json')
    assert not older['C13_confirmation_metrics_read'] and not new['C13_confirmation_metrics_read']
    assert older['queries']==new['queries']==240
    for parent in ['P01','P16']:assert abs(older['baseline_MRR'][parent]-new['comparisons'][parent]['baseline_mrr25'])<1e-15
    records=dict(older['groups']);records['C24-polarity-loss']=new
    order=json.loads((OUT/'development-protocol.json').read_text())['global_tie_order'];assert list(records)==order
    eligible=[n for n in order if records[n]['supported_development']]
    best=max((records[n]['comparisons']['P16']['candidate_mrr25'] for n in eligible),default=None)
    winner=next((n for n in eligible if records[n]['comparisons']['P16']['candidate_mrr25']>=best-1e-12),None)
    write(OUT/'global-selection-freeze.json',dict(frozen_utc=stamp(),selected_group=winner,new_candidate_selected=winner is not None,
        groups={n:dict(supported_development=records[n]['supported_development'],comparisons=records[n]['comparisons'],
            polarity=records[n]['polarity'],truths_lost=records[n]['truths_lost']) for n in order},
        source_model_protocol_sha256=sha(OUT/'development-protocol.json'),C21_protocol_sha256=sha(c21.OUT/'development-protocol.json'),
        C21_development_sha256=sha(c21.OUT/'development.json'),C24_development_sha256=sha(OUT/'development.json'),
        C13_confirmation_metrics_read=False,automatic_submission=False,submission_qualified=False,
        next_action='Freeze selected serving source and prospective fresh120 gate, then require confirmation, real Kaggle smoke and independent audits.' if winner else 'Reject all six fixed candidates; keep fresh120 and additional public approval unused.'))
    print('DONE SIX-GROUP GLOBAL SELECTION',winner,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','score','evaluate','select']);args=parser.parse_args()
    {'freeze':freeze,'score':score,'evaluate':evaluate,'select':select}[args.stage]()
