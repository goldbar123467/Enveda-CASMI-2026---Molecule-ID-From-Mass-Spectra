"""Frozen one-recipe full240 development. Never reads reserved C13 labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import hashlib
import json
from datetime import datetime,timezone

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase

from precision_cache import ROOT,sha,write
from c28_models import RUN,PACKAGE,EVALUATION,serving_module
from compatible_tail import apply
from followup_models import serving_bundle
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c28-evaluation-20261110'
INPUT=ROOT/'outputs/local/c21-negative-tail-20261110'
LABELS=[ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv',ROOT/'outputs/local/c20-negative-development-20261110/labels.csv']
FOLLOWUP=ROOT/'outputs/publish/c09-followup-models-20261109/followup-manifest.json'


def stamp():return datetime.now(timezone.utc).isoformat()


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'development-protocol.json').exists() and not (OUT/'development.json').exists()
    cfg=json.loads(EVALUATION.read_text());assert cfg['campaign']=='C28MV' and cfg['weight']==.2 and cfg['offset']==3.
    assert cfg['queries']==240 and not cfg['automatic_submission']
    contract=json.loads((RUN/'source-contract.json').read_text())
    for p,pin in contract['source_sha256'].items():assert sha(ROOT/p)==pin,p
    prep=json.loads((INPUT/'preparation.json').read_text())
    for name,pin in prep['files_sha256'].items():assert sha(INPUT/name)==pin,name
    assert prep['queries']==240 and prep['negative_P01_equals_P16']
    scoring=json.loads((INPUT/'scoring-receipt.json').read_text());assert sha(INPUT/'P15.csv')==scoring['P15_sha256']
    assert scoring['P15_reused_exact_parent_scores']
    parent=pd.read_csv(INPUT/'parent.csv',dtype=str,keep_default_na=False)
    assert len(parent)==240 and parent.molecule_id.is_unique
    assert parent.equals(pd.read_csv(INPUT/'P01.csv',dtype=str,keep_default_na=False))
    q=pd.read_parquet(INPUT/'queries.parquet');assert q.molecule_id.tolist()==parent.molecule_id.tolist()
    assert q.ionization_mode.value_counts().to_dict()=={'positive':160,'negative':80}
    sources=['develop_c28_multiview.py','c28_models.py','compatible_inference.py','compatible_tail.py',
        'followup_models.py','finish_c09_confirmation.py','tail_fusion.py']
    files=[INPUT/n for n in ['parent.csv','P01.csv','P14.csv','P15.csv','queries.parquet','fingerprint-smiles.parquet',
        'fingerprints.npy','preparation.json','development-protocol.json','scoring-receipt.json','development.json']]
    files+=LABELS+[FOLLOWUP,ROOT/'outputs/publish/c08-models-20261108/c08-manifest.json']
    write(OUT/'development-protocol.json',dict(prepared_utc=stamp(),evaluation_plan_sha256=sha(EVALUATION),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},training_contract_sha256=sha(RUN/'source-contract.json'),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},queries=240,positive=160,negative=80,
        all_development_previously_used=True,only_final5000_additional_updates=True,
        weight=.2,offset=3.,rank1_and_candidate_set_exact=True,
        selection='Both complete-parent paired95% lowerbounds strictly positive; both polarity means nonnegative; mean vs matchedC09 nonnegative; all missing truths zero.',
        C08_control='P15 is the exact all-mode C08parent0.20tail on identical P01/P16.',
        closed_previous_cycle='All six C21/C24 groups were rejected; no candidate is added retrospectively.',
        fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    print('C28 FULL240 PROTOCOL FROZEN',sha(OUT/'development-protocol.json'),flush=True)


def verify():
    p=json.loads((OUT/'development-protocol.json').read_text())
    assert sha(EVALUATION)==p['evaluation_plan_sha256'] and sha(RUN/'source-contract.json')==p['training_contract_sha256']
    for name,pin in p['source_sha256'].items():assert sha(ROOT/'scripts'/name)==pin,name
    for name,pin in p['input_sha256'].items():assert sha(ROOT/name)==pin,name
    return p


def evaluate():
    import msvcrt
    assert not (OUT/'development.json').exists(),'Completed development sealed'
    lock=(RUN/'trainer.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert (RUN/'results.json').exists() and not (ROOT/'.local/compute-paused.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert rdBase.rdkitVersion=='2026.03.3' and torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.7);assert system_memory()['free_gib']>=4
    protocol=verify();manifest,module=serving_module()
    assert manifest['evaluation_plan_sha256']==protocol['evaluation_plan_sha256']
    matched_bundle,group,matched_module=serving_bundle('cosine-control',OUT/'control-bundles')
    index=pd.read_parquet(INPUT/'fingerprint-smiles.parquet');fp=np.load(INPUT/'fingerprints.npy',mmap_mode='r')
    assert index.smiles.is_unique
    shared={s:np.asarray(fp[i]) for i,s in enumerate(index.smiles)}
    parent=pd.read_csv(INPUT/'parent.csv',dtype=str,keep_default_na=False)
    assert set(';'.join(parent.smiles).split(';'))<=set(shared)
    frames={'P16':parent};changes={};receipts={}
    sources={'C28':(PACKAGE,'multi-view',module),'matched_C09':(matched_bundle,group['arm'],matched_module)}
    for name,(bundle,arm,scorer) in sources.items():
        path=OUT/(name+'-scores.parquet');rpath=path.with_suffix(path.suffix+'.json')
        if path.exists() or rpath.exists():
            assert path.exists() and rpath.exists(),'Partial inference: inspect before recovery'
            receipt=json.loads(rpath.read_text());frame=pd.read_parquet(path)
            assert receipt['manifest_sha256']==sha(bundle/'c08-manifest.json')
            assert receipt['baseline_content_sha256']==hashlib.sha256(parent.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
            assert receipt['queries_sha256']==sha(INPUT/'queries.parquet')
        else:frame,receipt=scorer.score(bundle,arm,parent,INPUT/'queries.parquet',path,shared_fps=shared)
        assert receipt['scored_molecules']==240 and receipt['input_errors']==receipt['unprocessed_molecules']==0
        assert not receipt['unsupported_molecules'] and frame.c08_ok.astype(bool).all()
        frames[name],changes[name]=apply(parent,frame,protocol['weight'])
        assert changes[name]['rank_one_changes']==changes[name]['candidate_set_changes']==0
        receipts[name]=receipt;frames[name].to_csv(OUT/(name+'-predictions.csv'),index=False)
    write(OUT/'evaluation-input-freeze.json',dict(frozen_utc=stamp(),protocol_sha256=sha(OUT/'development-protocol.json'),
        C28_manifest_sha256=sha(PACKAGE/'c08-manifest.json'),matched_manifest_sha256=sha(matched_bundle/'c08-manifest.json'),
        score_sha256={n:sha(OUT/(n+'-scores.parquet')) for n in sources},
        predictions_sha256={n:sha(OUT/(n+'-predictions.csv')) for n in sources},receipts=receipts,changes=changes,
        fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    # Only already-used development labels are parsed, after prediction freeze.
    labels=pd.concat([pd.read_csv(p,dtype=str) for p in LABELS],ignore_index=True).set_index('molecule_id')
    ids=parent.molecule_id.tolist();labels=labels.loc[ids];assert len(labels)==labels.key.nunique()==240 and labels.index.is_unique
    q=pd.read_parquet(INPUT/'queries.parquet');modes=dict(zip(q.molecule_id,q.ionization_mode))
    annotated=labels['mode'].notna();assert all(labels.loc[mid,'mode']==modes[mid] for mid in labels.index[annotated])
    for name in ['P01','P14','P15']:
        frames[name]=pd.read_csv(INPUT/(name+'.csv'),dtype=str,keep_default_na=False)
    frames['C08-parent']=frames['P15']
    identity=dict(zip(index.smiles,index.metric_key))
    values={n:metric_vector(f,labels,identity) for n,f in frames.items()}
    strata={mode:np.array([modes[mid]==mode for mid in ids]) for mode in ['positive','negative']}
    comparisons={n:paired_summary(values[n],values['C28']) for n in ['P01','P16','matched_C09','C08-parent']}
    polarity={mode:{n:paired_summary(values[n][mask],values['C28'][mask]) for n in ['P01','P16','matched_C09']}
        for mode,mask in strata.items()}
    losses={n:int(((values[n]>0)&(values['C28']==0)).sum()) for n in ['P01','P16']}
    selected=all(comparisons[n]['delta']>0 and comparisons[n]['ci95'][0]>0 for n in ['P01','P16'])
    selected=selected and comparisons['matched_C09']['delta']>=0 and not any(losses.values())
    selected=selected and all(polarity[mode][n]['delta']>=0 for mode in strata for n in ['P01','P16'])
    controls={n:dict(aggregate=paired_summary(values['P01'],values[n]),polarity={mode:paired_summary(values['P01'][mask],values[n][mask])
        for mode,mask in strata.items()}) for n in ['P14','P15','matched_C09','C08-parent']}
    pd.DataFrame(dict(molecule_id=ids,mode=[modes[mid] for mid in ids],**values)).to_csv(OUT/'development-predictions.csv',index=False)
    write(OUT/'development.json',dict(completed_utc=stamp(),queries=240,positive=160,negative=80,
        candidate='P16_plus_C28MV_allmode_tail020',comparisons=comparisons,polarity=polarity,controls=controls,
        changes=changes,truths_lost=losses,truth_coverage={n:int((v>0).sum()) for n,v in values.items()},truth_absent_mrr=0,
        new_candidate_selected=bool(selected),protocol_sha256=sha(OUT/'development-protocol.json'),
        input_freeze_sha256=sha(OUT/'evaluation-input-freeze.json'),all_development_previously_used=True,
        limitation='Used CASMI2022 source-shifted development. P14 wrong-direction/negative no-op disclosed; no hidden-score guarantee. C20 includes real generated recovery of one originally empty pool.',
        fresh_C13_prediction_metrics_read=False,automatic_submission=False,submission_qualified=False))
    write(OUT/'selection-freeze.json',dict(frozen_utc=stamp(),new_candidate_selected=bool(selected),
        candidate='P16_plus_C28MV_allmode_tail020' if selected else None,
        development_sha256=sha(OUT/'development.json'),protocol_sha256=sha(OUT/'development-protocol.json'),
        model_manifest_sha256=sha(PACKAGE/'c08-manifest.json'),evaluation_plan_sha256=sha(EVALUATION),
        fresh_C13_prediction_metrics_read=False,automatic_submission=False,submission_qualified=False,
        next_action='Freeze new all-mode confirmation/deployment recipe before once-only reserved120 gate; real Kaggle smoke/audits still required.' if selected else 'Reject fixed recipe; preserve reserved120 and approved public trial.'))
    print('DONE C28 FULL240 DEVELOPMENT',json.dumps(dict(selected=bool(selected),comparisons=comparisons,polarity=polarity)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'evaluate':evaluate}[args.stage]()
