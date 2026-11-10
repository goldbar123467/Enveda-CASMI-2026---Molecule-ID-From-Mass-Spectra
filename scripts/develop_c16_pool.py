"""C16 broader-pool scoring/selection on used160; no fresh labels or submission."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import hashlib
import json
import shutil
from datetime import datetime,timezone

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase

from precision_cache import ROOT,sha,write
from c15_models import PACKAGE,serving_module
from followup_models import serving_bundle
from fold_agreement_expansion import expand
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c16-learned-pool-20261110'
CONFIG=ROOT/'configs/c16-learned-pool-20261110.json'
POOL=ROOT/'outputs/local/c12-deep-candidates-20261110/all'
CLOUD=ROOT/'outputs/forks/c10-casmi2022-validation-v1'
USED=ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv'


def stamp():return datetime.now(timezone.utc).isoformat()


def groups():
    bundle,_,control=serving_bundle('cosine-control',OUT/'bundles')
    _,candidate=serving_module()
    return {'cosine-control':(bundle,'cosine-control',control),'C15NP':(PACKAGE,'np-balanced',candidate)}


def prepare():
    pa.set_cpu_count(2);assert rdBase.rdkitVersion=='2026.03.3'
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    prep=json.loads((POOL/'preparation.json').read_text())
    for name,pin in prep['files'].items():assert sha(POOL/name)==pin,name
    original=pd.read_parquet(POOL/'pool.parquet').sort_values(['molecule_id','prior_rank'],kind='stable')
    parent=pd.read_csv(POOL/'parent.csv',dtype=str,keep_default_na=False)
    assert len(parent)==160 and parent.molecule_id.is_unique
    assert original.groupby('molecule_id').size().max()<=128
    for mid,g in original.groupby('molecule_id',sort=False):assert g.metric_key.is_unique
    strings=original.groupby('molecule_id',sort=False).smiles.agg(';'.join)
    wide=parent[['molecule_id']].copy();wide['smiles']=wide.molecule_id.map(strings)
    wide.to_csv(OUT/'candidate-pool.csv',index=False);shutil.copy2(POOL/'parent.csv',OUT/'parent.csv')
    model_groups=groups()
    sources=['develop_c16_pool.py','fold_agreement_expansion.py','c15_models.py','followup_models.py',
        'compatible_inference.py','finish_c09_confirmation.py','tail_fusion.py']
    write(OUT/'protocol.json',dict(prepared_utc=stamp(),config_sha256=sha(CONFIG),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},used_labels_sha256=sha(USED),
        pool_preparation_sha256=sha(POOL/'preparation.json'),pool_files_sha256=prep['files'],
        parent_sha256=sha(OUT/'parent.csv'),candidate_pool_sha256=sha(OUT/'candidate-pool.csv'),
        queries_sha256=sha(CLOUD/'c10_queries.parquet'),
        models={n:dict(bundle=str(b.relative_to(ROOT)),manifest_sha256=sha(b/'c08-manifest.json')) for n,(b,_,_) in model_groups.items()},
        narrow_score_sha256={n:sha(ROOT/'outputs/local/c15-evaluation-20261110'/file) for n,file in
            [('cosine-control','matched_C09-scores.parquet'),('C15NP','C15-scores.parquet')]},
        queries=160,candidates=len(original),raw_parent_score_parity_tolerance=1e-4,
        all_development_previously_used=True,fresh_C13_prediction_metrics_read=False,
        selection='Predeclared2groups/onefixedrule. Requirepositive95%lowerbound vsboth completeP01/P16 plusnewtruth. GreatestMRR,tieoldercontrol. No thresholds/checkpoints searched.',
        automatic_submission=False))
    print('PREPARED C16 LABELBLIND PROTOCOL',sha(OUT/'protocol.json'),flush=True)


def verify():
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(CONFIG)==p['config_sha256'] and sha(USED)==p['used_labels_sha256']
    for name,pin in p['source_sha256'].items():assert sha(ROOT/'scripts'/name)==pin,name
    for name,pin in p['pool_files_sha256'].items():assert sha(POOL/name)==pin,name
    assert sha(OUT/'parent.csv')==p['parent_sha256'] and sha(OUT/'candidate-pool.csv')==p['candidate_pool_sha256']
    assert sha(CLOUD/'c10_queries.parquet')==p['queries_sha256']
    return p


def score():
    import msvcrt
    lock=(OUT/'scoring.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'scoring-receipt.json').exists()
    p=verify();assert not (ROOT/'.local/compute-paused.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.7);assert system_memory()['free_gib']>=4
    index=pd.read_parquet(POOL/'fingerprint-smiles.parquet');fp=np.load(POOL/'fingerprints.npy',mmap_mode='r')
    assert index.smiles.is_unique and fp.shape==(len(index),2048)
    shared={str(s):np.asarray(fp[i]) for i,s in enumerate(index.smiles)}
    model_groups=groups();receipts={};parity={}
    for name,(bundle,arm,module) in model_groups.items():
        assert sha(bundle/'c08-manifest.json')==p['models'][name]['manifest_sha256']
        path=OUT/(name+'-scores.parquet');rpath=path.with_suffix(path.suffix+'.json')
        assert not path.exists() and not rpath.exists(),'Existing inference must be inspected,never blindly repeated'
        values,receipt=module.score(bundle,arm,OUT/'candidate-pool.csv',CLOUD/'c10_queries.parquet',path,shared_fps=shared)
        assert receipt['scored_molecules']==160 and not receipt['unsupported_molecules'] and receipt['input_errors']==0
        assert receipt['unprocessed_molecules']==0 and values.c08_ok.astype(bool).all()
        narrow=ROOT/'outputs/local/c15-evaluation-20261110'/('C15-scores.parquet' if name=='C15NP' else 'matched_C09-scores.parquet')
        assert sha(narrow)==p['narrow_score_sha256'][name]
        expected=pd.read_parquet(narrow).set_index(['molecule_id','smiles'])
        actual=values.set_index(['molecule_id','smiles']).loc[expected.index]
        error=float(np.max(np.abs(actual[['c08_raw_fold0','c08_raw_fold1']].to_numpy()-expected[['c08_raw_fold0','c08_raw_fold1']].to_numpy())))
        assert np.isfinite(error) and error<=p['raw_parent_score_parity_tolerance']
        parity[name]=dict(rows=len(expected),max_raw_score_abs_difference=error)
        receipts[name]=receipt
        print('C16 FULL POOL SCORED',name,len(values),'PARENT RAW ERROR',error,flush=True)
    write(OUT/'scoring-receipt.json',dict(completed_utc=stamp(),queries=160,groups=receipts,
        raw_parent_parity=parity,scores_sha256={n:sha(OUT/(n+'-scores.parquet')) for n in receipts},
        protocol_sha256=sha(OUT/'protocol.json'),query_labels_read=False,fresh_C13_prediction_metrics_read=False,
        peak_Torch_reserved_gib=torch.cuda.max_memory_reserved()/2**30,automatic_submission=False))
    print('DONE C16 LABELBLIND SCORING',flush=True)


def evaluate():
    p=verify();assert not (OUT/'development.json').exists()
    receipt=json.loads((OUT/'scoring-receipt.json').read_text());assert receipt['protocol_sha256']==sha(OUT/'protocol.json')
    cfg=json.loads(CONFIG.read_text());pool=pd.read_parquet(POOL/'pool.parquet');parent=pd.read_csv(OUT/'parent.csv',dtype=str,keep_default_na=False)
    frames={};actions={}
    for name in cfg['model_groups_in_tie_order']:
        path=OUT/(name+'-scores.parquet');assert sha(path)==receipt['scores_sha256'][name]
        scores=pd.read_parquet(path).rename(columns={'baseline_rank':'prior_rank'})
        merged=pool.merge(scores,on=['molecule_id','smiles','prior_rank'],validate='one_to_one')
        assert len(merged)==len(pool)
        frames[name],actions[name]=expand(parent,merged,cfg)
        frames[name].to_csv(OUT/(name+'-predictions.csv'),index=False);actions[name].to_csv(OUT/(name+'-actions.csv'),index=False)
    write(OUT/'development-input-freeze.json',dict(frozen_utc=stamp(),protocol_sha256=sha(OUT/'protocol.json'),
        score_receipt_sha256=sha(OUT/'scoring-receipt.json'),prediction_sha256={n:sha(OUT/(n+'-predictions.csv')) for n in frames},
        fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    labels=pd.read_csv(USED,dtype=str).set_index('molecule_id');assert len(labels)==160
    index=pd.read_parquet(POOL/'fingerprint-smiles.parquet');identities=dict(zip(index.smiles.astype(str),index.metric_key.astype(str)))
    baseline=metric_vector(parent,labels,identities)
    p01=pd.read_csv(CLOUD/'c10_p01.csv',dtype=str,keep_default_na=False)
    base01=metric_vector(p01,labels,identities);assert np.array_equal(base01,baseline)
    reports={};eligible=[]
    for name,frame in frames.items():
        values=metric_vector(frame,labels,identities);summary=paired_summary(baseline,values)
        new=int(((baseline==0)&(values>0)).sum());lost=int(((baseline>0)&(values==0)).sum())
        passed=summary['delta']>0 and summary['ci95'][0]>0 and new>0
        reports[name]=dict(vs_P01=summary,vs_P16=summary,absent_truths_recovered=new,truths_lost=lost,
            truth_coverage=int((values>0).sum()),supported_development=bool(passed),
            actions={str(k):int(v) for k,v in actions[name].reason.value_counts().items()})
        if passed:eligible.append(name)
    winner=max(eligible,key=lambda n:reports[n]['vs_P16']['candidate_mrr25']) if eligible else None
    report=dict(completed_utc=stamp(),queries=160,candidates=reports,selected_group=winner,
        new_candidate_selected=winner is not None,baseline_truths=int((baseline>0).sum()),truth_absent_mrr=0,
        all_development_previously_used=True,protocol_sha256=sha(OUT/'protocol.json'),
        score_receipt_sha256=sha(OUT/'scoring-receipt.json'),input_freeze_sha256=sha(OUT/'development-input-freeze.json'),
        fresh_C13_prediction_metrics_read=False,automatic_submission=False,submission_qualified=False,
        limitation='UsedpositiveCASMI2022,sharedfoldmodeltraining and knownwrong-signP14control limit hidden-score calibration. A selected recipe still needs fresh120,realKagglechanged-path smoke/independentaudits. No automatic trial.')
    write(OUT/'development.json',report)
    write(OUT/'selection-freeze.json',dict(frozen_utc=stamp(),selected_group=winner,new_candidate_selected=winner is not None,
        development_sha256=sha(OUT/'development.json'),config_sha256=sha(CONFIG),protocol_sha256=sha(OUT/'protocol.json'),
        fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    print('DONE C16 DEVELOPMENT',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','score','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'score':score,'evaluate':evaluate}[args.stage]()
