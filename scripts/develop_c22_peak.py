"""Source-frozen C22 final-list inference and development; no fresh labels."""
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

from precision_cache import ROOT,sha,write,features
from c22_models import RUN,PACKAGE,EVALUATION,serving_module
from followup_models import serving_bundle
from compatible_tail import apply
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c22-evaluation-20261110'
CLOUD=ROOT/'outputs/forks/c10-casmi2022-validation-v1'
USED=ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv'
CACHE=ROOT/'outputs/local/c12-deep-candidates-20261110/all'


def stamp():return datetime.now(timezone.utc).isoformat()


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'development-protocol.json').exists() and not (OUT/'development.json').exists()
    plan=json.loads(EVALUATION.read_text());assert plan['campaign']=='C22PK' and not plan['automatic_submission']
    previous=ROOT/'outputs/local/c12-deep-candidates-20261110/all'
    prep=json.loads((previous/'preparation.json').read_text())
    assert sha(previous/'parent.csv')==prep['files']['parent.csv']
    parent=pd.read_csv(previous/'parent.csv',dtype=str,keep_default_na=False)
    assert len(parent)==160 and parent.molecule_id.is_unique
    shutil.copy2(previous/'parent.csv',OUT/'parent.csv');shutil.copy2(CLOUD/'c10_queries.parquet',OUT/'queries.parquet')
    sources=['develop_c22_peak.py','c22_models.py','compatible_inference.py','compatible_tail.py',
        'followup_models.py','finish_c09_confirmation.py','tail_fusion.py']
    matched=json.loads((RUN/'matched-control-audit.json').read_text())
    cache_prep=json.loads((CACHE/'preparation.json').read_text())
    cache_contract=json.loads((CACHE.parent/'source-contract.json').read_text())
    assert cache_contract['feature_sha256']==sha(ROOT/'outputs/local/kit/features.py')
    for name in ['fingerprint-smiles.parquet','fingerprints.npy']:
        assert sha(CACHE/name)==cache_prep['files'][name]
    cached_index=pd.read_parquet(CACHE/'fingerprint-smiles.parquet');assert cached_index.smiles.is_unique
    assert set(';'.join(parent.smiles).split(';'))<=set(cached_index.smiles)
    # Compare every common cached structure with the original C08 packed cache.
    original_cache=ROOT/'outputs/publish/c08-models-20261108'
    original_index=pd.read_parquet(original_cache/'fingerprint-cache-smiles.parquet')
    old_lookup=dict(zip(original_index.smiles,range(len(original_index))))
    new_lookup=dict(zip(cached_index.smiles,range(len(cached_index))))
    common=sorted(set(old_lookup)&set(new_lookup));assert common
    a=np.load(original_cache/'fingerprint-cache.npy',mmap_mode='r')
    b=np.load(CACHE/'fingerprints.npy',mmap_mode='r')
    assert np.array_equal(np.unpackbits(np.asarray(a[[old_lookup[s] for s in common]]),axis=1,bitorder='big'),
        np.asarray(b[[new_lookup[s] for s in common]]))
    write(OUT/'development-protocol.json',dict(prepared_utc=stamp(),evaluation_plan_sha256=sha(EVALUATION),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},training_contract_sha256=sha(RUN/'source-contract.json'),
        matched_control_manifest_sha256=matched['control_manifest_sha256'],used_labels_sha256=sha(USED),
        parent_sha256=sha(OUT/'parent.csv'),queries_sha256=sha(OUT/'queries.parquet'),
        baseline_sha256=sha(CLOUD/'c10_p01.csv'),p14_sha256=sha(CLOUD/'c10_p14.csv'),
        C18_parent_prediction_sha256=sha(ROOT/'outputs/local/c18-evaluation-20261110/development.csv'),
        p15_sha256=sha(ROOT/'outputs/local/c10-candidates-20261110/p15-control.csv'),
        protocol_version=1,
        cached_feature_reuse='Exact preexisting source-pinned C12 feature/identity cache; no duplicate canonicalization or scoring arithmetic change.',
        cached_features_sha256={n:sha(CACHE/n) for n in ['preparation.json','fingerprint-smiles.parquet','fingerprints.npy']},
        cache_source_contract_sha256=sha(CACHE.parent/'source-contract.json'),
        original_cache_common_rows_exact=len(common),
        queries=160,all_development_previously_used=True,only_final_10000_update_checkpoints=True,
        weight=.20,offset=3.,rank1_and_candidate_set_exact=True,
        selection='Positive95% lowerbound vs completeP01/P16 and nonnegative mean vs matchedC09control; one fixed recipe/no coefficient search.',
        fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    print('PREPARED C22 DEVELOPMENT PROTOCOL',sha(OUT/'development-protocol.json'),flush=True)


def evaluate():
    import msvcrt
    assert not (OUT/'development.json').exists(),'Completed development is sealed'
    # The completed trainer must have released its exact lock before GPU use.
    lock=(RUN/'trainer.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert (RUN/'results.json').exists()
    assert not (ROOT/'.local/compute-paused.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.7);assert system_memory()['free_gib']>=4
    protocol=json.loads((OUT/'development-protocol.json').read_text())
    for name,pin in protocol['source_sha256'].items():assert sha(ROOT/'scripts'/name)==pin,name
    for p,key in [(EVALUATION,'evaluation_plan_sha256'),(RUN/'source-contract.json','training_contract_sha256'),
        (USED,'used_labels_sha256'),(OUT/'parent.csv','parent_sha256'),(OUT/'queries.parquet','queries_sha256'),
        (CLOUD/'c10_p01.csv','baseline_sha256'),(CLOUD/'c10_p14.csv','p14_sha256'),
        (ROOT/'outputs/local/c10-candidates-20261110/p15-control.csv','p15_sha256')]:assert sha(p)==protocol[key],key
    followup=ROOT/'outputs/publish/c09-followup-models-20261109/followup-manifest.json'
    assert sha(followup)==protocol['matched_control_manifest_sha256']
    manifest,module=serving_module()
    matched_bundle,group,matched_module=serving_bundle('cosine-control',OUT/'control-bundles')
    for name,pin in protocol['cached_features_sha256'].items():assert sha(CACHE/name)==pin,name
    assert sha(CACHE.parent/'source-contract.json')==protocol['cache_source_contract_sha256']
    assert manifest['files']['features.py']==json.loads((CACHE.parent/'source-contract.json').read_text())['feature_sha256']
    cached_index=pd.read_parquet(CACHE/'fingerprint-smiles.parquet');cached_fp=np.load(CACHE/'fingerprints.npy',mmap_mode='r')
    identities=dict(zip(cached_index.smiles.astype(str),cached_index.metric_key.astype(str)))
    parent=pd.read_csv(OUT/'parent.csv',dtype=str,keep_default_na=False)
    needed=set(';'.join(parent.smiles).split(';'))
    shared={str(s):np.asarray(cached_fp[i]) for i,s in enumerate(cached_index.smiles) if s in needed}
    assert needed<=shared.keys()
    sources={'C22':(PACKAGE,'peak-drop-dose',module),'matched_C09':(matched_bundle,group['arm'],matched_module)}
    scores={};receipts={};frames={'P16':parent};changes={}
    for name,(bundle,arm,scorer) in sources.items():
        path=OUT/(name+'-scores.parquet');rpath=path.with_suffix(path.suffix+'.json')
        if path.exists() or rpath.exists():
            assert path.exists() and rpath.exists(),'Partial inference: inspect before recovery'
            receipt=json.loads(rpath.read_text());frame=pd.read_parquet(path)
            assert receipt['manifest_sha256']==sha(bundle/'c08-manifest.json')
            assert receipt['baseline_content_sha256']==hashlib.sha256(parent.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
            assert receipt['queries_sha256']==sha(OUT/'queries.parquet')
        else:frame,receipt=scorer.score(bundle,arm,parent,OUT/'queries.parquet',path,shared_fps=shared)
        assert receipt['scored_molecules']==160 and receipt['input_errors']==0 and not receipt['unsupported_molecules']
        assert receipt['unprocessed_molecules']==0 and frame.c08_ok.astype(bool).all()
        scores[name]=frame;receipts[name]=receipt
        frames[name],changes[name]=apply(parent,frame,protocol['weight'])
        assert changes[name]['rank_one_changes']==changes[name]['candidate_set_changes']==0
    assert system_memory()['free_gib']>=4
    write(OUT/'development-input-freeze.json',dict(frozen_utc=stamp(),protocol_sha256=sha(OUT/'development-protocol.json'),
        C22_manifest_sha256=sha(PACKAGE/'c08-manifest.json'),matched_manifest_sha256=sha(matched_bundle/'c08-manifest.json'),
        scores_sha256={n:sha(OUT/(n+'-scores.parquet')) for n in scores},receipts=receipts,
        changes=changes,fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    labels=pd.read_csv(USED,dtype=str).set_index('molecule_id');assert len(labels)==160
    for name,path in [('P01',CLOUD/'c10_p01.csv'),('P14',CLOUD/'c10_p14.csv'),
        ('P15',ROOT/'outputs/local/c10-candidates-20261110/p15-control.csv')]:
        frames[name]=pd.read_csv(path,dtype=str,keep_default_na=False).set_index('molecule_id').loc[labels.index].reset_index()
    for frame in frames.values():
        for row in frame.itertuples(index=False):
            for s in row.smiles.split(';'):
                if s not in identities:identities[s]=features.structure(s)[0]
    parent_predictions=ROOT/'outputs/local/c18-evaluation-20261110/development.csv'
    assert sha(parent_predictions)==protocol['C18_parent_prediction_sha256']
    frames['C18-parent']=pd.read_csv(parent_predictions,dtype=str,keep_default_na=False)
    values={n:metric_vector(f,labels,identities) for n,f in frames.items()}
    comparisons={n:paired_summary(values[n],values['C22']) for n in ['P01','P16','matched_C09','C18-parent']}
    selected=all(comparisons[n]['delta']>0 and comparisons[n]['ci95'][0]>0 for n in ['P01','P16']) and comparisons['C18-parent']['delta']>=0
    controls={n:paired_summary(values['P01'],values[n]) for n in ['P14','P15','matched_C09']}
    frames['C22'].to_csv(OUT/'development.csv',index=False)
    pd.DataFrame(dict(molecule_id=labels.index,**values)).to_csv(OUT/'development-predictions.csv',index=False)
    write(OUT/'development.json',dict(completed_utc=stamp(),queries=160,candidate='P16_plus_C22PK_tail020',
        comparisons=comparisons,controls=controls,changes=changes,new_candidate_selected=bool(selected),
        truth_coverage={n:int((v>0).sum()) for n,v in values.items()},truth_absent_mrr=0,
        detects_known_public_regression={n:controls[n]['delta']<0 for n in ['P14','P15']},
        negative_control_interpretation='Wrong-sign controls are retained. Positive-only used CASMI2022/unknownCE shift limits hidden-score calibration. Model folds are not independent validation units.',
        protocol_sha256=sha(OUT/'development-protocol.json'),input_freeze_sha256=sha(OUT/'development-input-freeze.json'),
        all_development_previously_used=True,fresh_C13_prediction_metrics_read=False,automatic_submission=False,submission_qualified=False,
        next_action='Only if selected, require C20negative80 sensitivity and freeze before fresh120 evaluation. Otherwise reject the fixed recipe and preserve fresh120. No automatic public trial.'))
    write(OUT/'selection-freeze.json',dict(frozen_utc=stamp(),candidate='P16_plus_C22PK_tail020' if selected else 'P16',
        new_candidate_selected=bool(selected),development_sha256=sha(OUT/'development.json'),
        evaluation_plan_sha256=sha(EVALUATION),C22_manifest_sha256=sha(PACKAGE/'c08-manifest.json'),
        protocol_sha256=sha(OUT/'development-protocol.json'),fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    print('DONE C22 DEVELOPMENT',json.dumps(json.loads((OUT/'development.json').read_text())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-protocol',action='store_true');group.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();prepare() if args.prepare_protocol else evaluate()
