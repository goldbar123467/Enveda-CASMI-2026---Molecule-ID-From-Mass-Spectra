"""One frozen public-recipe development evaluation, full240 and alias view."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import json
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase
from precision_cache import ROOT,sha,write,features
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c29-public-replay-20261110'
CLOUD=ROOT/'outputs/forks/c29-public-replay-v1'
BASE=ROOT/'outputs/local/c21-negative-tail-20261110'
PLAN=ROOT/'configs/c29-public-replay-20261110.json'
LABELS=[ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv',ROOT/'outputs/local/c20-negative-development-20261110/labels.csv']


def stamp():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not (OUT/'development-protocol.json').exists() and not (OUT/'development.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    files=[PLAN,ROOT/'configs/c30-global-selection-20261110.json',ROOT/'outputs/local/c30-global-selection-20261110/protocol.json',
        ROOT/'kaggle/forks/c29-public-replay/validation-pin.json',ROOT/'kaggle/forks/c29-public-replay/notebook.ipynb',
        OUT/'source-audit.json',OUT/'push-receipt.json']
    files+=[BASE/n for n in ['parent.csv','P01.csv','P14.csv','P15.csv','queries.parquet','fingerprint-smiles.parquet','preparation.json']]+LABELS
    sources=['develop_c29_public.py','finish_c09_confirmation.py','tail_fusion.py','select_c30_global.py']
    write(OUT/'development-protocol.json',dict(frozen_utc=stamp(),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},recipe='One fixed C29all-reference output with mandatory alias-excluded sensitivity.',
        queries=240,all_development_previously_used=True,truth_absent_mrr=0,
        C29_prediction_metrics_read=False,C13_confirmation_metrics_read=False,automatic_submission=False))
    print('C29 DEVELOPMENT PROTOCOL FROZEN',sha(OUT/'development-protocol.json'),flush=True)


def evaluate():
    assert not (OUT/'development.json').exists(),'Completed development sealed'
    assert not (ROOT/'.local/compute-paused.json').exists()
    pa.set_cpu_count(2);assert rdBase.rdkitVersion=='2026.03.3' and system_memory()['free_gib']>=4
    p=json.loads((OUT/'development-protocol.json').read_text())
    for n,pin in p['input_sha256'].items():assert sha(ROOT/n)==pin,n
    for n,pin in p['source_sha256'].items():assert sha(ROOT/'scripts'/n)==pin,n
    collected=json.loads((OUT/'collector-complete.json').read_text());assert collected['state']=='collected'
    for n,pin in collected['artifact_sha256'].items():assert sha(CLOUD/n)==pin,n
    report=json.loads((CLOUD/'c29_public_report.json').read_text())
    assert report['plan_sha256']==sha(PLAN) and report['queries']==240 and not report['metrics_computed']
    assert report['mass']['status']=='on' and report['mass_lead']['status']=='ran' and report['rr']['status']=='off in this preset'
    allocation=json.loads((CLOUD/'c29_t4_allocation.json').read_text());assert len(allocation['devices'])==2 and all(d['name']=='Tesla T4' for d in allocation['devices'])
    execution=json.loads((CLOUD/'execution_receipt.json').read_text());assert execution['is_full_run'] and execution['processed_molecules']==240
    control=json.loads((CLOUD/'c29_report.json').read_text());assert control['p14_receipt']['input_errors']==0
    parent=pd.read_csv(BASE/'parent.csv',dtype=str,keep_default_na=False)
    p01=pd.read_csv(CLOUD/'c29_p01.csv',dtype=str,keep_default_na=False)
    assert parent.equals(p01),'New full240 P01 rerun differs from authoritative parents; diagnose before label metrics'
    q=pd.read_parquet(CLOUD/'c29_queries.parquet');old=pd.read_parquet(BASE/'queries.parquet')
    assert q.molecule_id.tolist()==parent.molecule_id.tolist()==old.molecule_id.tolist()
    for a,b in zip(q.itertuples(index=False),old.itertuples(index=False)):
        assert a.precursor_mz==b.precursor_mz and a.adduct==b.adduct and a.ionization_mode==b.ionization_mode
        for field in ['ms2_mzs','ms2_normalized_intensities']:assert np.array_equal(features.array_value(getattr(a,field)),features.array_value(getattr(b,field)))
    frames={'P01':p01,'P16':parent,'P14':pd.read_csv(CLOUD/'c29_p14.csv',dtype=str,keep_default_na=False),
        'P15':pd.read_csv(BASE/'P15.csv',dtype=str,keep_default_na=False)}
    for name in ['all_reference','input_alias_excluded']:frames[name]=pd.read_csv(CLOUD/('c29_'+name+'.csv'),dtype=str,keep_default_na=False)
    index=pd.read_parquet(BASE/'fingerprint-smiles.parquet');identities=dict(zip(index.smiles,index.metric_key))
    for frame in frames.values():
        assert frame.molecule_id.tolist()==parent.molecule_id.tolist() and frame.molecule_id.is_unique
        for row in frame.itertuples(index=False):
            keys=[]
            for s in row.smiles.split(';'):
                if s not in identities:identities[s]=features.structure(s)[0]
                keys.append(identities[s])
            assert 1<=len(keys)<=25 and all(keys) and len(keys)==len(set(keys))
    write(OUT/'evaluation-input-freeze.json',dict(frozen_utc=stamp(),protocol_sha256=sha(OUT/'development-protocol.json'),
        collector_sha256=sha(OUT/'collector-complete.json'),cloud_report_sha256=sha(CLOUD/'c29_public_report.json'),
        source_predictions_sha256={n:sha(CLOUD/('c29_'+n+'.csv')) for n in ['all_reference','input_alias_excluded']},
        parent_replay_exact=True,all240retained=True,C29_prediction_metrics_read=False,C13_confirmation_metrics_read=False))
    labels=pd.concat([pd.read_csv(f,dtype=str) for f in LABELS],ignore_index=True).set_index('molecule_id').loc[parent.molecule_id]
    assert len(labels)==labels.key.nunique()==240 and labels.index.is_unique
    values={n:metric_vector(f,labels,identities) for n,f in frames.items()}
    masks={mode:q.ionization_mode.eq(mode).to_numpy() for mode in ['positive','negative']}
    assert [int(masks[m].sum()) for m in masks]==[160,80]
    comparisons={n:paired_summary(values[n],values['all_reference']) for n in ['P01','P16']}
    alias={n:paired_summary(values[n],values['input_alias_excluded']) for n in ['P01','P16']}
    polarity={mode:{n:paired_summary(values[n][mask],values['all_reference'][mask]) for n in ['P01','P16']} for mode,mask in masks.items()}
    losses={n:int(((values[n]>0)&(values['all_reference']==0)).sum()) for n in ['P01','P16']}
    passed=all(s['delta']>0 and s['ci95'][0]>0 for s in comparisons.values()) and not any(losses.values())
    passed=passed and all(s['delta']>=0 and s['ci95'][0]>=0 for s in alias.values())
    passed=passed and all(s['delta']>=0 for strata in polarity.values() for s in strata.values())
    controls={n:dict(aggregate=paired_summary(values['P01'],values[n]),polarity={mode:paired_summary(values['P01'][mask],values[n][mask]) for mode,mask in masks.items()}) for n in ['P14','P15']}
    pd.DataFrame(dict(molecule_id=parent.molecule_id,mode=q.ionization_mode,**values)).to_csv(OUT/'development-predictions.csv',index=False)
    result=dict(completed_utc=stamp(),queries=240,positive=160,negative=80,candidate='C29_fixed_public_mass_dualref',comparisons=comparisons,
        alias_excluded=alias,polarity=polarity,truths_lost=losses,truth_coverage={n:int((v>0).sum()) for n,v in values.items()},
        controls=controls,source_alias_audits={v:report['results'][v]['audit'] for v in ['all_reference','input_alias_excluded']},
        public_mass_adduct_unsupported=report['public_mass_adduct_unsupported'],truth_absent_mrr=0,new_candidate_selected=bool(passed),
        protocol_sha256=sha(OUT/'development-protocol.json'),input_freeze_sha256=sha(OUT/'evaluation-input-freeze.json'),
        limitation='Used CASMI2022 development and reference-source overlap; bounded alias checks do not prove universal exclusion. Preserve wrong-sign controls. Public author score is not this account score.',
        all_development_previously_used=True,C13_confirmation_metrics_read=False,automatic_submission=False,submission_qualified=False)
    write(OUT/'development.json',result)
    write(OUT/'selection-freeze.json',dict(frozen_utc=stamp(),new_candidate_selected=bool(passed),candidate=result['candidate'] if passed else None,
        development_sha256=sha(OUT/'development.json'),protocol_sha256=sha(OUT/'development-protocol.json'),
        C13_confirmation_metrics_read=False,automatic_submission=False,submission_qualified=False,next_action='C30outer selector must also wait C28 terminal; do not read C13 directly.'))
    print('DONE C29 DEVELOPMENT',json.dumps(dict(selected=bool(passed),comparisons=comparisons,alias_excluded=alias,polarity=polarity)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','evaluate']);a=p.parse_args();{'freeze':freeze,'evaluate':evaluate}[a.stage]()
