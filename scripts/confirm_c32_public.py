"""Prepare/freeze/evaluate one C30-selected C29 recipe on reserved120 once.

Preparation reads predictions and acquisition metadata only. Freeze requires
completed development selection and a production-source seal. No automatic
submission or confirmation worker calls this script.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import json
import shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase
from precision_cache import ROOT,sha,write,features
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from c14_confirmation_gate import gate
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c32-public-confirmation-20261110'
C31=ROOT/'outputs/local/c31-reserved-public-20261110'
CLOUD=ROOT/'outputs/forks/c31-reserved-public-replay-v1'
C25=ROOT/'outputs/local/c25-reserved-negative-20261110'
LOCAL=ROOT/'outputs/local/c13-confirmation-20261110'
GLOBAL=ROOT/'outputs/local/c30-global-selection-20261110/selection-freeze.json'
PRODUCTION=ROOT/'outputs/local/c33-public-production-20261110/production-source-freeze.json'
SOURCES=['confirm_c32_public.py','c14_confirmation_gate.py','finish_c09_confirmation.py','tail_fusion.py']


def stamp():return datetime.now(timezone.utc).isoformat()


def setup():
    pa.set_cpu_count(2)
    assert rdBase.rdkitVersion=='2026.03.3' and system_memory()['free_gib']>=4
    assert not (ROOT/'.local/compute-paused.json').exists()
    OUT.mkdir(exist_ok=True)


def selected():
    assert GLOBAL.exists(),'C30 development selection pending; reserved metrics remain locked'
    s=json.loads(GLOBAL.read_text())
    assert s['new_candidate_selected'] and s['selected_cycle']=='C29','Only the passing C30-selected C29 recipe is eligible'
    assert not s['C13_confirmation_metrics_read'] and not s['automatic_submission']
    assert s['source_sha256']==sha(ROOT/'scripts/select_c30_global.py')
    implementation=ROOT/'outputs/local/c30-global-selection-20261110/implementation-source-freeze.json'
    for n,pin in json.loads(implementation.read_text())['files_sha256'].items():assert sha(ROOT/n)==pin,n
    policy=ROOT/'outputs/local/c30-global-selection-20261110/protocol.json'
    assert s['protocol_sha256']==sha(policy) and s['policy_sha256']==sha(ROOT/'configs/c30-global-selection-20261110.json')
    for n,pin in json.loads(policy.read_text())['files_sha256'].items():assert sha(ROOT/n)==pin,n
    paths={'C28MV':ROOT/'outputs/local/c28-evaluation-20261110','C29':ROOT/'outputs/local/c29-public-replay-20261110'}
    for name,path in paths.items():
        assert s['development_pins'][name]['development_sha256']==sha(path/'development.json')
        assert s['development_pins'][name]['selection_sha256']==sha(path/'selection-freeze.json')
    d=json.loads((paths['C29']/'development.json').read_text());assert d['new_candidate_selected'] and d['queries']==240
    assert all(d['comparisons'][n]['delta']>0 and d['comparisons'][n]['ci95'][0]>0 for n in ['P01','P16'])
    assert all(d['alias_excluded'][n]['delta']>=0 and d['alias_excluded'][n]['ci95'][0]>=0 for n in ['P01','P16'])
    assert not any(d['truths_lost'].values()) and all(v['delta']>=0 for row in d['polarity'].values() for v in row.values())
    return s


def prepare():
    setup();assert not (OUT/'preparation.json').exists(),'Existing preparation sealed; inspect before recovery'
    assert not (LOCAL/'evaluation-consumption.json').exists()
    c=json.loads((C31/'collector-complete.json').read_text());assert c['state']=='collected' and c['quarantined'] and c['full120_P01_exact_original']
    for n,pin in c['artifact_sha256'].items():assert sha(CLOUD/n)==pin,n
    control=json.loads((C25/'control-receipt.json').read_text());assert sha(C25/'P15.csv')==control['P15_sha256']
    prep=json.loads((C25/'preparation.json').read_text())
    for n,pin in prep['files_sha256'].items():assert sha(C25/n)==pin,n
    q=pd.read_parquet(CLOUD/'c29_queries.parquet');original=pd.read_parquet(ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet')
    pd.testing.assert_frame_equal(q[original.columns],original,check_exact=True)
    assert len(q)==q.molecule_id.nunique()==120 and q.ionization_mode.value_counts().to_dict()=={'positive':60,'negative':60}
    assert not {'key','smiles','inchikey14','formula'}&set(q.columns)
    sources={'P01.csv':CLOUD/'c29_p01.csv','P14.csv':CLOUD/'c29_p14.csv','P15.csv':C25/'P15.csv',
        'candidate-all_reference.csv':CLOUD/'c29_all_reference.csv','candidate-input_alias_excluded.csv':CLOUD/'c29_input_alias_excluded.csv'}
    for view in ['all_reference','input_alias_excluded']:sources['P16-'+view+'.csv']=C25/('P16-'+view+'.csv')
    frames={n:pd.read_csv(p,dtype=str,keep_default_na=False) for n,p in sources.items()}
    assert all(f.molecule_id.tolist()==q.molecule_id.tolist() and f.molecule_id.is_unique for f in frames.values())
    assert all(frames['P01.csv'].equals(frames['P16-'+v+'.csv']) for v in ['all_reference','input_alias_excluded'])
    neg=q.ionization_mode.eq('negative').to_numpy();assert frames['P14.csv'].loc[neg].equals(frames['P01.csv'].loc[neg])
    index=pd.read_parquet(C25/'fingerprint-smiles.parquet');identity=dict(zip(index.smiles,index.metric_key));new=0
    for f in frames.values():
        for row in f.itertuples(index=False):
            keys=[]
            for s in row.smiles.split(';'):
                if s not in identity:identity[s]=features.structure(s)[0];new+=1
                keys.append(identity[s])
            assert 1<=len(keys)<=25 and all(keys) and len(keys)==len(set(keys))
    for n,p in sources.items():shutil.copy2(p,OUT/n)
    shutil.copy2(CLOUD/'c29_queries.parquet',OUT/'queries.parquet')
    pd.DataFrame(dict(smiles=list(identity),metric_key=list(identity.values()))).to_parquet(OUT/'identity.parquet',index=False)
    names=list(sources)+['queries.parquet','identity.parquet']
    write(OUT/'preparation.json',dict(prepared_utc=stamp(),queries=120,positive=60,negative=60,
        files_sha256={n:sha(OUT/n) for n in names},collector_sha256=sha(C31/'collector-complete.json'),
        C25_parent_preparation_sha256=sha(C25/'preparation.json'),P15_control_receipt_sha256=sha(C25/'control-receipt.json'),
        existing_identity_cache_source_sha256=sha(C25/'fingerprint-smiles.parquet'),new_candidate_structures=new,
        full120_P01_P16_exact_original=True,P14_negative_exact_fallback=True,
        confirmation_metrics_read=False,automatic_submission=False))
    print('DONE C32 LABEL-BLIND RESERVED INPUT PREPARATION',flush=True)


def prepared():
    p=json.loads((OUT/'preparation.json').read_text());assert p['queries']==120 and not p['confirmation_metrics_read']
    for n,pin in p['files_sha256'].items():assert sha(OUT/n)==pin,n
    for path,key in [(C31/'collector-complete.json','collector_sha256'),(C25/'preparation.json','C25_parent_preparation_sha256'),
        (C25/'control-receipt.json','P15_control_receipt_sha256'),(C25/'fingerprint-smiles.parquet','existing_identity_cache_source_sha256')]:assert sha(path)==p[key],key
    return p


def production():
    assert PRODUCTION.exists(),'Production source must be sealed before reserved metrics'
    p=json.loads(PRODUCTION.read_text())
    assert p['kind']=='C29_production_source_freeze' and p['selected_recipe']=='C29_fixed_public_mass_dualref'
    assert p['validation_template_sha256']==sha(ROOT/'kaggle/forks/c29-public-replay/notebook.ipynb')
    assert not p['reserved_metrics_read']
    for n,pin in p['source_files_sha256'].items():assert sha(ROOT/n)==pin,n
    return p


def freeze():
    setup();selected();prepared();production()
    assert not (OUT/'confirmation-protocol.json').exists() and not (LOCAL/'evaluation-consumption.json').exists()
    inputs=[GLOBAL,PRODUCTION,C31/'push-receipt.json',ROOT/'kaggle/forks/c31-reserved-public-replay/reserved-pin.json',
        ROOT/'kaggle/forks/c31-reserved-public-replay/notebook.ipynb',ROOT/'configs/c29-public-replay-20261110.json',
        ROOT/'configs/c31-reserved-public-20261110.json',LOCAL/'labels.csv']
    write(OUT/'confirmation-protocol.json',dict(frozen_utc=stamp(),selected_cycle='C29',candidate='C29_fixed_public_mass_dualref',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},source_sha256={n:sha(ROOT/'scripts'/n) for n in SOURCES},
        preparation_sha256=sha(OUT/'preparation.json'),queries=120,positive=60,negative=60,
        statistical_gate='Unchanged C14 full120/alias/polarity gate; all absent truths zero. No coefficient, checkpoint or subgroup selection.',
        confirmation_metrics_read=False,automatic_submission=False,submission_qualified=False))
    print('DONE C32 SELECTED SOURCE/MODEL/RECIPE/PREDICTION FREEZE',flush=True)


def verify():
    selected();prepared();production();p=json.loads((OUT/'confirmation-protocol.json').read_text())
    assert p['preparation_sha256']==sha(OUT/'preparation.json')
    for n,pin in p['input_sha256'].items():assert sha(ROOT/n)==pin,n
    for n,pin in p['source_sha256'].items():assert sha(ROOT/'scripts'/n)==pin,n
    return p


def evaluate(recover_same_recipe=False):
    import msvcrt
    setup();verify()
    lock=(LOCAL/'evaluation.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'confirmation.json').exists(),'Completed confirmation sealed; never reselect'
    q=pd.read_parquet(OUT/'queries.parquet');ids=q.molecule_id.tolist()
    names=['P01','P14','P15']+['P16-'+v for v in ['all_reference','input_alias_excluded']]+['candidate-'+v for v in ['all_reference','input_alias_excluded']]
    frames={n:pd.read_csv(OUT/(n+'.csv'),dtype=str,keep_default_na=False) for n in names}
    assert all(f.molecule_id.tolist()==ids for f in frames.values())
    index=pd.read_parquet(OUT/'identity.parquet');identity=dict(zip(index.smiles,index.metric_key))
    marker=LOCAL/'evaluation-consumption.json'
    expected=dict(campaign='C32CONF',selected_cycle='C29',protocol_sha256=sha(OUT/'confirmation-protocol.json'),
        prediction_preparation_sha256=sha(OUT/'preparation.json'),other_recipe_reselection_permitted=False,automatic_submission=False)
    if recover_same_recipe:
        assert marker.exists(),'Recovery requires an existing consumption marker'
        prior=json.loads(marker.read_text());assert all(prior.get(k)==v for k,v in expected.items()),'Different recipe or prediction freeze; recovery forbidden'
    else:
        assert not marker.exists(),'Reserved120 already consumed; only explicit same-recipe recovery is permitted'
        with marker.open('x',encoding='utf-8') as stream:json.dump(dict(consumed_utc=stamp(),**expected),stream,indent=2)
    # First parsing of structure labels occurs only after the durable global marker.
    labels=pd.read_csv(LOCAL/'labels.csv',dtype=str).set_index('molecule_id').loc[ids]
    assert len(labels)==labels.key.nunique()==120 and labels.index.is_unique and labels.partition.eq('confirmation').all()
    assert labels['mode'].tolist()==q.ionization_mode.tolist()
    values={n:metric_vector(f,labels,identity) for n,f in frames.items()}
    comparisons={};polarity={};losses={}
    for view,prefix in [('all_reference','candidate'),('input_alias_excluded','alias_excluded')]:
        v=values['candidate-'+view]
        for parent in ['P01','P16']:
            b=values['P01' if parent=='P01' else 'P16-'+view]
            comparisons[prefix+'_vs_'+parent]=paired_summary(b,v);losses[view+':'+parent]=int(((b>0)&(v==0)).sum())
            for mode in ['positive','negative']:
                mask=q.ionization_mode.eq(mode).to_numpy();assert int(mask.sum())==60
                polarity[view+':'+mode+':'+parent]=paired_summary(b[mask],v[mask])
    decision=gate(comparisons,polarity)
    controls={n:dict(aggregate=paired_summary(values['P01'],values[n]),polarity={mode:paired_summary(values['P01'][q.ionization_mode.eq(mode).to_numpy()],values[n][q.ionization_mode.eq(mode).to_numpy()]) for mode in ['positive','negative']}) for n in ['P14','P15']}
    pd.DataFrame(dict(molecule_id=ids,mode=q.ionization_mode,**values)).to_csv(OUT/'confirmation-predictions.csv',index=False)
    write(OUT/'confirmation.json',dict(completed_utc=stamp(),queries=120,positive=60,negative=60,candidate='C29_fixed_public_mass_dualref',
        comparisons=comparisons,polarity=polarity,statistical_gate=decision,controls=controls,
        detects_known_public_regression={n:controls[n]['aggregate']['delta']<0 for n in controls},
        truth_coverage={n:int((v>0).sum()) for n,v in values.items()},truths_lost=losses,truth_absent_mrr=0,
        P14_supported=59,P14_explicit_fallback=61,P14_negative_noop_disclosed=True,
        protocol_sha256=sha(OUT/'confirmation-protocol.json'),preparation_sha256=sha(OUT/'preparation.json'),consumption_sha256=sha(marker),
        confirmation_metrics_read=True,confirmation_recipe_reselection=False,automatic_submission=False,submission_qualified=False,
        limitation='Source-shifted CASMI2022 data and mixed reference attribution. Bounded acquisition-alias checks do not exclude all provenance or historical model exposure. Wrong-sign controls remain visible.',
        next_action='Require actual production-source Kaggle changed-path smoke and independent execution/model/identity/fallback/arithmetic audits before the already approved trial.' if decision['passed'] else 'Reject frozen C29recipe; reserved120 is now consumed, preserve0.480champion and unused public approval.'))
    print('DONE C32 FROZEN RESERVED CONFIRMATION',json.dumps(dict(passed=decision['passed'],comparisons=comparisons,polarity=polarity)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','evaluate']);p.add_argument('--recover-same-recipe',action='store_true');a=p.parse_args()
    assert not a.recover_same_recipe or a.stage=='evaluate'
    {'prepare':prepare,'freeze':freeze,'evaluate':lambda:evaluate(a.recover_same_recipe)}[a.stage]()
