"""One globally selected negative-tail recipe on reserved120; never submits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from precision_cache import ROOT,sha,write
from prepare_c25_reserved_inputs import OUT,CONFIG,LOCAL,verify as verify_inputs
import develop_c21_negative as c21
import develop_c24_negative as c24
from c24_models import PACKAGE as C24_PACKAGE,serving_module as c24_module
from negative_tail import merge
from compatible_tail import apply
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary
from c14_confirmation_gate import gate

GLOBAL=c24.OUT/'global-selection-freeze.json'
SOURCES=['confirm_c25_negative.py','prepare_c25_reserved_inputs.py','run_c25_confirmation_worker.py',
    'negative_tail.py','compatible_tail.py','compatible_inference.py','followup_models.py',
    'c14_models.py','c15_models.py','c18_models.py','c24_models.py',
    'develop_c21_negative.py','develop_c24_negative.py','finish_c09_confirmation.py','tail_fusion.py','c14_confirmation_gate.py']


def stamp():return datetime.now(timezone.utc).isoformat()


def sources():return {n:sha(ROOT/'scripts'/n) for n in SOURCES}


def model_groups():
    groups=c21.groups();_,module=c24_module();groups['C24-polarity-loss']=(C24_PACKAGE,'polarity-loss',module)
    return groups


def fingerprints():
    index=pd.read_parquet(OUT/'fingerprint-smiles.parquet');fp=np.load(OUT/'fingerprints.npy',mmap_mode='r')
    return {s:np.asarray(fp[i]) for i,s in enumerate(index.smiles)}


def controls():
    c21.setup();verify_inputs();assert not (OUT/'control-source-freeze.json').exists()
    assert not (LOCAL/'evaluation-consumption.json').exists()
    bundle,arm,module=model_groups()['C08-parent']
    write(OUT/'control-source-freeze.json',dict(frozen_utc=stamp(),source_sha256=sources(),config_sha256=sha(CONFIG),
        preparation_sha256=sha(OUT/'preparation.json'),bundle=str(bundle.relative_to(ROOT)),arm=arm,
        manifest_sha256=sha(bundle/'c08-manifest.json'),recipe='Original C08/P15 all-mode0.20 tail after complete P01; no labels or model selection.',
        C13_confirmation_metrics_read=False,automatic_submission=False))
    torch.cuda.set_per_process_memory_fraction(.7);assert '2070' in torch.cuda.get_device_name()
    parent=pd.read_csv(OUT/'P01.csv',dtype=str,keep_default_na=False);path=OUT/'P15-scores.parquet';assert not path.exists()
    scores,receipt=module.score(bundle,arm,parent,OUT/'queries.parquet',path,shared_fps=fingerprints())
    assert receipt['scored_molecules']==120 and receipt['input_errors']==receipt['unprocessed_molecules']==0 and not receipt['unsupported_molecules']
    assert scores.c08_ok.astype(bool).all()
    final,actions=apply(parent,scores,.2);assert actions['scored_molecules']==120 and not actions['passthrough_molecules']
    assert actions['rank_one_changes']==actions['candidate_set_changes']==0
    final.to_csv(OUT/'P15.csv',index=False)
    write(OUT/'control-receipt.json',dict(completed_utc=stamp(),source_freeze_sha256=sha(OUT/'control-source-freeze.json'),
        scores_sha256=sha(path),P15_sha256=sha(OUT/'P15.csv'),receipt=receipt,actions=actions,
        C13_confirmation_metrics_read=False,automatic_submission=False))
    print('DONE C25 LABEL-BLIND P15 CONTROL',flush=True)


def control_ready():
    verify_inputs();p=json.loads((OUT/'control-source-freeze.json').read_text())
    assert p['source_sha256']==sources() and sha(CONFIG)==p['config_sha256'] and sha(OUT/'preparation.json')==p['preparation_sha256']
    assert sha(ROOT/p['bundle']/'c08-manifest.json')==p['manifest_sha256']
    r=json.loads((OUT/'control-receipt.json').read_text())
    assert r['source_freeze_sha256']==sha(OUT/'control-source-freeze.json')
    assert r['scores_sha256']==sha(OUT/'P15-scores.parquet') and r['P15_sha256']==sha(OUT/'P15.csv')


def freeze():
    c21.setup();control_ready();c24.verify()
    assert not (OUT/'confirmation-protocol.json').exists() and not (LOCAL/'evaluation-consumption.json').exists()
    selected=json.loads(GLOBAL.read_text());assert selected['new_candidate_selected'] and selected['selected_group']
    assert not selected['C13_confirmation_metrics_read'] and selected['C21_protocol_sha256']==sha(c21.OUT/'development-protocol.json')
    assert selected['source_model_protocol_sha256']==sha(c24.OUT/'development-protocol.json')
    assert selected['C21_development_sha256']==sha(c21.OUT/'development.json') and selected['C24_development_sha256']==sha(c24.OUT/'development.json')
    order=json.loads((c24.OUT/'development-protocol.json').read_text())['global_tie_order']
    assert list(selected['groups'])==order and len(order)==6
    eligible=[]
    for name in order:
        record=selected['groups'][name]
        passed=all(v['delta']>0 and v['ci95'][0]>0 for v in record['comparisons'].values()) and not any(record['truths_lost'].values()) and all(v['delta']>=0 for row in record['polarity'].values() for v in row.values())
        assert bool(passed)==record['supported_development']
        if passed:eligible.append(name)
    assert eligible
    best=max(selected['groups'][n]['comparisons']['P16']['candidate_mrr25'] for n in eligible)
    winner=next(n for n in eligible if selected['groups'][n]['comparisons']['P16']['candidate_mrr25']>=best-1e-12)
    assert winner==selected['selected_group']
    bundle,arm,_=model_groups()[winner]
    if winner=='C24-polarity-loss':expected=json.loads((c24.OUT/'development-protocol.json').read_text())['model_manifest_sha256']
    else:expected=json.loads((c21.OUT/'development-protocol.json').read_text())['serving_models'][winner]['manifest_sha256']
    assert sha(bundle/'c08-manifest.json')==expected
    manifest=json.loads((bundle/'c08-manifest.json').read_text())
    for name,pin in manifest['files'].items():assert sha(bundle/name)==pin,name
    write(OUT/'confirmation-protocol.json',dict(frozen_utc=stamp(),selected_group=winner,bundle=str(bundle.relative_to(ROOT)),arm=arm,
        selected_manifest_sha256=expected,source_sha256=sources(),config_sha256=sha(CONFIG),
        global_selection_sha256=sha(GLOBAL),preparation_sha256=sha(OUT/'preparation.json'),control_receipt_sha256=sha(OUT/'control-receipt.json'),
        C21_protocol_sha256=sha(c21.OUT/'development-protocol.json'),C24_protocol_sha256=sha(c24.OUT/'development-protocol.json'),
        C20_recovery_push_sha256=sha(c21.LOCAL/'push-receipt-v2.json'),labels_sha256=sha(LOCAL/'labels.csv'),
        only_one_selected_recipe=True,C13_confirmation_metrics_read=False,automatic_submission=False,submission_qualified=False))
    print('DONE C25 SELECTED RECIPE FREEZE',winner,flush=True)


def verify():
    control_ready();c24.verify();p=json.loads((OUT/'confirmation-protocol.json').read_text())
    assert p['source_sha256']==sources() and p['config_sha256']==sha(CONFIG)
    for path,key in [(GLOBAL,'global_selection_sha256'),(OUT/'preparation.json','preparation_sha256'),
        (OUT/'control-receipt.json','control_receipt_sha256'),(LOCAL/'labels.csv','labels_sha256'),
        (ROOT/p['bundle']/'c08-manifest.json','selected_manifest_sha256'),
        (c21.OUT/'development-protocol.json','C21_protocol_sha256'),(c24.OUT/'development-protocol.json','C24_protocol_sha256'),
        (c21.LOCAL/'push-receipt-v2.json','C20_recovery_push_sha256')]:assert sha(path)==p[key],key
    return p


def score():
    c21.setup();p=verify();cfg=json.loads(CONFIG.read_text())
    assert not (OUT/'prediction-freeze.json').exists() and not (LOCAL/'evaluation-consumption.json').exists()
    torch.cuda.set_per_process_memory_fraction(.7);assert '2070' in torch.cuda.get_device_name()
    bundle,arm,module=model_groups()[p['selected_group']]
    q=pd.read_parquet(OUT/'queries.parquet');modes=dict(zip(q.molecule_id,q.ionization_mode));shared=fingerprints()
    prior_parent=None;prior_scores=None;receipts={};changes={}
    for view in cfg['reference_views']:
        parent=pd.read_csv(OUT/('P16-'+view+'.csv'),dtype=str,keep_default_na=False)
        negative=parent[parent.molecule_id.map(modes).eq('negative')].copy()
        if prior_parent is not None and negative.equals(prior_parent):
            values=prior_scores;receipt=receipts['all_reference']['inference'];reused=True
        else:
            path=OUT/(view+'-scores.parquet');assert not path.exists()
            values,receipt=module.score(bundle,arm,negative,OUT/'negative-queries.parquet',path,shared_fps=shared)
            assert receipt['scored_molecules']==60 and receipt['input_errors']==receipt['unprocessed_molecules']==0 and not receipt['unsupported_molecules']
            assert values.c08_ok.astype(bool).all() and values.molecule_id.nunique()==60
            reused=False
        final,actions=merge(parent,values,modes,cfg['weight'])
        assert actions['positive_exact_passthrough']==60 and actions['scored_molecules']==60 and not actions['passthrough_molecules']
        assert actions['rank_one_changes']==actions['candidate_set_changes']==0
        final.to_csv(OUT/('candidate-'+view+'.csv'),index=False)
        receipts[view]=dict(inference=receipt,reused_exact_all_reference_negative_scores=reused,
            scores_sha256=sha(OUT/(('all_reference' if reused else view)+'-scores.parquet')))
        changes[view]=actions;prior_parent=negative;prior_scores=values
    files=['P01.csv','P14.csv','P15.csv']+['P16-'+v+'.csv' for v in cfg['reference_views']]+['candidate-'+v+'.csv' for v in cfg['reference_views']]
    write(OUT/'prediction-freeze.json',dict(frozen_utc=stamp(),protocol_sha256=sha(OUT/'confirmation-protocol.json'),
        files_sha256={n:sha(OUT/n) for n in files},inference=receipts,changes=changes,
        C13_confirmation_metrics_read=False,automatic_submission=False))
    print('DONE C25 FULL120 PREDICTION FREEZE',flush=True)


def evaluate():
    import msvcrt
    c21.setup();p=verify();cfg=json.loads(CONFIG.read_text())
    lock=(OUT/'confirmation.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'confirmation.json').exists() and not (LOCAL/'evaluation-consumption.json').exists(),'Reserved120 already consumed; never reselect or repeat blind'
    frozen=json.loads((OUT/'prediction-freeze.json').read_text());assert frozen['protocol_sha256']==sha(OUT/'confirmation-protocol.json')
    for n,pin in frozen['files_sha256'].items():assert sha(OUT/n)==pin,n
    for view,entry in frozen['inference'].items():
        origin='all_reference' if entry['reused_exact_all_reference_negative_scores'] else view
        assert sha(OUT/(origin+'-scores.parquet'))==entry['scores_sha256']
    frames={n:pd.read_csv(OUT/(n+'.csv'),dtype=str,keep_default_na=False) for n in ['P01','P14','P15']}
    for view in cfg['reference_views']:
        for kind in ['P16','candidate']:frames[kind+'-'+view]=pd.read_csv(OUT/(kind+'-'+view+'.csv'),dtype=str,keep_default_na=False)
    q=pd.read_parquet(OUT/'queries.parquet');ids=q.molecule_id.tolist()
    assert all(f.molecule_id.tolist()==ids for f in frames.values())
    # A durable global marker is written BEFORE first label parsing. Any later
    # failure requires an explicit evidence-based recovery of this SAME recipe.
    with (LOCAL/'evaluation-consumption.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(consumed_utc=stamp(),campaign='C25CONF',selected_group=p['selected_group'],
            protocol_sha256=sha(OUT/'confirmation-protocol.json'),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),
            other_recipe_reselection_permitted=False,automatic_submission=False),stream,indent=2)
    labels=pd.read_csv(LOCAL/'labels.csv',dtype=str).set_index('molecule_id').loc[ids]
    assert len(labels)==labels.key.nunique()==120 and labels.index.is_unique and labels.partition.eq('confirmation').all()
    assert labels['mode'].tolist()==q.ionization_mode.tolist()
    index=pd.read_parquet(OUT/'fingerprint-smiles.parquet');identity=dict(zip(index.smiles,index.metric_key))
    values={n:metric_vector(frame,labels,identity) for n,frame in frames.items()}
    comparisons={};polarity={};truth_losses={}
    for view,prefix in [('all_reference','candidate'),('input_alias_excluded','alias_excluded')]:
        candidate=values['candidate-'+view]
        for parent in ['P01','P16']:
            baseline=values['P01' if parent=='P01' else 'P16-'+view]
            comparisons[prefix+'_vs_'+parent]=paired_summary(baseline,candidate)
            truth_losses[view+':'+parent]=int(((baseline>0)&(candidate==0)).sum())
            for mode in ['positive','negative']:
                mask=q.ionization_mode.eq(mode).to_numpy();assert int(mask.sum())==60
                polarity[view+':'+mode+':'+parent]=paired_summary(baseline[mask],candidate[mask])
    decision=gate(comparisons,polarity)
    controls={n:dict(aggregate=paired_summary(values['P01'],values[n]),polarity={mode:paired_summary(values['P01'][q.ionization_mode.eq(mode).to_numpy()],values[n][q.ionization_mode.eq(mode).to_numpy()]) for mode in ['positive','negative']}) for n in ['P14','P15']}
    pd.DataFrame(dict(molecule_id=ids,mode=q.ionization_mode,**values)).to_csv(OUT/'confirmation-predictions.csv',index=False)
    write(OUT/'confirmation.json',dict(completed_utc=stamp(),queries=120,positive=60,negative=60,selected_group=p['selected_group'],
        comparisons=comparisons,polarity=polarity,statistical_gate=decision,controls=controls,
        detects_known_public_regression={n:controls[n]['aggregate']['delta']<0 for n in controls},
        truth_coverage={n:int((v>0).sum()) for n,v in values.items()},truths_lost=truth_losses,truth_absent_mrr=0,
        positive_parent_preserved_exact=True,P14_supported=59,P14_explicit_fallback=61,P14_negative_noop_disclosed=True,
        protocol_sha256=sha(OUT/'confirmation-protocol.json'),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),
        consumption_sha256=sha(LOCAL/'evaluation-consumption.json'),C13_confirmation_metrics_read=True,
        confirmation_recipe_reselection=False,broader_pretraining_exposure_excluded=False,automatic_submission=False,submission_qualified=False,
        limitation=cfg['source_limitations'],next_action='Require real Kaggle changed-path smoke and independent execution/model/identity/fallback/fusion/parity audits before public trial.' if decision['passed'] else 'Reject the frozen recipe, preserve0.480champion and approval; C13mixed120 is now used development and cannot be called fresh.'))
    print('DONE C25 FROZEN CONFIRMATION',json.dumps(json.loads((OUT/'confirmation.json').read_text())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['controls','freeze','score','evaluate']);args=parser.parse_args()
    {'controls':controls,'freeze':freeze,'score':score,'evaluate':evaluate}[args.stage]()
