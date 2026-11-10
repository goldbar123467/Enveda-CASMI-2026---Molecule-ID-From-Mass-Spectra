"""Label-blind C13 full120 parent/reference/feature preparation; no metrics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase
from precision_cache import ROOT,sha,write,features
from external_reference_overlay import search,apply_overlay
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c25-reserved-negative-20261110'
CLOUD=ROOT/'outputs/forks/c13-mixed-casmi2022-validation-v2'
LOCAL=ROOT/'outputs/local/c13-confirmation-20261110'
INPUTS=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110'
CONFIG=ROOT/'configs/c25-reserved-negative-20261110.json'
POLICY=ROOT/'configs/c09-external-reference-20261109.json'
REFERENCE=ROOT/'outputs/research/2026-11-09-silver/extlib'
CACHE=ROOT/'outputs/local/c12-deep-candidates-20261110/all'
SOURCES=['prepare_c25_reserved_inputs.py','external_reference_overlay.py','c09_spectrum_audit.py']


def stamp():return datetime.now(timezone.utc).isoformat()


def main():
    pa.set_cpu_count(2);assert rdBase.rdkitVersion=='2026.03.3' and system_memory()['free_gib']>=4
    assert not (ROOT/'.local/compute-paused.json').exists()
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'preparation.json').exists() and not (OUT/'preparation-intent.json').exists(),'Inspect any prior preparation before recovery'
    cfg=json.loads(CONFIG.read_text());assert (cfg['queries'],cfg['positive'],cfg['negative'],cfg['weight'],cfg['offset'])==(120,60,60,.2,3.)
    audit=json.loads((LOCAL/'execution-v2-audit.json').read_text())
    assert audit['version']==2 and audit['queries']==120 and audit['engine2_covered_ids']==audit['engine2_nonempty_lists']==120
    assert audit['all_input_ids_retained'] and audit['baseline_replay_exact'] and not audit['fresh_prediction_label_metrics_read']
    for name,pin in audit['artifact_sha256'].items():assert sha(CLOUD/name)==pin,name
    assert sha(CLOUD/'c13_p01.csv')==audit['baseline_sha256']
    report=json.loads((CLOUD/'c13_report.json').read_text());execution=json.loads((CLOUD/'execution_receipt.json').read_text())
    assert report['queries']==120 and report['baseline_replay_exact'] and execution['is_full_run'] and execution['processed_molecules']==120
    assert not report['metrics_computed'] and not report['query_labels_mounted'] and not report['recipe_selection']
    assert report['validation_contract_sha256']==sha(INPUTS/'validation-contract.json')
    contract=json.loads((INPUTS/'validation-contract.json').read_text())
    assert contract['labels_sha256']==sha(LOCAL/'labels.csv')
    for name,pin in contract['files'].items():assert sha(INPUTS/name)==pin,name
    q=pd.read_parquet(CLOUD/'c13_queries.parquet');original=pd.read_parquet(INPUTS/'validation-queries.parquet')
    assert len(q)==q.molecule_id.nunique()==120 and q.ionization_mode.value_counts().to_dict()=={'positive':60,'negative':60}
    # Engine2 adds its neutral-mass column. Every original query field remains exact.
    pd.testing.assert_frame_equal(q[original.columns],original,check_exact=True)
    parent=pd.read_csv(CLOUD/'c13_p01.csv',dtype=str,keep_default_na=False)
    control=pd.read_csv(CLOUD/'c13_p14.csv',dtype=str,keep_default_na=False)
    assert parent.molecule_id.tolist()==q.molecule_id.tolist()==control.molecule_id.tolist()
    assert parent.columns.tolist()==control.columns.tolist()==['molecule_id','smiles']
    p14=report['p14_receipt'];assert p14['scored_molecules']==59 and p14['input_errors']==0 and len(p14['coverage'])==120
    supported={str(r['molecule_id']) for r in p14['coverage'] if r['supported_spectra']>0}
    fallback=~parent.molecule_id.isin(supported);assert int(fallback.sum())==61
    assert control.loc[fallback].equals(parent.loc[fallback])
    assert not set(q[q.ionization_mode.eq('negative')].molecule_id)&supported
    evidence=pd.read_parquet(CLOUD/'c13_neural_p01.parquet')
    assert evidence.processed.astype(bool).all() and evidence.groupby('molecule_id').library_max.nunique().eq(1).all()
    confidence=evidence.groupby('molecule_id').library_max.first().to_dict();assert set(confidence)==set(q.molecule_id)
    for name in ['neural_public.parquet','neural_ours.parquet']:
        neural=pd.read_parquet(CLOUD/name);assert neural.molecule_id.nunique()==120 and neural.neural_ok.astype(bool).all()
    production=ROOT/'outputs/forks/p16-external-reference-c09-v2/p16_execution_receipt.json'
    served=json.loads(production.read_text());assert sha(POLICY)==served['policy_sha256']
    for name,pin in served['reference_files'].items():assert sha(REFERENCE/name)==pin,name
    cache_prep=json.loads((CACHE/'preparation.json').read_text());source_contract=CACHE.parent/'source-contract.json'
    assert json.loads(source_contract.read_text())['feature_sha256']==sha(ROOT/'outputs/local/kit/features.py')
    for name in ['fingerprint-smiles.parquet','fingerprints.npy']:assert sha(CACHE/name)==cache_prep['files'][name]
    inputs=[CLOUD/n for n in ['c13_report.json','c13_p01.csv','c13_p14.csv','c13_queries.parquet','c13_neural_p01.parquet','execution_receipt.json','neural_public.parquet','neural_ours.parquet']]
    inputs+=[LOCAL/'execution-v2-audit.json',LOCAL/'labels.csv',INPUTS/'validation-contract.json',INPUTS/'validation-queries.parquet',production,source_contract,CACHE/'preparation.json',CACHE/'fingerprint-smiles.parquet',CACHE/'fingerprints.npy']
    write(OUT/'preparation-intent.json',dict(prepared_utc=stamp(),config_sha256=sha(CONFIG),policy_sha256=sha(POLICY),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in SOURCES},feature_sha256=sha(ROOT/'outputs/local/kit/features.py'),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},reference_files_sha256=served['reference_files'],
        label_values_accessed=False,C13_prediction_metrics_read=False,recipe_selected=False,automatic_submission=False))
    matches,search_report=search(q,REFERENCE,features,json.loads(POLICY.read_text()))
    assert len(matches)==240 and matches.groupby('view').molecule_id.nunique().eq(120).all()
    matches.to_parquet(OUT/'reference-matches.parquet',index=False)
    write(OUT/'reference-search.json',search_report)
    index=pd.read_parquet(CACHE/'fingerprint-smiles.parquet');fp=np.load(CACHE/'fingerprints.npy',mmap_mode='r')
    lookup=dict(zip(index.smiles,range(len(index))));structures={};reused=0
    def identity(smiles):
        nonlocal reused
        if smiles not in structures:
            if smiles in lookup:
                i=lookup[smiles];structures[smiles]=(str(index.metric_key.iloc[i]),np.asarray(fp[i]));reused+=1
            else:
                value=features.structure(smiles);structures[smiles]=(value[0],value[-1])
        return structures[smiles][0]
    for frame in [parent,control]:
        for row in frame.itertuples(index=False):
            keys=[identity(s) for s in row.smiles.split(';')];assert 1<=len(keys)<=25 and len(keys)==len(set(keys))
    parent.to_csv(OUT/'P01.csv',index=False);control.to_csv(OUT/'P14.csv',index=False)
    q.to_parquet(OUT/'queries.parquet',index=False)
    q[q.ionization_mode.eq('negative')].to_parquet(OUT/'negative-queries.parquet',index=False)
    actions={};files=['P01.csv','P14.csv','queries.parquet','negative-queries.parquet','reference-matches.parquet','reference-search.json']
    for view in cfg['reference_views']:
        overlay,changes=apply_overlay(parent,matches[matches.view.eq(view)],confidence,json.loads(POLICY.read_text()),identity)
        assert not changes.reason.isin(['invalid_external_structure','missing_pipeline_library_confidence']).any()
        overlay.to_csv(OUT/('P16-'+view+'.csv'),index=False);changes.to_csv(OUT/('actions-'+view+'.csv'),index=False)
        for row in overlay.itertuples(index=False):
            keys=[identity(s) for s in row.smiles.split(';')];assert 1<=len(keys)<=25 and len(keys)==len(set(keys))
        actions[view]=dict(changed=int(changes.changed.sum()),reasons=changes.reason.value_counts().to_dict())
        files+=['P16-'+view+'.csv','actions-'+view+'.csv']
    names=list(structures);pd.DataFrame(dict(smiles=names,metric_key=[structures[s][0] for s in names])).to_parquet(OUT/'fingerprint-smiles.parquet',index=False)
    np.save(OUT/'fingerprints.npy',np.stack([structures[s][1] for s in names]).astype(np.uint8))
    files+=['fingerprint-smiles.parquet','fingerprints.npy']
    assert system_memory()['free_gib']>=4
    write(OUT/'preparation.json',dict(completed_utc=stamp(),preparation_intent_sha256=sha(OUT/'preparation-intent.json'),
        files_sha256={n:sha(OUT/n) for n in files},queries=120,positive=60,negative=60,original_query_fields_exact=True,
        Engine2_complete120=True,P14_supported=59,P14_exact_fallback=61,P14_negative_noop_disclosed=True,
        feature_structures=len(names),existing_feature_rows_reused=reused,reference_search=search_report,actions=actions,
        label_values_accessed=False,C13_prediction_metrics_read=False,recipe_selected=False,automatic_submission=False))
    print('DONE C25 LABEL-BLIND RESERVED INPUTS',json.dumps(json.loads((OUT/'preparation.json').read_text())),flush=True)


def verify():
    p=json.loads((OUT/'preparation.json').read_text());intent=json.loads((OUT/'preparation-intent.json').read_text())
    assert p['preparation_intent_sha256']==sha(OUT/'preparation-intent.json') and sha(CONFIG)==intent['config_sha256'] and sha(POLICY)==intent['policy_sha256']
    assert sha(ROOT/'outputs/local/kit/features.py')==intent['feature_sha256']
    for n,pin in intent['source_sha256'].items():assert sha(ROOT/'scripts'/n)==pin,n
    for n,pin in intent['input_sha256'].items():assert sha(ROOT/n)==pin,n
    for n,pin in intent['reference_files_sha256'].items():assert sha(REFERENCE/n)==pin,n
    for n,pin in p['files_sha256'].items():assert sha(OUT/n)==pin,n
    return p


if __name__=='__main__':main()
