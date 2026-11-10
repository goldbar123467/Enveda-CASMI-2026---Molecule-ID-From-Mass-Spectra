"""Prepare label-blind C13 development/fresh-confirmation scoring pools."""
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
from external_reference_overlay import search,apply_overlay
from build_external_cache import OFFSETS,normalize_adduct
from training_monitor import system_memory

OUT = ROOT / 'outputs/local/c13-lotus-agreement-20261110'
CONFIG = ROOT / 'configs/c13-lotus-agreement-20261110.json'
DEV = ROOT / 'outputs/forks/c10-casmi2022-validation-v1'
CONF = ROOT / 'outputs/forks/c13-mixed-casmi2022-validation-v1'
SOURCE = ROOT / 'outputs/research/c12-lotus-20261110'
COVERAGE = ROOT / 'outputs/local/c13-lotus-development-20261110'


def cloud(scope): return DEV if scope == 'development' else CONF


def main(scope):
    pa.set_cpu_count(2);assert rdBase.rdkitVersion == '2026.03.3'
    cfg=json.loads(CONFIG.read_text());assert system_memory()['free_gib'] >= cfg['minimum_free_ram_gib']
    assert not (ROOT/'.local/compute-paused.json').exists()
    out=OUT/scope;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'preparation.json').exists(),'Prepared scope is sealed'
    audit_path=ROOT/'outputs/local/c13-lotus-source-20261110/audit.json'
    audit=json.loads(audit_path.read_text());assert audit['traced_mirror_rows']==149983 and audit['unresolved_mirror_rows']==0
    contract=dict(config_sha256=sha(CONFIG),source_sha256={n:sha(ROOT/'scripts'/n) for n in
        ['prepare_c13_pool.py','score_c13_pool.py','agreement_expansion.py']},
        feature_sha256=sha(ROOT/'outputs/local/kit/features.py'),
        loader_sha256=sha(ROOT/'outputs/research/c08-pipeline/modules/neural_scores.py'),
        model_audit_sha256=sha(ROOT/'outputs/forks/p16-external-reference-c09-v2/p16_execution_receipt.json'),
        source_audit_sha256=sha(audit_path),catalogue_sha256=sha(SOURCE/'candidates_lotus.parquet'),
        P16_policy_sha256=sha(ROOT/'configs/c09-external-reference-20261109.json'))
    if (OUT/'source-contract.json').exists():assert json.loads((OUT/'source-contract.json').read_text())==contract
    else:write(OUT/'source-contract.json',contract)
    path=cloud(scope);prefix='c10' if scope=='development' else 'c13'
    baseline=pd.read_csv(path/(prefix+'_p01.csv'),dtype=str,keep_default_na=False)
    original=pd.read_parquet(path/(prefix+'_neural_p01.parquet'))
    queries=pd.read_parquet(path/(prefix+'_queries.parquet'))
    assert baseline.molecule_id.is_unique and len(baseline)==(160 if scope=='development' else 120)
    assert set(queries.molecule_id)==set(baseline.molecule_id)
    confidence=original.groupby('molecule_id').library_max.first().to_dict()
    cache={}
    if scope=='development':
        old=ROOT/'outputs/local/c12-deep-candidates-20261110/all'
        for folder in [old,COVERAGE]:
            index=pd.read_parquet(folder/'fingerprint-smiles.parquet');fps=np.load(folder/'fingerprints.npy',mmap_mode='r')
            for i,row in enumerate(index.itertuples(index=False)):cache[str(row.smiles)]=(str(row.metric_key),fps[i])
    def structure(smiles):
        if smiles not in cache:
            value=features.structure(smiles,2048);cache[smiles]=(value[0],value[-1])
        return cache[smiles]
    def identity(smiles):return structure(smiles)[0]
    policy=json.loads((ROOT/'configs/c09-external-reference-20261109.json').read_text())
    if scope=='development':
        matches=pd.read_parquet(ROOT/'outputs/local/c10-candidates-20261110/matches.parquet')
        matches=matches[matches.recipe.eq('P16')].drop(columns='recipe')
    else:
        reference=ROOT/'outputs/research/2026-11-09-silver/extlib'
        pins=json.loads((ROOT/'outputs/local/c09-external-reference-20261109/development.json').read_text())['reference_sha256']
        for name,pin in pins.items():assert sha(reference/name)==pin
        matches,report=search(queries,reference,features,policy);write(out/'reference-search.json',report)
    matches.to_parquet(out/'reference-matches.parquet',index=False)
    parent,_=apply_overlay(baseline,matches[matches.view.eq('all_reference')],confidence,policy,identity)
    if scope=='development':
        expected=pd.read_csv(ROOT/'outputs/local/c12-deep-candidates-20261110/all/parent.csv',dtype=str,keep_default_na=False)
        assert parent.equals(expected)
        lotus=pd.read_parquet(COVERAGE/'lotus-candidates.parquet')
    else:
        catalogue=pd.read_parquet(SOURCE/'candidates_lotus.parquet');masses=catalogue.monoisotopic_mass.to_numpy();extra=[]
        for row in queries.itertuples(index=False):
            adduct=normalize_adduct(row.adduct)
            if adduct not in OFFSETS or (adduct.endswith('+')!=(str(row.ionization_mode).lower()=='positive')):continue
            charge,offset=OFFSETS[adduct];neutral=row.precursor_mz*charge-offset
            width=(row.precursor_mz*cfg['precursor_ppm']*1e-6+cfg['precursor_absolute_da'])*charge
            positions=np.arange(np.searchsorted(masses,neutral-width,'left'),np.searchsorted(masses,neutral+width,'right'))
            positions=sorted(positions,key=lambda i:(abs((masses[i]+offset)/charge-row.precursor_mz),int(i)))
            taken=set()
            for pos in positions:
                smiles=str(catalogue.smiles.iloc[pos]);key=identity(smiles)
                if key in taken:continue
                taken.add(key);extra.append(dict(molecule_id=row.molecule_id,smiles=smiles,metric_key=key,source_rank=len(taken)))
        lotus=pd.DataFrame(extra,columns=['molecule_id','smiles','metric_key','source_rank'])
    records=[]
    for row in parent.itertuples(index=False):
        mid=row.molecule_id;extra=lotus[lotus.molecule_id.eq(mid)].sort_values('source_rank',kind='stable')
        lotus_keys=set(extra.metric_key);taken=set();pool=[]
        def add(smiles,is_parent=False):
            key=identity(smiles)
            if key in taken:return
            taken.add(key);pool.append(dict(molecule_id=mid,smiles=smiles,metric_key=key,prior_rank=len(pool)+1,
                is_parent=is_parent,is_lotus=key in lotus_keys,library_max=float(confidence[mid]),processed=True))
        for smiles in row.smiles.split(';'):add(smiles,True)
        assert len(pool)==len(row.smiles.split(';'))
        for smiles in original[original.molecule_id.eq(mid)].sort_values('baseline_rank',kind='stable').smiles:add(str(smiles))
        for smiles in extra.smiles:
            if len(pool)>=cfg['maximum_total_candidates']:break
            add(str(smiles))
        records.extend(pool)
    frame=pd.DataFrame(records);smiles=frame.smiles.drop_duplicates().tolist();lookup={s:i for i,s in enumerate(smiles)}
    frame['fp_index']=frame.smiles.map(lookup)
    frame.to_parquet(out/'pool.parquet',index=False);parent.to_csv(out/'parent.csv',index=False)
    queries.to_parquet(out/'queries.parquet',index=False)
    np.save(out/'fingerprints.npy',np.stack([cache[s][1] for s in smiles]).astype(np.uint8))
    pd.DataFrame(dict(smiles=smiles,metric_key=[cache[s][0] for s in smiles])).to_parquet(out/'fingerprint-smiles.parquet',index=False)
    names=['pool.parquet','parent.csv','queries.parquet','fingerprints.npy','fingerprint-smiles.parquet','reference-matches.parquet']
    write(out/'preparation.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),scope=scope,
        queries=len(parent),candidates=len(frame),unique_smiles=len(smiles),maximum_query_candidates=int(frame.groupby('molecule_id').size().max()),
        source_contract_sha256=sha(OUT/'source-contract.json'),files={n:sha(out/n) for n in names},
        query_labels_read=False,prediction_metrics_computed=False,automatic_submission=False))
    print('DONE C13 LABELBLIND POOL',scope,len(parent),len(frame),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['development','confirmation'],required=True)
    main(parser.parse_args().scope)
