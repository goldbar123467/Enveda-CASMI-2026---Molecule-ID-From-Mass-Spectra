"""Global outcome-blind main/Engine2 chemistry and acquisition quarantine.

Preserve every prior C56reference exclusion. Add all USED/fresh512/global231
reserved chemistry, then propagate original raw/official aliases to closure.
No per-query answer mapping is packaged for inference.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from prepare_c48_fit_boundary import alias_closure
from training_monitor import system_memory
OUT=ROOT/'outputs/local/c73-clean-target-validation-20261112'
PANEL=ROOT/'outputs/local/c72-clean-target-holdout-20261112'
OLD=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
CACHE=ROOT/'outputs/local/long-frozen-20261107-r1'
FAMILY=ROOT/'outputs/local/c71-final-clean-direct-pair-20261112'
REFERENCE=ROOT/'outputs/research/c08-pipeline/reference/train_structs.parquet'


def guard():
    assert not (PANEL/'evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    binding=read(FAMILY/'complete-pipeline-binding.json')
    for name,pin in binding['files_sha256'].items():assert sha(ROOT/name)==pin
    panel=read(PANEL/'validation-freeze.json')
    for name,pin in panel['files_sha256'].items():assert sha(ROOT/name)==pin
    return panel


def classify():
    guard();OUT.mkdir(exist_ok=True);assert not (OUT/'chemistry-classification-freeze.json').exists()
    assert system_memory()['free_gib']>=4;RDLogger.DisableLog('rdApp.*')
    inputs=[ROOT/'outputs/local/c69-clean-parent-evidence-oof-20261112/queries.parquet',PANEL/'labels.csv',
        ROOT/'outputs/local/c68-reserved-fit-boundary-20261112/global-exclusion-chemistry.parquet']
    old_paths=['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
        'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']
    inputs += [ROOT/p for p in old_paths]
    frames=[pd.read_parquet(inputs[0])[['key','smiles','formula']],pd.read_csv(inputs[1],usecols=['key','smiles','formula']),pd.read_parquet(inputs[2])]
    frames += [pd.read_csv(ROOT/p,usecols=lambda n:n in {'key','smiles','formula'}) for p in old_paths]
    held=pd.concat(frames,ignore_index=True).drop_duplicates('key');held.to_parquet(OUT/'global-exclusion-chemistry.parquet',index=False)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    ms=[Chem.MolFromSmiles(s) for s in held.smiles];assert all(m is not None for m in ms)
    fps=[gen.GetFingerprint(m) for m in ms];keys=set(held.key);forms=set(held.formula)
    scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in ms};scaffolds.discard('')
    witness=OLD/'original-structure-identity-witnesses.parquet';w=pd.read_parquet(witness)
    oldofficial=set(w.loc[w.exclude,'official_key'].dropna());rawfile=OLD/'engine2-raw-key-exclusion.json'
    previous_raw=set(read(rawfile)['keys'])
    oldofficial.update(w.loc[w.raw_key.isin(previous_raw),'official_key'].dropna())
    mol=pd.read_parquet(CACHE/'molecules.parquet');assert sha(CACHE/'molecules.parquet')==read(CACHE/'prepared.json')['mol_id_order_sha256']
    canonical=mol.drop_duplicates('key').set_index('key')[['smiles','formula']].to_dict('index')
    metric=module('c73_extra_identity','c09e_features.py').structure
    source_pin=sha(__file__)
    write(OUT/'reference-mask-protocol.json',dict(started_utc=stamp(),source_sha256=source_pin,
        rule='Preserve old60280raw-key quarantine; add ALL USED, fresh512 and ENTIRE global231reserved chemistry by officialidentity/formula/nonemptyMurcko/Morgan2>=.70. Add full/top96acquisition aliases, then close raw/official identity graph. Main SID/global Engine2mask only; never add answers.',
        complete_pipeline_binding_sha256=sha(FAMILY/'complete-pipeline-binding.json'),panel_freeze_sha256=sha(PANEL/'validation-freeze.json'),
        C13_query_mapping_or_outcomes_read=False,quality_outcomes_read=False))
    records=[];unique=w[w.official_key.notna()].drop_duplicates('official_key')
    for i,row in enumerate(unique.itertuples(index=False),1):
        key=row.official_key;reason=None;maximum=None
        if key in oldofficial:reason='prior_reference_quarantine'
        elif key in keys:reason='used_fresh_or_global_reserved_identity'
        else:
            d=canonical.get(key)
            if d is None:
                actual,text,formula,*_=metric(row.smiles);assert actual==key;d=dict(smiles=text,formula=formula)
            if d['formula'] in forms:reason='used_fresh_or_global_reserved_formula'
            else:
                m=Chem.MolFromSmiles(d['smiles']);assert m is not None
                scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
                if scaffold and scaffold in scaffolds:reason='used_fresh_or_global_reserved_scaffold'
                else:
                    maximum=max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),fps))
                    if maximum>=.7:reason='used_fresh_or_global_reserved_near_chemistry'
        records.append(dict(key=key,excluded=reason is not None,reason=reason,maximum_global_tanimoto=maximum))
        if i%10000==0:
            assert system_memory()['free_gib']>=4
            write(OUT/'reference-mask-progress.json',dict(checked_utc=stamp(),pid=os.getpid(),stage='global_chemistry',completed=i,total=len(unique)))
            if i%50000==0:print('C73 REFERENCE CHEMISTRY',i,'/',len(unique),flush=True)
    table=pd.DataFrame(records);table.to_parquet(OUT/'reference-chemistry-classification.parquet',index=False)
    assert sha(__file__)==source_pin
    files=[Path(__file__),witness,rawfile,CACHE/'molecules.parquet',*inputs,OUT/'global-exclusion-chemistry.parquet',OUT/'reference-chemistry-classification.parquet',OUT/'reference-mask-protocol.json']
    write(OUT/'chemistry-classification-freeze.json',dict(completed_utc=stamp(),source_sha256=source_pin,global_excluded_chemistry_identities=len(held),
        original_reference_official_identities=len(table),excluded_official_identities=int(table.excluded.sum()),reasons=table.reason.value_counts().to_dict(),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},quality_outcomes_read=False,C13_query_mapping_read=False))
    print('C73 GLOBAL REFERENCE CHEMISTRY FROZEN',len(table),int(table.excluded.sum()),flush=True)


def finalize():
    guard();assert not (OUT/'reference-exclusion-freeze.json').exists()
    pin=read(OUT/'chemistry-classification-freeze.json')
    assert pin['source_sha256']==sha(__file__)
    for name,value in pin['files_sha256'].items():assert sha(ROOT/name)==value
    aliases=read(PANEL/'raw-source-alias-audit.json');assert aliases['queries']==512 and aliases['queries_removed']==0
    for name,value in aliases['files_sha256'].items():assert sha(PANEL/name)==value
    table=pd.read_parquet(OUT/'reference-chemistry-classification.parquet');w=pd.read_parquet(OLD/'original-structure-identity-witnesses.parquet')
    blocked=set(table.loc[table.excluded,'key']);raw=set(read(OLD/'engine2-raw-key-exclusion.json')['keys'])
    raw.update(read(PANEL/'additional-raw-alias-quarantine.json')['keys'])
    official,raw,rounds=alias_closure(w,blocked,raw)
    st=pd.read_parquet(REFERENCE,columns=['sid','inchikey14','smiles']);assert np.array_equal(st.sid,np.arange(len(st))) and len(st)==275810
    mapping=dict(zip(w.smiles,w.official_key))
    mask=st.inchikey14.astype(str).isin(raw).to_numpy()|np.array([mapping.get(str(s)) is None or mapping.get(str(s)) in official for s in st.smiles])
    previous=np.load(ROOT/'outputs/local/c57-target-validation-20261112/main-reference-sid-exclusion.npy')
    assert mask[previous].all() and not st.loc[~mask,'inchikey14'].astype(str).isin(raw).any()
    np.save(OUT/'main-reference-sid-exclusion.npy',mask)
    write(OUT/'engine2-raw-key-exclusion.json',dict(keys=sorted(raw),scope='Global reference quarantine only; no perquery truth mapping or candidate insertion'))
    w['exclude']=w.official_key.isna()|w.official_key.isin(official)|w.raw_key.isin(raw);w.to_parquet(OUT/'original-structure-identity-witnesses.parquet',index=False)
    files=[OUT/n for n in ['main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json','original-structure-identity-witnesses.parquet']]
    write(OUT/'reference-exclusion-freeze.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),
        panel_freeze_sha256=sha(PANEL/'validation-freeze.json'),classification_freeze_sha256=sha(OUT/'chemistry-classification-freeze.json'),
        raw_alias_audit_sha256=sha(PANEL/'raw-source-alias-audit.json'),complete_pipeline_binding_sha256=sha(FAMILY/'complete-pipeline-binding.json'),
        reference_structure_sha256=sha(REFERENCE),queries=512,acquisitions=2215,reference_SIDs=len(st),excluded_main_SIDs=int(mask.sum()),
        excluded_raw_keys=len(raw),excluded_official_keys=len(official),alias_closure_rounds=rounds,all_prior_reference_exclusions_preserved=True,
        global_all_USED_fresh512_reserved231_chemistry_and_alias_quarantine=True,files_sha256={p.name:sha(p) for p in files},
        query_truth_map_mounted=False,quality_outcomes_read=False,C13_read=False))
    print('C73 REFERENCE MASKS FINALIZED',int(mask.sum()),'main SIDs',len(raw),'raw keys',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['classify','finalize']);args=parser.parse_args()
    (classify if args.phase=='classify' else finalize)()
