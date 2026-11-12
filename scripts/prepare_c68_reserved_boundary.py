"""Quarantine the ENTIRE public231identity C13 inventory from all fitting.

Read only public source SMILES and the global key set, never a reserved query
mapping, selected120truth file, prediction or metric. Preserve all869evaluation
queries and every candidate. The boundary acts only on training pairs/scaling.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import tarfile
from functools import lru_cache
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp,module
DATA=ROOT/'outputs/local/c66-parent-evidence-oof-20261112'
OUT=ROOT/'outputs/local/c68-reserved-fit-boundary-20261112'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'boundary-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    RDLogger.DisableLog('rdApp.*')
    archive=ROOT/'outputs/research/c10-casmi2022-20261110/casmi_2022.tgz'
    inv=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet'
    manifest=read(ROOT/'outputs/local/c13-confirmation-20261110/inventory.json')
    assert sha(archive)==manifest['source_sha256'] and sha(inv)==manifest['inventory_sha256']
    reserved=set(pd.read_parquet(inv,columns=['key']).key);assert len(reserved)==231
    write(OUT/'protocol.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),
        rule='All231global public-inventory identities plus sameformula/nonemptyMurcko/Morgan2Tanimoto>=.70; remove connected query chemistry components from fitting only and quarantine candidate rows from pair loss AND global feature scaling. No query/candidate is removed from evaluation.',
        public_SMILES_only=True,query_structure_mapping_read=False,C13_selected120_labels_read=False,C13_outcomes_read=False,
        original_data_freeze_sha256=sha(DATA/'data-freeze.json'),archive_sha256=sha(archive),global_inventory_sha256=sha(inv)))
    identity=module('c68_public_identity','c09e_features.py').structure
    with tarfile.open(archive,'r:gz') as source:
        meta=pd.read_csv(source.extractfile('casmi_2022/all_meta_results.csv'),header=1,usecols=['SMILES'])
    resolved={}
    for text in meta.SMILES.drop_duplicates():
        key,canonical,formula,*_=identity(text)
        if key in reserved:resolved[key]=dict(key=key,smiles=canonical,formula=formula)
    assert set(resolved)==reserved,('Unresolved global reserved identities',len(reserved-set(resolved)))
    # This file deliberately contains no challenge/source number, mode or
    # reserved query identifier. Its sole downstream role is global quarantine.
    catalogue=pd.DataFrame(resolved.values()).sort_values('key').reset_index(drop=True)
    catalogue.to_parquet(OUT/'global-exclusion-chemistry.parquet',index=False)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    @lru_cache(maxsize=40000)
    def desc(text):
        m=Chem.MolFromSmiles(text);assert m is not None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False),gen.GetFingerprint(m)
    described=[desc(s) for s in catalogue.smiles];forms=set(catalogue.formula);scaffolds={v[0] for v in described if v[0]};fps=[v[1] for v in described]
    @lru_cache(maxsize=40000)
    def reason(key,formula,text):
        if key in reserved:return 'global_reserved_identity'
        if formula in forms:return 'global_reserved_formula'
        sc,fp=desc(text)
        if sc and sc in scaffolds:return 'global_reserved_scaffold'
        if max(DataStructs.BulkTanimotoSimilarity(fp,fps))>=.7:return 'global_reserved_near_chemistry'
        return None
    q=pd.read_parquet(DATA/'queries.parquet');rows=pd.read_parquet(DATA/'candidates.parquet')
    q['reserved_connection']=[reason(r.key,r.formula,r.smiles) for r in q.itertuples(index=False)]
    blocked_groups=set(q.loc[q.reserved_connection.notna(),'chemistry_group'])
    q['fitting_eligible']=~q.chemistry_group.isin(blocked_groups)
    rows['reserved_connection']=[reason(r.key,r.formula,r.smiles) for r in rows.itertuples(index=False)]
    query_eligible=q.set_index('molecule_id').fitting_eligible
    rows['fitting_eligible']=rows.reserved_connection.isna()&rows.molecule_id.map(query_eligible)
    assert len(q)==869 and len(rows)==21719 and q.acquisitions.sum()==3452
    assert not set(rows.loc[rows.fitting_eligible,'key'])&reserved
    assert q.fitting_eligible.sum()>500 and rows.fitting_eligible.sum()>10000
    q.to_parquet(OUT/'queries.parquet',index=False);rows.to_parquet(OUT/'candidates.parquet',index=False)
    contamination=dict(checked_utc=stamp(),original_feature_candidate_reserved_identity_overlap=int(rows.key.isin(reserved).sum()),
        original_reserved_positive_overlap=int((rows.key.isin(reserved)&rows.target.eq(1)).sum()),
        old_C66_C67_require_refit=True,qualification_held=True,C13_outcomes_read=False,scoring_entries=0)
    write(OUT/'prior-pair-reserved-audit.json',contamination)
    files=[DATA/'data-freeze.json',DATA/'features.npy',DATA/'feature-names.json',inv,archive,OUT/'protocol.json',OUT/'global-exclusion-chemistry.parquet',OUT/'queries.parquet',OUT/'candidates.parquet']
    write(OUT/'boundary-freeze.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),queries=869,acquisitions=3452,candidates=21719,
        fitting_queries=int(q.fitting_eligible.sum()),fitting_candidate_rows=int(rows.fitting_eligible.sum()),global_reserved_identities=len(reserved),resolved_global_reserved_chemistry=len(resolved),
        query_first_exclusion_reasons=q.reserved_connection.value_counts().to_dict(),candidate_first_exclusion_reasons=rows.reserved_connection.value_counts().to_dict(),
        excluded_query_chemistry_components=len(blocked_groups),every_original_evaluation_query_and_candidate_retained=True,
        fitting_reserved_identity_or_chemistry_overlap=0,global_normalization_must_use_only_fitting_eligible_rows=True,
        C13_selected_labels_or_outcomes_read=False,models_fitted=False,scoring_entries=0,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('C68 RESERVED FIT BOUNDARY READY',q.fitting_eligible.sum(),'fit queries',rows.fitting_eligible.sum(),'fit rows; all869/21719evaluated',flush=True)


if __name__=='__main__':main()
