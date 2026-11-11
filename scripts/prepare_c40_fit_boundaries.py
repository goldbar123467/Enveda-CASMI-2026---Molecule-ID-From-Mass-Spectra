"""Exclude held-out C39 chemistry and known acquisition aliases from new heads.

Frozen public encoders' historic pretraining exposure is disclosed separately;
this script establishes boundaries for the newly fitted CPU/GPU heads only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json,hashlib
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from precision_cache import ROOT,sha,write
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c40-crossmodal-heads-20261111'
C39=ROOT/'outputs/local/c39-source-supported-validation-20261111'
CACHE=ROOT/'outputs/local/long-frozen-20261107-r1'


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');assert system_memory()['free_gib']>=4
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'fit-boundary-freeze.json').exists()
    freeze=json.loads((C39/'validation-freeze.json').read_text())
    for p,h in freeze['files_sha256'].items():assert sha(ROOT/p)==h,p
    holdout=pd.read_csv(C39/'labels.csv');mol=pd.read_parquet(CACHE/'molecules.parquet')
    old=json.loads((CACHE/'prepared.json').read_text());assert sha(CACHE/'molecules.parquet')==old['mol_id_order_sha256']
    assert len(mol)==273833
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    query_mols=[Chem.MolFromSmiles(s) for s in holdout.smiles];assert all(m is not None for m in query_mols)
    query_fp=[gen.GetFingerprint(m) for m in query_mols]
    keys=set(holdout.key);forms=set(holdout.formula)
    scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in query_mols};scaffolds.discard('')
    group=np.load(ROOT/'outputs/local/long-20261106/spectrum_groups.npz',allow_pickle=False)
    assert len(group['groups'])>=len(mol)
    group_counts=np.bincount(group['groups']);alias=group_counts[group['groups'][:len(mol)]]>1
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=sha(__file__),
        C39_freeze_sha256=sha(C39/'validation-freeze.json'),molecule_order_sha256=sha(CACHE/'molecules.parquet'),
        existing_encoded_feature_manifest_sha256=sha(CACHE/'manifest.json'),
        rule='Exclude any fitted molecule with held-out official identity, same formula, same nonempty scaffold, Morgan radius2 Tanimoto>=0.70, or known acquisition-alias group.',
        no_quality_metrics_read=True,old_head_weights_not_reused=True,reserved_C13_read=False,
        scope='New CPU/GPU head fitting. Frozen pretrained encoders/public parent exposure is not erased or claimed absent.')
    write(OUT/'fit-boundary-protocol.json',protocol)
    excluded=np.zeros(len(mol),bool);reasons=Counter();maximum=np.full(len(mol),np.nan,np.float32)
    for i,r in enumerate(mol.itertuples(index=False)):
        reason=None
        if r.key in keys:reason='identity'
        elif alias[i]:reason='known_acquisition_alias'
        elif r.formula in forms:reason='formula'
        else:
            m=Chem.MolFromSmiles(r.smiles);assert m is not None
            scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
            if scaffold and scaffold in scaffolds:reason='scaffold'
            else:
                similarity=max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),query_fp));maximum[i]=similarity
                if similarity>=.7:reason='fingerprint_similarity'
        if reason:excluded[i]=True;reasons[reason]+=1
        if (i+1)%5000==0:
            assert system_memory()['free_gib']>=4
            print('C40 CHEMISTRY EXCLUSIONS',i+1,'of',len(mol),'removed',int(excluded[:i+1].sum()),flush=True)
    prior=np.load(CACHE/'split.npz',allow_pickle=False)
    eligible=np.asarray(prior['train'],np.int64)
    eligible=eligible[~excluded[eligible]]
    assert len(eligible)>50000,'Insufficient clean fitting identities'
    selected=pd.read_parquet(CACHE/'selected.parquet',columns=['mol_idx','valid'])
    row_indices=np.flatnonzero(selected.valid.to_numpy()&selected.mol_idx.isin(eligible).to_numpy())
    assert set(selected.mol_idx.iloc[row_indices]).issubset(set(eligible))
    # Preserve one weight per identity; do not upweight compounds with many views.
    counts=np.bincount(selected.mol_idx.iloc[row_indices],minlength=len(mol))
    assert np.all(counts[eligible]>0)
    np.save(OUT/'excluded-molecules.npy',excluded)
    np.save(OUT/'maximum-heldout-similarity.npy',maximum)
    np.save(OUT/'fitting-identities.npy',eligible)
    np.save(OUT/'fitting-spectrum-rows.npy',row_indices)
    assert not set(mol.key.iloc[eligible])&keys
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'fit-boundary-protocol.json'),
        heldout_queries=len(holdout),heldout_chemistry_groups=int(holdout.group.nunique()),total_catalogue=len(mol),
        excluded_catalogue_identities=int(excluded.sum()),first_exclusion_reason_counts=dict(reasons),
        original_fit_identities=len(prior['train']),new_fit_identities=len(eligible),new_fit_spectrum_rows=len(row_indices),
        maximum_remaining_heldout_tanimoto=float(np.nanmax(maximum[eligible])),identity_overlap=0,
        files_sha256={n:sha(OUT/n) for n in ['excluded-molecules.npy','maximum-heldout-similarity.npy','fitting-identities.npy','fitting-spectrum-rows.npy']},
        source_queries_not_used_for_fit_or_hyperparameters=True,old_trained_head_weights_not_reused=True,
        frozen_encoder_pretraining_exposure_not_claimed_absent=True,
        quality_metrics_computed=False,reserved_C13_read=False,reference_mask_not_yet_built=True,
        raw_near_spectrum_audit_remaining=True,qualified_for_scoring=False)
    write(OUT/'fit-boundary-freeze.json',report);print('DONE C40 FIT BOUNDARIES',json.dumps(report),flush=True)


if __name__=='__main__':main()
