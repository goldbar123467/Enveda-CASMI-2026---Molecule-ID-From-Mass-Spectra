"""Map original reference labels to official identities and C39 exclusions.

The global mask removes heldout chemistry from both original reference engines.
No per-query answer map is mounted and no true structure is added to candidates.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from precision_cache import ROOT,sha,write,features
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c41-masked-pipeline-validation-20261111'
C40=ROOT/'outputs/local/c40-crossmodal-heads-20261111'
C39=ROOT/'outputs/local/c39-source-supported-validation-20261111'


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');assert system_memory()['free_gib']>=4
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'reference-exclusion-freeze.json').exists()
    reference=ROOT/'outputs/research/c08-pipeline/reference/train_structs.parquet';rawpath=ROOT/'outputs/competition-raw/train.parquet'
    holdout=pd.read_csv(C39/'labels.csv');mol=pd.read_parquet(ROOT/'outputs/local/n0/cache/molecules.parquet')
    excluded=np.load(C40/'excluded-molecules.npy',allow_pickle=False);assert len(excluded)==len(mol)
    excluded_keys=set(mol.key[excluded])|set(holdout.key)
    lookup=dict(zip(mol.smiles,mol.key));querykeys=set(holdout.key);queryforms=set(holdout.formula)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    querymol=[Chem.MolFromSmiles(s) for s in holdout.smiles]
    queryfp=[gen.GetFingerprint(m) for m in querymol]
    queryscaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in querymol};queryscaffolds.discard('')
    boundary=json.loads((C40/'fit-boundary-freeze.json').read_text())
    assert boundary['files_sha256']['excluded-molecules.npy']==sha(C40/'excluded-molecules.npy')
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),builder_sha256=sha(__file__),
        reference_sha256=sha(reference),C39_freeze_sha256=sha(C39/'validation-freeze.json'),
        C40_boundaries_sha256=sha(C40/'fit-boundary-freeze.json'),
        policy='Remove heldout chemistry and known alias groups by global reference SID/raw-key masks in both engines. No candidate truth insertion.',
        query_labels_not_mounted=True,quality_metrics_read=False,C13_read=False,
        pretraining_caveat='Reference exclusion does not remove historic public neural-model exposure.')
    write(OUT/'reference-exclusion-protocol.json',protocol)
    raw=pd.read_parquet(rawpath,columns=['inchikey14','normalized_smiles'])
    unique=raw.drop_duplicates(['inchikey14','normalized_smiles']).copy()
    extra_path=OUT/'official-identity-cache.json';extra=json.loads(extra_path.read_text()) if extra_path.exists() else {}
    lookup.update(extra);counts=Counter();records=[]
    for i,r in enumerate(unique.itertuples(index=False)):
        s=str(r.normalized_smiles);key=lookup.get(s);reason=None
        if s not in lookup:
            try:
                key,canonical,formula,mass,_=features.structure(s)
                lookup[s]=key;extra[s]=key
                if key not in excluded_keys:
                    m=Chem.MolFromSmiles(canonical);scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
                    if formula in queryforms or (scaffold and scaffold in queryscaffolds) or max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),queryfp))>=.7:
                        excluded_keys.add(key);reason='additional_original_structure_chemistry'
            except (ValueError,TypeError,RuntimeError):
                lookup[s]=None;extra[s]=None;key=None
            if len(extra)%100==0:write(extra_path,extra)
        mask=key is None or key in excluded_keys
        if key is None:reason='invalid_original_structure_quarantined'
        elif key in excluded_keys and reason is None:reason='heldout_chemistry_or_known_alias'
        counts[reason or 'allowed']+=1
        records.append(dict(raw_key=str(r.inchikey14),smiles=s,official_key=key,exclude=bool(mask)))
        if (i+1)%10000==0:
            assert system_memory()['free_gib']>=4
            print('C41 REFERENCE IDENTITIES',i+1,'of',len(unique),'new identities',len(extra),'excluded',sum(v for k,v in counts.items() if k!='allowed'),flush=True)
    write(extra_path,extra)
    frame=pd.DataFrame(records)
    raw_mask=frame.groupby('raw_key',sort=False).exclude.any()
    assert set(raw.inchikey14.astype(str))<=set(raw_mask.index)
    rawkeys=sorted(raw_mask.index[raw_mask])
    st=pd.read_parquet(reference,columns=['sid','inchikey14','key','rawkey','smiles'])
    assert st.sid.tolist()==list(range(len(st)))
    mask=st.inchikey14.astype(str).isin(rawkeys).to_numpy()
    keymask=np.asarray([lookup.get(str(s)) in excluded_keys or lookup.get(str(s)) is None for s in st.smiles])
    mask|=keymask
    assert not any(lookup.get(str(s)) in querykeys for s in st.loc[~mask,'smiles'])
    np.save(OUT/'main-reference-sid-exclusion.npy',mask)
    write(OUT/'engine2-raw-key-exclusion.json',dict(keys=rawkeys,scope='Global reference exclusion only; never use as candidate-ranking features'))
    frame.to_parquet(OUT/'original-structure-identity-witnesses.parquet',index=False)
    # Audit the new heads against ACTUAL original training-row identities,
    # not just their historically assigned cached target indices.
    selected=pd.read_parquet(ROOT/'outputs/local/long-frozen-20261107-r1/selected.parquet',columns=['row','mol_idx','valid'])
    fitting=np.load(C40/'fitting-identities.npy',allow_pickle=False)
    fitrows=selected[selected.valid & selected.mol_idx.isin(fitting)].copy()
    actual_raw=raw.inchikey14.to_numpy()[fitrows.row.to_numpy()]
    actual_blocked=np.asarray([bool(raw_mask.loc[str(k)]) for k in actual_raw])
    affected_fit_ids=np.unique(fitrows.mol_idx.to_numpy()[actual_blocked])
    np.save(OUT/'additional-fit-identities-to-quarantine.npy',affected_fit_ids)
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'reference-exclusion-protocol.json'),
        source_structure_pairs=len(frame),source_raw_identities=int(frame.raw_key.nunique()),new_canonicalizations=len(extra),
        raw_reference_keys_excluded=len(rawkeys),main_reference_SIDs_excluded=int(mask.sum()),main_reference_SIDs=len(mask),
        reference_query_identity_survivors=0,original_structure_counts=dict(counts),
        new_heads_additional_original_row_exclusions=len(affected_fit_ids),new_heads_original_rows_requiring_quarantine=int(actual_blocked.sum()),
        current_head_fit_boundary_passed=len(affected_fit_ids)==0,
        files_sha256={n:sha(OUT/n) for n in ['main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json','original-structure-identity-witnesses.parquet','additional-fit-identities-to-quarantine.npy']},
        runtime_enforcement_not_yet_built=True,pipeline_quality_metrics_read=False,C13_read=False,qualified_for_submission=False)
    write(OUT/'reference-exclusion-freeze.json',report);print('DONE C41 REFERENCE EXCLUSIONS',json.dumps(report),flush=True)


if __name__=='__main__':main()
