"""Add all auxiliary chemistry/aliases to immutable C40R1 fitting exclusions."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,module,descriptor
CONFIG=ROOT/'configs/c48-clean-residual-pair-20261112.json'
OUT=ROOT/'outputs/local/c48-clean-residual-heads-20261112'
PANEL=ROOT/'outputs/local/c48-auxiliary-holdout-20261112'
OLD=ROOT/'outputs/local/c40r1-crossmodal-heads-20261111'
REFERENCE=ROOT/'outputs/local/c41r1-masked-pipeline-validation-20261111'
CACHE=ROOT/'outputs/local/long-frozen-20261107-r1'


def alias_closure(witnesses,blocked_official,blocked_raw):
    blocked_official=set(blocked_official);blocked_raw=set(blocked_raw)
    valid=witnesses.official_key.notna();count=0
    while True:
        size=(len(blocked_official),len(blocked_raw))
        bad=witnesses.raw_key.isin(blocked_raw)|witnesses.official_key.isin(blocked_official)|~valid
        blocked_raw.update(witnesses.loc[bad,'raw_key'])
        blocked_official.update(witnesses.loc[witnesses.raw_key.isin(blocked_raw)&valid,'official_key'])
        count+=1
        if size==(len(blocked_official),len(blocked_raw)):return blocked_official,blocked_raw,count


def main():
    RDLogger.DisableLog('rdApp.*');OUT.mkdir(exist_ok=True);assert not (OUT/'fit-boundary-freeze.json').exists()
    assert read(ROOT/'.local/c38-active-goal.json')['active']
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    original=read(OLD/'fit-boundary-freeze.json');panel=read(PANEL/'validation-freeze.json');aliases=read(PANEL/'raw-source-alias-audit.json')
    for name,pin in original['files_sha256'].items():assert sha(OLD/name)==pin
    for name,pin in panel['files_sha256'].items():assert sha(ROOT/name)==pin
    for name,pin in aliases['files_sha256'].items():assert sha(PANEL/name)==pin
    assert aliases['queries_removed']==0 and aliases['queries']==35
    ref=read(REFERENCE/'reference-exclusion-freeze.json')
    for name,pin in ref['files_sha256'].items():assert sha(REFERENCE/name)==pin
    source_pin=sha(__file__);config_pin=sha(CONFIG)
    write(OUT/'fit-boundary-protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_pin,config_sha256=config_pin,
        original_boundary_sha256=sha(OLD/'fit-boundary-freeze.json'),panel_freeze_sha256=sha(PANEL/'validation-freeze.json'),
        alias_audit_sha256=sha(PANEL/'raw-source-alias-audit.json'),witness_sha256=sha(REFERENCE/'original-structure-identity-witnesses.parquet'),
        rule='Preserve ALL original exclusions. Add complete auxiliary chemistry; propagate raw-to-official identity quarantine to closure. Any excluded original raw acquisition removes its whole fitting identity. Never filter validation by observed predictions.',
        C39_outcomes_known=True,new_source_quality_read=False,C13_labels_read=False))
    molecules=pd.read_parquet(CACHE/'molecules.parquet');prepared=read(CACHE/'prepared.json')
    assert len(molecules)==273833 and sha(CACHE/'molecules.parquet')==prepared['mol_id_order_sha256']
    labels=pd.read_csv(PANEL/'labels.csv',dtype=str);assert len(labels)==35
    features=module('c48_fit_exclusion_features','c09e_features.py');query=[descriptor(s,features) for s in labels.smiles]
    for item,key in zip(query,labels.key):assert item['key']==key
    keys=set(labels.key);forms=set(labels.formula);scaffolds={item['scaffold'] for item in query if item['scaffold']};fps=[item['fp'] for item in query]
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    for item in query:assert DataStructs.TanimotoSimilarity(item['fp'],generator.GetFingerprint(Chem.MolFromSmiles(item['canonical'])))==1.
    excluded=np.load(OLD/'excluded-molecules.npy',allow_pickle=False).copy();maximum=np.full(len(molecules),np.nan,np.float32)
    old_excluded=excluded.copy();reasons={};new_blocked=set(keys)
    def chemistry(smiles,formula):
        if formula in forms:return 'formula',None
        mol=Chem.MolFromSmiles(smiles);assert mol is not None
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
        if scaffold and scaffold in scaffolds:return 'nonempty_scaffold',None
        similarity=max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol),fps))
        return ('similarity' if similarity>=.7 else None),similarity
    for i,row in enumerate(molecules.itertuples(index=False)):
        if excluded[i]:continue
        if row.key in keys:reason='identity';similarity=None
        else:reason,similarity=chemistry(row.smiles,row.formula)
        if similarity is not None:maximum[i]=similarity
        if reason:excluded[i]=True;reasons[reason]=reasons.get(reason,0)+1
        if (i+1)%10000==0:
            write(OUT/'boundary-status.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage='catalogue_chemistry',completed=i+1,total=len(molecules),excluded=int(excluded.sum())))
    new_blocked.update(molecules.key[excluded]);witnesses=pd.read_parquet(REFERENCE/'original-structure-identity-witnesses.parquet')
    new_blocked.update(witnesses.loc[witnesses.exclude,'official_key'].dropna())
    # Only362official identities lack a canonical catalogue witness. Parse one
    # original structure for each, using the exact official standardization.
    extras=witnesses[witnesses.official_key.notna()&~witnesses.official_key.isin(set(molecules.key))].drop_duplicates('official_key')
    for row in extras.itertuples(index=False):
        if row.official_key in new_blocked:continue
        item=descriptor(row.smiles,features);assert item['key']==row.official_key
        reason,_=chemistry(item['canonical'],item['formula'])
        if reason:new_blocked.add(row.official_key)
    raw_blocked=set(read(REFERENCE/'engine2-raw-key-exclusion.json')['keys'])|set(read(PANEL/'additional-raw-alias-quarantine.json')['keys'])
    blocked_official,blocked_raw,rounds=alias_closure(witnesses,new_blocked,raw_blocked)
    excluded|=molecules.key.isin(blocked_official).to_numpy()
    old_ids=np.load(OLD/'fitting-identities.npy',allow_pickle=False);old_rows=np.load(OLD/'fitting-spectrum-rows.npy',allow_pickle=False)
    selected=pd.read_parquet(CACHE/'selected.parquet',columns=['row','mol_idx','valid'])
    raw=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14'])
    fit_raw=raw.inchikey14.astype(str).to_numpy()[selected.row.to_numpy()[old_rows]]
    bad_rows=np.isin(fit_raw,list(blocked_raw));bad_ids=np.unique(selected.mol_idx.to_numpy()[old_rows[bad_rows]])
    excluded[bad_ids]=True
    ids=old_ids[~excluded[old_ids]];rows=old_rows[selected.mol_idx.iloc[old_rows].isin(set(ids)).to_numpy()]
    assert len(ids)>50000 and set(ids)<=set(old_ids) and set(rows)<=set(old_rows)
    assert not set(molecules.key.iloc[ids])&set(labels.key)
    assert not set(raw.inchikey14.iloc[selected.row.to_numpy()[rows]].astype(str))&blocked_raw
    assert np.all(excluded[old_excluded]) and not excluded[ids].any()
    assert float(np.nanmax(maximum[ids]))<.7
    np.save(OUT/'excluded-molecules.npy',excluded);np.save(OUT/'maximum-auxiliary-similarity.npy',maximum)
    np.save(OUT/'fitting-identities.npy',ids);np.save(OUT/'fitting-spectrum-rows.npy',rows)
    write(OUT/'engine2-raw-key-exclusion.json',dict(keys=sorted(blocked_raw),scope='C39+auxiliary global identity/chemistry/acquisition quarantine; no per-query truth mapping.'))
    final_witness=witnesses.copy();final_witness['exclude']=final_witness.raw_key.isin(blocked_raw)|final_witness.official_key.isna()|final_witness.official_key.isin(blocked_official)
    final_witness.to_parquet(OUT/'original-structure-identity-witnesses.parquet',index=False)
    assert sha(__file__)==source_pin and sha(CONFIG)==config_pin
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_pin,config_sha256=config_pin,
        protocol_sha256=sha(OUT/'fit-boundary-protocol.json'),original_boundary_sha256=sha(OLD/'fit-boundary-freeze.json'),
        total_catalogue=len(molecules),original_fitting_identities=len(old_ids),new_fit_identities=len(ids),new_fit_spectrum_rows=len(rows),
        original_excluded=int(old_excluded.sum()),total_excluded=int(excluded.sum()),new_chemistry_first_reasons=reasons,
        heldout_C39_queries=352,auxiliary_queries=35,all_original_exclusions_preserved=True,new_query_identity_overlap=0,
        original_raw_alias_survivors=0,maximum_remaining_auxiliary_tanimoto=float(np.nanmax(maximum[ids])),
        excluded_raw_keys=len(blocked_raw),excluded_official_keys=len(blocked_official),alias_closure_iterations=rounds,
        files_sha256={name:sha(OUT/name) for name in ['excluded-molecules.npy','maximum-auxiliary-similarity.npy','fitting-identities.npy','fitting-spectrum-rows.npy','engine2-raw-key-exclusion.json','original-structure-identity-witnesses.parquet']},
        C39_quality_already_known=True,new_source_quality_read=False,C13_labels_read=False,qualified_for_scoring=False,
        limitation='Boundary for new fitted components only; public encoder/parent historical exposure is not erased or claimed absent.')
    write(OUT/'fit-boundary-freeze.json',result);print('C48 CLEAN FIT BOUNDARY',json.dumps(result))


if __name__=='__main__':main()
