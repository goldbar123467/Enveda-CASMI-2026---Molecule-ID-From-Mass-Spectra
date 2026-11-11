"""Apply exclusions to every raw-key alias of each blocked official identity.

The first pass records identities while discovering additional source-structure
variants. Finalize AFTER discovery so an earlier alias cannot escape a key
excluded later. Uses immutable first-pass witnesses; no re-canonicalization or
prediction outcome is involved.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write


def finalize(frame):
    blocked=set(frame.loc[frame.exclude,'official_key'].dropna())
    final=frame.copy();final['exclude']=final.official_key.isna()|final.official_key.isin(blocked)
    per_raw=final.groupby('raw_key',sort=False).exclude.any()
    return final,blocked,per_raw


def main():
    source=ROOT/'outputs/local/c41-masked-pipeline-validation-20261111'
    out=ROOT/'outputs/local/c41r1-masked-pipeline-validation-20261111';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'reference-exclusion-freeze.json').exists()
    original=json.loads((source/'reference-exclusion-freeze.json').read_text())
    frame_path=source/'original-structure-identity-witnesses.parquet'
    assert sha(frame_path)==original['files_sha256'][frame_path.name]
    frame=pd.read_parquet(frame_path);final,blocked,per_raw=finalize(frame)
    assert not final.loc[final.official_key.isin(blocked),'exclude'].eq(False).any()
    rawkeys=sorted(per_raw.index[per_raw])
    reference=ROOT/'outputs/research/c08-pipeline/reference/train_structs.parquet'
    st=pd.read_parquet(reference,columns=['sid','inchikey14','smiles'])
    mapping=dict(zip(final.smiles,final.official_key))
    mask=st.inchikey14.astype(str).isin(rawkeys).to_numpy()|np.asarray([mapping.get(str(s)) is None or mapping.get(str(s)) in blocked for s in st.smiles])
    np.save(out/'main-reference-sid-exclusion.npy',mask)
    write(out/'engine2-raw-key-exclusion.json',dict(keys=rawkeys,scope='Final global reference mask; all raw aliases of a blocked official identity excluded.'))
    final.to_parquet(out/'original-structure-identity-witnesses.parquet',index=False)
    mol=pd.read_parquet(ROOT/'outputs/local/n0/cache/molecules.parquet',columns=['key'])
    old=ROOT/'outputs/local/c40-crossmodal-heads-20261111'
    oldfit=np.load(old/'fitting-identities.npy',allow_pickle=False)
    selected=pd.read_parquet(ROOT/'outputs/local/long-frozen-20261107-r1/selected.parquet',columns=['row','mol_idx','valid'])
    raw=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14'])
    fitrows=selected[selected.valid & selected.mol_idx.isin(oldfit)]
    badrow=np.asarray([bool(per_raw.loc[str(k)]) for k in raw.inchikey14.to_numpy()[fitrows.row.to_numpy()]])
    bad_ids=set(fitrows.mol_idx.to_numpy()[badrow])
    bad_ids.update(np.flatnonzero(mol.key.isin(blocked).to_numpy()).tolist())
    additional=np.asarray(sorted(set(oldfit)&bad_ids),np.int64)
    np.save(out/'additional-fit-identities-to-quarantine.npy',additional)
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        initial_pass_sha256=sha(source/'reference-exclusion-freeze.json'),initial_witness_sha256=sha(frame_path),
        source_structure_pairs=len(final),source_raw_identities=int(final.raw_key.nunique()),
        raw_reference_keys_excluded=len(rawkeys),main_reference_SIDs_excluded=int(mask.sum()),main_reference_SIDs=len(mask),
        aliases_retroactively_excluded=int((final.exclude&~frame.exclude).sum()),blocked_official_identities=len(blocked),
        new_heads_total_original_fit_identities_to_quarantine=len(additional),reference_query_identity_survivors=0,
        first_pass_discovery_order_no_longer_affects_alias_masks=True,
        files_sha256={n:sha(out/n) for n in ['main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json','original-structure-identity-witnesses.parquet','additional-fit-identities-to-quarantine.npy']},
        quality_metrics_read=False,C13_read=False,runtime_enforcement_not_yet_verified=True,
        caveat='Initial C41 smoke may establish the identical guard code path; full inference must use these finalized global masks and new heads must exclude the finalized quarantine set.')
    write(out/'reference-exclusion-freeze.json',report);print('DONE C41 GLOBAL MASK FINALIZATION',json.dumps(report),flush=True)


if __name__=='__main__':main()
