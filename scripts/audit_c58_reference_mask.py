"""Verify original-row alignment and exclusions independently, without outcomes."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import numpy as np
import pandas as pd
from rdkit import RDLogger
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c58-complete-target-20261112'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'outputs/local/c45-complete-parent-validation-20261112/frozen-p16-code'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    RDLogger.DisableLog('rdApp.*');target=OUT/'target-masked-reference';assert not (target/'independent-mask-audit.json').exists()
    manifest=read(target/'mask-manifest.json');original=ROOT/'outputs/research/2026-11-09-silver/extlib'
    assert manifest['source_sha256']==sha(ROOT/'scripts/c58_complete_target.py')
    assert manifest['policy_sha256']==sha(ROOT/'configs/c56-bounded-semantic-pair-20261112.json')
    for name,pin in manifest['original_reference_files'].items():assert sha(original/name)==pin
    for name,pin in {**manifest['reference_files'],**manifest['audit_files']}.items():assert sha(target/name)==pin
    meta=pd.read_parquet(original/'ext_meta.parquet');selected=pd.read_parquet(target/'ext_meta.parquet')
    rows=pd.read_parquet(target/'row-exclusion-audit.parquet');identities=pd.read_parquet(target/'identity-audit.parquet').set_index('smiles')
    assert np.array_equal(rows.original_row.to_numpy(),np.arange(len(meta))) and len(meta)==200724
    flags=rows.drop(columns='original_row');keep=np.flatnonzero(~flags.any(axis=1).to_numpy())
    assert len(keep)==len(selected)==manifest['retained_rows'] and not flags.iloc[keep].any().any()
    pd.testing.assert_frame_equal(selected,meta.iloc[keep].reset_index(drop=True),check_exact=True)
    for name in ['ext_mz.npy','ext_in.npy']:
        a,b=np.load(original/name,mmap_mode='r'),np.load(target/name,mmap_mode='r')
        assert a.shape==(len(meta),128) and b.shape==(len(keep),128) and a.dtype==b.dtype
        for start in range(0,len(keep),2048):np.testing.assert_array_equal(a[keep[start:start+2048]],b[start:start+2048])
    query_path=ROOT/'outputs/publish/c53-target-holdout-20261112/validation-queries.parquet';assert sha(query_path)==manifest['query_sha256']
    q=pd.read_parquet(query_path);reachable=np.zeros(len(meta),bool);mass=meta.precursor_mz.to_numpy()
    # Direct comparisons instead of the mask's sort/search implementation.
    for center in q.precursor_mz:
        width=center*10e-6+.002;reachable|=(mass>=center-width)&(mass<=center+width)
    assert reachable.sum()==manifest['exact_precursor_reachable_rows'] and np.array_equal(~reachable,flags.outside_all_frozen_precursor_windows)
    structural=['invalid_identity','query_identity','known_official_alias','same_formula','same_nonempty_scaffold','near_query_tanimoto']
    expected=identities.loc[meta.loc[reachable,'normalized_smiles'].astype(str),structural].to_numpy(bool)
    np.testing.assert_array_equal(flags.loc[reachable,structural].to_numpy(bool),expected)
    assert np.array_equal(identities.maximum_query_tanimoto.ge(.7),identities.near_query_tanimoto)
    labels_path=ROOT/'outputs/local/c53-target-holdout-20261112/labels.csv';assert sha(labels_path)==manifest['exclusion_structure_source_sha256']
    # Structure fields only: never match a prediction to these labels.
    labels=pd.read_csv(labels_path,usecols=['key','formula'])
    assert np.array_equal(identities.official_key.isin(set(labels.key)),identities.query_identity)
    assert np.array_equal(identities.formula.isin(set(labels.formula)),identities.same_formula)
    raw_path=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/engine2-raw-key-exclusion.json'
    assert sha(raw_path)==manifest['original_global_raw_alias_sha256'];raw=set(read(raw_path)['keys']);assert len(raw)==60280
    np.testing.assert_array_equal(meta.inchikey14.astype(str).isin(raw),flags.known_raw_alias)
    witness=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/original-structure-identity-witnesses.parquet'
    assert sha(witness)==manifest['original_structure_witness_sha256'];w=pd.read_parquet(witness,columns=['raw_key','official_key'])
    official=set(w.loc[w.raw_key.isin(raw),'official_key'].dropna())|set(labels.key)
    np.testing.assert_array_equal(identities.official_key.isin(official),identities.known_official_alias)
    features=load('c45_mask_audit_features','c09e_features.py');signature=load('c45_mask_audit_signature','c09_spectrum_audit.py').signatures
    signatures=set()
    for row in q.itertuples(index=False):signatures.update(signature(row,features) or [])
    mz=np.load(original/'ext_mz.npy',mmap_mode='r');intensity=np.load(original/'ext_in.npy',mmap_mode='r')
    alias=np.zeros(len(meta),bool)
    for i in np.flatnonzero(reachable):alias[i]=bool(signatures.intersection(signature(SimpleNamespace(ms2_mzs=mz[i],ms2_normalized_intensities=intensity[i]),features) or []))
    np.testing.assert_array_equal(alias,flags.query_input_alias)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),mask_manifest_sha256=sha(target/'mask-manifest.json'),
        original_rows=len(meta),reachable_rows=int(reachable.sum()),retained_rows=len(keep),
        all_original_row_and_peak_alignment_exact=True,direct_precursor_union_exact=True,
        every_recorded_structure_exclusion_applied=True,global_alias_quarantine_exact=True,
        independently_recomputed_all_reachable_peak_aliases=True,forbidden_rows_retained=0,
        quality_outcomes_read=False,C13_read=False,
        limitation='Identity descriptor generation is source-pinned and unit-tested; this alignment audit does not independently reimplement tautomer chemistry or prove universal provenance exclusion.')
    (target/'independent-mask-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('C58 INDEPENDENT MASK PASS',json.dumps(result))


if __name__=='__main__':main()
