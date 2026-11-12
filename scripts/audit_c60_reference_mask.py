"""Independent C13whole231identity/signature mask; no per-query truth."""
import numpy as np
import pandas as pd
from types import SimpleNamespace
from c45_complete_parent import ROOT,REFERENCE,read,write,sha,stamp,module
OUT=ROOT/'outputs/local/c60-complete-c13-20261112'


def main():
    from release_c59_c13 import verified
    verified();target=OUT/'C13-masked-reference';result_path=target/'independent-mask-audit.json';assert not result_path.exists()
    manifest=read(target/'mask-manifest.json');assert manifest['source_sha256']==sha(ROOT/'scripts/c60_complete_c13.py')
    assert manifest['policy_sha256']==sha(ROOT/'configs/c56-bounded-semantic-pair-20261112.json')
    for name,pin in manifest['original_reference_files'].items():assert sha(REFERENCE/name)==pin
    for name,pin in {**manifest['reference_files'],**manifest['audit_files']}.items():assert sha(target/name)==pin
    inventory=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet';assert sha(inventory)==manifest['global_inventory_sha256']
    keys=set(pd.read_parquet(inventory,columns=['key']).key);assert len(keys)==231
    original=pd.read_parquet(REFERENCE/'ext_meta.parquet');selected=pd.read_parquet(target/'ext_meta.parquet');rows=pd.read_parquet(target/'row-exclusion-audit.parquet')
    assert len(original)==200724 and np.array_equal(rows.original_row,np.arange(len(original)))
    flags=rows.drop(columns='original_row');keep=np.flatnonzero(~flags.any(axis=1).to_numpy())
    assert len(keep)==len(selected)==manifest['retained_rows']
    pd.testing.assert_frame_equal(selected,original.iloc[keep].reset_index(drop=True),check_exact=True)
    for name in ['ext_mz.npy','ext_in.npy']:
        a,b=np.load(REFERENCE/name,mmap_mode='r'),np.load(target/name,mmap_mode='r');assert a.dtype==b.dtype
        np.testing.assert_array_equal(a[keep],b)
    query_path=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet';assert sha(query_path)==manifest['query_sha256']
    q=pd.read_parquet(query_path);assert len(q)==120
    mass=original.precursor_mz.to_numpy();reachable=np.zeros(len(original),bool)
    for center in q.precursor_mz:
        width=center*10e-6+.002;reachable|=(mass>=center-width)&(mass<=center+width)
    np.testing.assert_array_equal(~reachable,flags.outside_all_frozen_precursor_windows)
    assert int(reachable.sum())==manifest['exact_precursor_reachable_rows']
    identity=pd.read_parquet(target/'identity-audit.parquet').set_index('smiles');index=original.loc[reachable,'normalized_smiles'].astype(str)
    np.testing.assert_array_equal(flags.loc[reachable,'global231_official_identity'],identity.loc[index,'official_key'].isin(keys).to_numpy())
    np.testing.assert_array_equal(flags.loc[reachable,'invalid_identity'],identity.loc[index,'invalid_identity'].to_numpy(bool))
    np.testing.assert_array_equal(flags.global231_raw_identity,original.inchikey14.astype(str).isin(keys))
    features=module('c60_independent_alias_features','c09e_features.py');signature=module('c60_independent_alias_signatures','c09_spectrum_audit.py').signatures
    signatures=set()
    for row in q.itertuples(index=False):signatures.update(signature(row,features) or [])
    mz=np.load(REFERENCE/'ext_mz.npy',mmap_mode='r');intensity=np.load(REFERENCE/'ext_in.npy',mmap_mode='r');aliases=np.zeros(len(original),bool)
    for i in np.flatnonzero(reachable):aliases[i]=bool(signatures.intersection(signature(SimpleNamespace(ms2_mzs=mz[i],ms2_normalized_intensities=intensity[i]),features) or []))
    np.testing.assert_array_equal(aliases,flags.query_input_alias)
    assert not flags.iloc[keep].any().any()
    write(result_path,dict(completed_utc=stamp(),source_sha256=sha(__file__),mask_manifest_sha256=sha(target/'mask-manifest.json'),
        original_rows=len(original),reachable_rows=int(reachable.sum()),retained_rows=len(keep),all_original_row_and_peak_alignment_exact=True,
        direct_precursor_union_exact=True,whole231raw_and_official_identity_exclusions_exact=True,
        independently_recomputed_all_reachable_input_aliases=True,forbidden_rows_retained=0,per_query_C13_labels_read=False,quality_outcomes_read=False,
        limitation='Bounded identity/signature mask; source-pinned canonicalization is not independently reimplemented and universal historical exposure is not established.'))
    print('C60 INDEPENDENT WHOLE231REFERENCE MASK PASS',len(keep),flush=True)


if __name__=='__main__':main()
