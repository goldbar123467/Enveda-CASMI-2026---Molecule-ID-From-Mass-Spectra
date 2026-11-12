"""Complete frozen P16 on faithful C13 core; no per-query truth labels.

The mask uses the WHOLE231identity inventory and all120input signatures only.
Source/production admission is mandatory before either phase can run.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import argparse,copy,shutil,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,CODE,REFERENCE,read,write,sha,stamp,frozen_contract,module,precursor_reachability,audit_replay
OUT=ROOT/'outputs/local/c60-complete-c13-20261112'
CLOUD=ROOT/'outputs/forks/c59-c13-main-compatibility-v1'
POLICY=ROOT/'configs/c56-bounded-semantic-pair-20261112.json'


def prior():
    from release_c59_c13 import verified
    verified();assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    return read(ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/complete-candidate-freeze.json')


def mask():
    family=prior();OUT.mkdir(exist_ok=True);target=OUT/'C13-masked-reference';assert not target.exists()
    contract=frozen_contract();source=sha(__file__)
    for name,pin in contract['reference_files'].items():assert sha(REFERENCE/name)==pin
    query_path=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet'
    query_pin=read(ROOT/'kaggle/forks/c59-c13-main-compatibility/compatibility-pin.json');assert sha(query_path)==query_pin['query_sha256']
    q=pd.read_parquet(query_path);assert len(q)==q.molecule_id.nunique()==120
    assert q.ionization_mode.value_counts().to_dict()=={'positive':60,'negative':60}
    inventory=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet';keys=set(pd.read_parquet(inventory,columns=['key']).key);assert len(keys)==231
    features=module('c60_reference_identity','c09e_features.py');signature=module('c60_reference_signatures','c09_spectrum_audit.py').signatures
    signatures=set()
    for row in q.itertuples(index=False):signatures.update(signature(row,features) or [])
    meta=pd.read_parquet(REFERENCE/'ext_meta.parquet');mz=np.load(REFERENCE/'ext_mz.npy',mmap_mode='r');intensity=np.load(REFERENCE/'ext_in.npy',mmap_mode='r')
    assert mz.shape==intensity.shape==(len(meta),128)
    reachable=precursor_reachability(q,meta.precursor_mz,read(CODE/'c09e_policy.json')['thresholds']);positions=np.flatnonzero(reachable)
    # Reuse only independently source-pinned deterministic official identity
    # descriptors. Query-conditioned exclusion flags/similarities are ignored.
    cache_dir=ROOT/'outputs/local/c58-complete-target-20261112/target-masked-reference';cache_manifest=read(cache_dir/'mask-manifest.json')
    assert read(cache_dir/'independent-mask-audit.json')['mask_manifest_sha256']==sha(cache_dir/'mask-manifest.json')
    assert cache_manifest['audit_files']['identity-audit.parquet']==sha(cache_dir/'identity-audit.parquet')
    cached=pd.read_parquet(cache_dir/'identity-audit.parquet').set_index('smiles').to_dict('index');identities={}
    for text in dict.fromkeys(meta.loc[reachable,'normalized_smiles'].astype(str)):
        if text in cached:identities[text]=dict(smiles=text,official_key=cached[text]['official_key'],invalid_identity=bool(cached[text]['invalid_identity']))
        else:
            try:key,*_=features.structure(text);identities[text]=dict(smiles=text,official_key=key,invalid_identity=False)
            except (ValueError,TypeError,RuntimeError):identities[text]=dict(smiles=text,official_key=None,invalid_identity=True)
    descriptors=pd.DataFrame(identities.values()).set_index('smiles')
    flags=pd.DataFrame(dict(original_row=np.arange(len(meta)),outside_all_frozen_precursor_windows=~reachable,
        global231_raw_identity=meta.inchikey14.astype(str).isin(keys).to_numpy(),global231_official_identity=np.zeros(len(meta),bool),
        invalid_identity=np.zeros(len(meta),bool),query_input_alias=np.zeros(len(meta),bool)))
    flags.loc[positions,'global231_official_identity']=descriptors.loc[meta.loc[reachable,'normalized_smiles'].astype(str),'official_key'].isin(keys).to_numpy()
    flags.loc[positions,'invalid_identity']=descriptors.loc[meta.loc[reachable,'normalized_smiles'].astype(str),'invalid_identity'].to_numpy(bool)
    for i in positions:flags.loc[i,'query_input_alias']=bool(signatures.intersection(signature(SimpleNamespace(ms2_mzs=mz[i],ms2_normalized_intensities=intensity[i]),features) or []))
    keep=np.flatnonzero(~flags.drop(columns='original_row').any(axis=1).to_numpy());assert len(keep)>0
    target.mkdir();flags.to_parquet(target/'row-exclusion-audit.parquet',index=False);descriptors.reset_index().to_parquet(target/'identity-audit.parquet',index=False)
    meta.iloc[keep].reset_index(drop=True).to_parquet(target/'ext_meta.parquet',index=False)
    for name,values in [('ext_mz.npy',mz),('ext_in.npy',intensity)]:
        result=np.lib.format.open_memmap(target/name,mode='w+',dtype=values.dtype,shape=(len(keep),128));result[:]=values[keep];result.flush();del result
        np.testing.assert_array_equal(np.load(target/name,mmap_mode='r'),values[keep])
    pd.testing.assert_frame_equal(pd.read_parquet(target/'ext_meta.parquet'),meta.iloc[keep].reset_index(drop=True),check_exact=True)
    assert sha(__file__)==source
    write(target/'mask-manifest.json',dict(frozen_utc=stamp(),source_sha256=source,policy_sha256=sha(POLICY),original_deployment_sha256=sha(CODE/'deployment.json'),
        query_sha256=sha(query_path),global_inventory_sha256=sha(inventory),global_identity_count=231,query_count=120,
        original_rows=len(meta),exact_precursor_reachable_rows=len(positions),retained_rows=len(keep),forbidden_rows_retained=0,
        lossless_precursor_pruning=True,all_original_row_and_peak_alignment_exact=True,
        original_reference_files=contract['reference_files'],reference_files={name:sha(target/name) for name in contract['reference_files']},
        audit_files={name:sha(target/name) for name in ['row-exclusion-audit.parquet','identity-audit.parquet']},
        reason_counts={name:int(flags[name].sum()) for name in flags if name!='original_row'},
        identity_descriptor_cache_sha256=sha(cache_dir/'identity-audit.parquet'),mask_scope='Whole231official/raw identities and all120bounded input signatures; no per-query truth mapping or chemistry-cohort filtering.',
        production_reference_changed=False,per_query_C13_labels_read=False,quality_outcomes_read=False,
        limitation='Identity/input-alias confirmation holdout, not a scaffold-OOD or universal spectral-provenance claim. Public parent/encoder pretraining exposure persists.'))
    shutil.copy2(ROOT/'outputs/local/c56-bounded-semantic-heads-20261112/complete-candidate-freeze.json',OUT/'candidate-family-freeze.json')
    print('C60 WHOLE231C13IDENTITY REFERENCE MASK READY',len(keep),'retained',flush=True)


def replay():
    family=prior();collection=read(CLOUD/'collection-audit.json');fidelity=read(CLOUD/'acquisition-fidelity-audit.json')
    assert collection['queries']==120 and collection['actual_main_fragment_sources_exact_production_adapter']
    assert fidelity['all_original_acquisitions_retained'] and fidelity['all_original_metadata_peaks_and_unknown_CE_exact']
    mask_dir=OUT/'C13-masked-reference';manifest=read(mask_dir/'mask-manifest.json');audit=read(mask_dir/'independent-mask-audit.json')
    assert manifest['source_sha256']==sha(__file__) and manifest['policy_sha256']==sha(POLICY)
    assert audit['mask_manifest_sha256']==sha(mask_dir/'mask-manifest.json') and audit['forbidden_rows_retained']==0
    for name,pin in {**manifest['reference_files'],**manifest['audit_files']}.items():assert sha(mask_dir/name)==pin
    contract=frozen_contract();work=OUT/'C13-parent';assert not work.exists();work.mkdir()
    shutil.copy2(CLOUD/'c13_p01.csv',work/'submission.csv');shutil.copy2(CLOUD/'c13_queries.parquet',work/'effective_query_spectra.parquet')
    shutil.copy2(CLOUD/'c13_neural_p01.parquet',work/'neural_candidate_features.parquet')
    for name in ['frozen_candidate_evidence.parquet','outsider_neural_features.parquet']:shutil.copy2(CLOUD/name,work/name)
    for name in list(contract['source_files'])+['c09e_policy.json']:shutil.copy2(CODE/name,work/name)
    contract=copy.deepcopy(contract);contract.update(reference_files=manifest['reference_files'],validation_only=True,
        reference_mask_manifest_sha256=sha(mask_dir/'mask-manifest.json'),original_deployment_sha256=sha(CODE/'deployment.json'))
    write(work/'validation-deployment.json',contract)
    with (work/'frozen-P16-replay.log').open('w',encoding='utf8') as log:
        subprocess.run([sys.executable,str(work/'apply_external_reference.py'),'--work',str(work),'--input-root',str(mask_dir),'--contract',str(work/'validation-deployment.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    audit_replay(work);receipt=read(work/'c09e_external_receipt.json');assert receipt['search']['queries']==receipt['search']['spectra']==120
    assert receipt['search']['query_input_alias_reference_rows']==0
    for name,source in [('P01.csv',CLOUD/'c13_p01.csv'),('P14.csv',CLOUD/'c13_p14.csv'),('P16.csv',work/'submission.csv'),('queries.parquet',CLOUD/'c13_queries.parquet')]:
        assert not (OUT/name).exists();shutil.copy2(source,OUT/name)
    write(OUT/'complete-parent-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(POLICY),
        candidate_family_sha256=sha(OUT/'candidate-family-freeze.json'),mask_manifest_sha256=sha(mask_dir/'mask-manifest.json'),
        parent_audit_sha256=sha(work/'independent-parent-audit.json'),faithful_C59_collection_sha256=sha(CLOUD/'collection-audit.json'),
        faithful_C59_fidelity_sha256=sha(CLOUD/'acquisition-fidelity-audit.json'),files_sha256={n:sha(OUT/n) for n in ['P01.csv','P14.csv','P16.csv','queries.parquet']},
        per_query_C13_labels_read=False,quality_outcomes_read=False))
    print('C60 COMPLETE FAITHFUL C13PARENT FROZEN; QUALITY UNREAD',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['mask','replay']);a=p.parse_args()
    if a.phase=='mask':mask()
    else:replay()
