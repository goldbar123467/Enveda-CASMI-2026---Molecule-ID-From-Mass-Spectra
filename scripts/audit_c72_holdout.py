"""Independently verify every retained/omitted raw acquisition and source split."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp,module
from legacy_formate_compat import normalize_legacy_adduct
OUT=ROOT/'outputs/local/c72-clean-target-holdout-20261112'


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');target=OUT/'independent-acquisition-audit.json';assert not target.exists()
    freeze=read(OUT/'validation-freeze.json');assert freeze['source_sha256']==sha(ROOT/'scripts/prepare_c72_target_holdout.py')
    for path,pin in freeze['files_sha256'].items():assert sha(ROOT/path)==pin
    q=pd.read_parquet(ROOT/'outputs/publish/c72-clean-target-holdout-20261112/validation-queries.parquet').set_index('spectrum_id')
    labels=pd.read_csv(OUT/'labels.csv',dtype=str).set_index('molecule_id');assert len(labels)==512 and len(q)==2215 and labels.key.nunique()==512
    assert not {'smiles','key','inchikey14','formula','normalized_smiles'}&set(q.columns)
    prov=pd.read_csv(OUT/'provenance.csv');assert len(prov)==len(q) and prov.raw_row.is_unique
    actual=set(prov.raw_row);excluded={r['raw_row']:r for r in read(OUT/'acquisition-exclusions.json')['exclusions']}
    w=pd.read_parquet(ROOT/'outputs/local/c48-clean-residual-heads-20261112/original-structure-identity-witnesses.parquet',columns=['raw_key','official_key'])
    w=w[w.official_key.notna()].drop_duplicates();raw_to_official=w.drop_duplicates('raw_key').set_index('raw_key').official_key.to_dict()
    sources={(r.source,r.key):mid for mid,r in labels.iterrows()};wanted=set(w.loc[w.official_key.isin(set(labels.key)),'raw_key'])
    allowed=set(read(ROOT/'configs/c39-source-supported-validation-20261111.json')['allowed_query_adducts'])
    columns=['ingest_lib','inchikey14','adduct','precursor_mz','ionization_mode','instrument_type','collision_energy_ev','ms2_mzs','ms2_normalized_intensities']
    offset=0;retained=set();omitted=set()
    for batch in pq.ParquetFile(ROOT/'outputs/competition-raw/train.parquet').iter_batches(batch_size=16384,columns=columns):
        meta=batch.select(['ingest_lib','inchikey14']).to_pandas();pick=meta.inchikey14.isin(wanted)&meta.ingest_lib.isin({'enveda-180','enveda-np-examples'})
        ix=np.flatnonzero(pick)
        for local,row in zip(ix,batch.take(pa.array(ix)).to_pandas().itertuples(index=False)):
            mid=sources.get((row.ingest_lib,raw_to_official[row.inchikey14]))
            if mid is None:continue
            raw_row=offset+int(local);mz=np.asarray(row.ms2_mzs,float);it=np.asarray(row.ms2_normalized_intensities,float)
            finite=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0);mz,it=mz[finite],it[finite]
            usable=normalize_legacy_adduct(row.adduct) in allowed and np.isfinite(row.precursor_mz) and row.precursor_mz>0 and len(mz)>=3
            if not usable:
                assert raw_row in excluded and raw_row not in actual;omitted.add(raw_row);continue
            assert raw_row in actual and raw_row not in excluded;retained.add(raw_row)
            r=q.loc['C72S_'+str(raw_row)];assert r.molecule_id==mid and r.adduct==normalize_legacy_adduct(row.adduct)
            assert r.precursor_mz==float(row.precursor_mz) and r.ionization_mode==str(row.ionization_mode) and r.instrument_type==str(row.instrument_type)
            ce=np.asarray(row.collision_energy_ev,float) if row.collision_energy_ev is not None else np.asarray([],float)
            np.testing.assert_array_equal(np.asarray(r.collision_energy_ev),ce)
            # Independent direct order reconstruction; no cutoff, rounding or
            # fixed top-k is permitted in the stored query acquisition.
            permutation=sorted(range(len(mz)),key=lambda i:(mz[i],i))
            np.testing.assert_array_equal(np.asarray(r.ms2_mzs),mz[permutation])
            np.testing.assert_array_equal(np.asarray(r.ms2_normalized_intensities),it[permutation]/max(it))
            assert r.num_peaks==len(mz)
        offset+=len(batch)
    assert offset==2539608 and retained==actual and omitted==set(excluded)
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]);assert modes.to_dict()==labels.query_mode.to_dict()
    assert labels.query_mode.value_counts().to_dict()=={'positive':353,'mixed':124,'negative':35}
    assert labels.source.value_counts().to_dict()=={'enveda-180':512}
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);descriptors=[]
    official=module('c72_official_identity_audit','c09e_features.py').structure
    for row in labels.itertuples():
        key,_,formula,*_=official(row.smiles);assert key==row.key and formula==row.formula
        m=Chem.MolFromSmiles(row.smiles);descriptors.append(dict(source=row.source,formula=row.formula,scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False),fp=gen.GetFingerprint(m)))
    # Independently check ALL previously USED and full global reserved chemistry.
    old=pd.read_parquet(ROOT/'outputs/local/c69-clean-parent-evidence-oof-20261112/queries.parquet')[['key','smiles','formula']]
    paths=['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
        'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']
    old=pd.concat([old]+[pd.read_csv(ROOT/p,usecols=lambda c:c in {'key','smiles','formula'}) for p in paths]+[
        pd.read_parquet(ROOT/'outputs/local/c68-reserved-fit-boundary-20261112/global-exclusion-chemistry.parquet')],ignore_index=True).drop_duplicates('key')
    oms=[Chem.MolFromSmiles(s) for s in old.smiles];ofs=set(old.formula);osc={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in oms};osc.discard('')
    ofp=[gen.GetFingerprint(m) for m in oms];maximum=0.
    for a in descriptors:
        similarity=max(DataStructs.BulkTanimotoSimilarity(a['fp'],ofp));maximum=max(maximum,similarity)
        assert a['formula'] not in ofs and not(a['scaffold'] and a['scaffold'] in osc) and similarity<.7
    headkeys=set(pd.read_parquet(ROOT/'outputs/local/c69-clean-parent-evidence-oof-20261112/candidates.parquet',columns=['key']).key)
    assert not set(labels.key)&(set(old.key)|headkeys)
    family=ROOT/'outputs/local/c71-final-clean-direct-pair-20261112/complete-pipeline-binding.json'
    assert freeze['complete_pipeline_binding_sha256']==sha(family)
    assert read(family)['frozen_utc']<freeze['completed_utc']
    aliases=read(OUT/'raw-source-alias-audit.json');assert aliases['queries_removed']==0 and aliases['queries']==512
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),panel_freeze_sha256=sha(OUT/'validation-freeze.json'),
        queries=512,acquisitions=2215,chemistry_groups=int(labels.group.nunique()),actual_query_modes={'positive':353,'mixed':124,'negative':35},
        all_original_raw_rows_checked=offset,all_usable_acquisitions_retained=True,all_original_metadata_and_untrimmed_peak_arrays_exact=True,
        all512official_structure_identities_verified=True,actual_instrument_acquisitions=q.instrument_type.value_counts().to_dict(),
        every_unusable_acquisition_explicitly_accounted=True,unusable_acquisitions=len(omitted),all_USED_and_global231reserved_identity_formula_scaffold_and_near_chemistry_disjoint=True,all_head_input_identity_overlap=0,
        maximum_to_USED_or_global_reserved_Morgan2_Tanimoto=maximum,query_labels_mounted=False,prediction_outcomes_read=False,C13_read=False,
        limitation='Controlled in-corpus holdout. Public parent/encoder historic exposure persists; no independent laboratory or universal spectral novelty claim.'))
    print('C72 INDEPENDENT ALL-ACQUISITION AUDIT PASS',len(retained),'retained',len(omitted),'explicit exclusions',flush=True)


if __name__=='__main__':main()
