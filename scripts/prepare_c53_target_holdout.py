"""Freeze new controlled target-domain acquisitions before the next model fit."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import hashlib
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from legacy_formate_compat import normalize_legacy_adduct
from c45_complete_parent import ROOT,read,write,sha,stamp
INV=ROOT/'outputs/local/c52-unused-target-inventory-20261112'
OUT=ROOT/'outputs/local/c53-target-holdout-20261112'
PUBLISH=ROOT/'outputs/publish/c53-target-holdout-20261112'
TARGET_COUNTS=dict(positive=265,mixed=93,negative=26)


def draw(key):return hashlib.sha256(('C53-target-holdout-v1:'+key).encode()).hexdigest()


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');OUT.mkdir(exist_ok=True);PUBLISH.mkdir(exist_ok=True)
    assert not (OUT/'validation-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    census=read(INV/'inventory.json');assert census['source_sha256']==sha(ROOT/'scripts/inventory_c52_unused_target.py')
    assert census['inventory_sha256']==sha(INV/'candidate-chemistry-inventory.parquet')
    source_pin=sha(__file__)
    write(OUT/'preparation-protocol.json',dict(frozen_utc=stamp(),source_sha256=source_pin,inventory_sha256=sha(INV/'inventory.json'),
        design='Entire remaining eligible98NPidentities plus384deterministically drawn newEnveda180identities, source-isolated by identity/formula/nonempty scaffold/Morgan2Tanimoto>=.70. No candidate-quality or error-rate sampling.',
        Enveda180_actual_query_mode_counts=TARGET_COUNTS,target400mode_counts=dict(positive=276,mixed=97,negative=27),
        usable_acquisitions='All source acquisitions with original allowed7adducts, positive finite precursor and at least3finite positive peaks. Canonical legacy formate adapter only. Preserve full untrimmed peaks, original collision energies and actual mixed modes.',
        plant='19remaining plant-standard identities are inventory only; missing adduct/mode/CE/provenance is unresolved and no plant prediction is made.',
        role='New target-domain controlled confirmation and NP source stress. Public parent/encoder historic exposure persists; this is not an independent-laboratory novel-identity experiment.',
        no_prediction_outcomes_read=True,C13_labels_or_predictions_read=False,new_model_fits_must_exclude_entire_panel=True))
    pool=pd.read_parquet(INV/'candidate-chemistry-inventory.parquet');pool=pool[~pool.excluded].copy()
    np_rows=pool[pool.source.eq('enveda-np-examples')].drop_duplicates('key');assert len(np_rows)==98
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    def desc(row):
        mol=Chem.MolFromSmiles(row.smiles);assert mol is not None
        return dict(key=row.key,formula=row.formula,scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False),fp=gen.GetFingerprint(mol))
    np_desc=[desc(row) for row in np_rows.itertuples(index=False)]
    nforms={s['formula'] for s in np_desc};nscaf={s['scaffold'] for s in np_desc if s['scaffold']};nfps=[s['fp'] for s in np_desc]
    selected=np_rows.copy();chosen=Counter();exclusions=Counter();candidates=pool[pool.source.eq('enveda-180')].copy();candidates['draw']=candidates.key.map(draw)
    rows=[];seen=set(np_rows.key)
    for row in candidates.sort_values('draw',kind='stable').itertuples(index=False):
        if chosen[row.mode]>=TARGET_COUNTS[row.mode]:continue
        if row.key in seen:exclusions['duplicate_identity']+=1;continue
        d=desc(row)
        if d['formula'] in nforms or bool(d['scaffold'] and d['scaffold'] in nscaf) or max(DataStructs.BulkTanimotoSimilarity(d['fp'],nfps))>=.7:
            exclusions['NP_source_chemistry_connection']+=1;continue
        rows.append({name:getattr(row,name) for name in pool.columns});seen.add(row.key);chosen[row.mode]+=1
        if dict(chosen)==TARGET_COUNTS:break
    assert dict(chosen)==TARGET_COUNTS
    selected=pd.concat([selected,pd.DataFrame(rows)],ignore_index=True);assert len(selected)==selected.key.nunique()==482
    selected['molecule_id']=selected.apply(lambda r:'C53Q_'+hashlib.sha256((r.source+':'+r.key).encode()).hexdigest()[:20],axis=1)
    selected.to_parquet(OUT/'selected-chemistry.parquet',index=False)
    witnesses=pd.read_parquet(ROOT/'outputs/local/c48-clean-residual-heads-20261112/original-structure-identity-witnesses.parquet')
    valid=witnesses[witnesses.official_key.notna()].drop_duplicates(['raw_key','official_key'])
    assert valid.groupby('raw_key').official_key.nunique().loc[selected.raw_key].eq(1).all()
    # Include every source raw alias of a selected official identity, rather
    # than silently retaining just one spelling or one acquisition subset.
    selected_lookup={(r.source,r.key):r for r in selected.itertuples(index=False)}
    raw_map=valid.drop_duplicates('raw_key').set_index('raw_key').official_key.to_dict()
    needed=set(valid.loc[valid.official_key.isin(set(selected.key)),'raw_key'])
    exact_witness={(r.raw_key,r.smiles):r.official_key for r in witnesses[witnesses.official_key.notna()].itertuples(index=False)}
    allowed=set(read(ROOT/'configs/c39-source-supported-validation-20261111.json')['allowed_query_adducts'])
    queries=[];provenance=[];acq_excluded=[];offset=0
    raw=ROOT/'outputs/competition-raw/train.parquet';original_columns=['ingest_lib','normalized_smiles','inchikey14','adduct','ionization_mode','instrument_type','num_peaks','precursor_mz','collision_energy_ev','ms2_mzs','ms2_normalized_intensities']
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=16384,columns=original_columns):
        meta=batch.select(['ingest_lib','inchikey14']).to_pandas();keep=meta.ingest_lib.isin(['enveda-180','enveda-np-examples'])&meta.inchikey14.isin(needed)
        if keep.any():
            indices=np.flatnonzero(keep);part=batch.take(pa.array(indices)).to_pandas()
            for ix,row in zip(indices,part.itertuples(index=False)):
                official=raw_map.get(row.inchikey14);s=selected_lookup.get((row.ingest_lib,official))
                if s is None:continue
                assert exact_witness[(row.inchikey14,row.normalized_smiles)]==s.key
                global_row=offset+int(ix);adduct=normalize_legacy_adduct(row.adduct);reason=None
                if adduct not in allowed:reason='unsupported_existing_target_adduct'
                elif not np.isfinite(row.precursor_mz) or row.precursor_mz<=0:reason='invalid_precursor'
                mz=np.asarray(row.ms2_mzs,float);it=np.asarray(row.ms2_normalized_intensities,float)
                assert mz.ndim==it.ndim==1 and len(mz)==len(it)
                good=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0);mz,it=mz[good],it[good]
                if len(mz)<3:reason=reason or 'fewer_than3finite_positive_peaks'
                if reason:acq_excluded.append(dict(raw_row=global_row,key=s.key,reason=reason));continue
                order=np.argsort(mz,kind='stable');mz,it=mz[order],it[order];it=it/it.max()
                spectrum_id='C53S_'+str(global_row);queries.append(dict(molecule_id=s.molecule_id,spectrum_id=spectrum_id,adduct=adduct,
                    precursor_mz=float(row.precursor_mz),ionization_mode=str(row.ionization_mode),instrument_type=str(row.instrument_type),
                    collision_energy_ev=np.asarray(row.collision_energy_ev,float) if row.collision_energy_ev is not None else np.asarray([],float),
                    ms2_mzs=mz,ms2_normalized_intensities=it,num_peaks=len(mz)))
                provenance.append(dict(molecule_id=s.molecule_id,spectrum_id=spectrum_id,source=row.ingest_lib,raw_row=global_row,raw_key=row.inchikey14,
                    original_adduct=row.adduct,original_num_peaks=row.num_peaks,retained_finite_positive_peaks=len(mz)))
        offset+=len(batch)
    assert offset==2539608
    q=pd.DataFrame(queries);assert q.spectrum_id.is_unique and set(q.molecule_id)==set(selected.molecule_id)
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda x:'mixed' if x.nunique()>1 else x.iloc[0])
    labels=selected[['molecule_id','key','smiles','formula','source']].copy();labels['query_mode']=labels.molecule_id.map(modes)
    assert labels.loc[labels.source.eq('enveda-180'),'query_mode'].value_counts().to_dict()==TARGET_COUNTS
    assert labels.loc[labels.source.eq('enveda-np-examples'),'query_mode'].value_counts().to_dict()=={'mixed':48,'positive':42,'negative':8}
    descriptors=[desc(r) for r in selected.itertuples(index=False)];parents=list(range(len(labels)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(descriptors):
        similarities=DataStructs.BulkTanimotoSimilarity(a['fp'],[b['fp'] for b in descriptors[:i]])
        for j,b in enumerate(descriptors[:i]):
            if a['formula']==b['formula'] or bool(a['scaffold'] and a['scaffold']==b['scaffold']) or similarities[j]>=.7:parents[find(i)]=find(j)
    groups=[find(i) for i in range(len(labels))];group_keys={r:min(labels.key.iloc[i] for i,g in enumerate(groups) if r==g) for r in set(groups)}
    labels['group']=[group_keys[g] for g in groups]
    assert not set(labels.loc[labels.source.eq('enveda-180'),'group'])&set(labels.loc[labels.source.eq('enveda-np-examples'),'group'])
    labels.to_csv(OUT/'labels.csv',index=False);pd.DataFrame(provenance).to_csv(OUT/'provenance.csv',index=False)
    write(OUT/'acquisition-exclusions.json',dict(exclusions=acq_excluded,quality_outcomes_used=False))
    q.to_parquet(PUBLISH/'validation-queries.parquet',index=False);pd.DataFrame(dict(molecule_id=labels.molecule_id,smiles='CCO')).to_csv(PUBLISH/'validation-sample.csv',index=False)
    assert sha(__file__)==source_pin
    files=[OUT/'labels.csv',OUT/'provenance.csv',OUT/'selected-chemistry.parquet',OUT/'acquisition-exclusions.json',PUBLISH/'validation-queries.parquet',PUBLISH/'validation-sample.csv']
    write(OUT/'validation-freeze.json',dict(completed_utc=stamp(),source_sha256=source_pin,protocol_sha256=sha(OUT/'preparation-protocol.json'),
        queries=len(labels),acquisitions=len(q),chemistry_groups=labels.group.nunique(),sources=labels.source.value_counts().to_dict(),
        actual_query_modes=labels.query_mode.value_counts().to_dict(),source_chemistry_groups_disjoint=True,source_isolation_exclusions=dict(exclusions),
        all_usable_original_acquisitions_retained=True,original_raw_rows_inspected=offset,excluded_acquisitions=len(acq_excluded),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},no_prediction_outcomes_read=True,C13_labels_read=False,
        model_fit_and_reference_masks_remaining=True,public_parent_and_encoder_exposure_not_erased=True,qualified_for_scoring=False))
    print('C53 TARGET HOLDOUT FROZEN',len(labels),'queries',len(q),'acquisitions',labels.query_mode.value_counts().to_dict(),flush=True)


if __name__=='__main__':main()
