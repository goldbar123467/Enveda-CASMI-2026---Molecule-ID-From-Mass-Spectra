"""Create fresh external queries absent from C08 and the production spectral SIDs.

Only already-approved MassBank/GNPS records enter the panel. Labels/provenance
remain a separate sidecar; query rows contain no structure names or identities.
No candidate is injected, and absent truths must count as zero during evaluation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pathlib import Path
import hashlib,json,sys
from precision_cache import ROOT,sha,write
from build_external_cache import collision_energy,normalize_adduct

sys.path.insert(0,str(ROOT/'outputs/local/kit'))
import features


def main():
    torch.set_num_threads(2);pa.set_cpu_count(2)
    out=ROOT/'outputs/publish/c08-validation-20261108';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'validation-contract.json').exists(),'Completed validation dataset exists; never resample it'
    base=pd.read_parquet(ROOT/'outputs/local/n0/cache/molecules.parquet')
    external=ROOT/'outputs/external/cache-20261106';mol=pd.read_parquet(external/'molecules.parquet');spectra=pd.read_parquet(external/'spectra.parquet')
    reference=ROOT/'outputs/research/c08-pipeline/reference/train_structs.parquet'
    assert sha(reference)=='f20aa1cac23af42e430a6633707fda34e48dea97114bdc3e443897f692df3656'
    st=pd.read_parquet(reference,columns=['key','inchikey14','rawkey'])
    rawkeys=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14']).inchikey14
    forbidden=set(base.key)|set(rawkeys)
    for column in st:forbidden.update(st[column].dropna().astype(str).str[:14])
    eligible=mol.loc[~mol.key.isin(forbidden)].copy();assert len(eligible)>=128
    eligible['draw']=eligible.key.map(lambda key:hashlib.sha256(('C08-external-panel-v1:'+key).encode()).hexdigest())
    # Stratify source coverage without consulting candidate recall or scores.
    sources=spectra.groupby('mol_idx').source.min();eligible['source']=sources.reindex(eligible.index).astype(int)
    selected=pd.concat([g.sort_values('draw').head(64) for _,g in eligible.groupby('source')]).sort_values('draw').head(128)
    assert len(selected)==128 and selected.key.is_unique
    selected['molecule_id']=selected.draw.map(lambda value:'C08V_'+value[:16])
    selected['partition']=['development' if i%2==0 else 'confirmation' for i in range(len(selected))]
    labels=selected[['molecule_id','key','smiles','formula','mass','source','partition']].copy()
    chosen=spectra[spectra.mol_idx.isin(selected.index)].copy()
    chosen['draw']=chosen.accession.map(lambda value:hashlib.sha256(('C08-view:'+str(value)).encode()).hexdigest())
    chosen=chosen.sort_values('draw',kind='stable').drop_duplicates(['mol_idx','adduct_id','source']).groupby('mol_idx',sort=False).head(4)
    # Fill identities with only one adduct using distinct accepted acquisitions.
    allrows=spectra[spectra.mol_idx.isin(selected.index)].copy();allrows['draw']=allrows.accession.map(lambda value:hashlib.sha256(('C08-view:'+str(value)).encode()).hexdigest())
    chosen=pd.concat([chosen,allrows.sort_values('draw',kind='stable')]).drop_duplicates(['source','accession']).groupby('mol_idx',sort=False).head(4)
    lookup={(int(row.source),str(row.accession)):int(row.mol_idx) for row in chosen.itertuples()}
    records=[];provenance=[];seen=set();verified_keys=set()
    for source,name,identifier in [(1,'massbank','accession'),(2,'gnps','spectrum_id')]:
        wanted=[accession for (src,accession) in lookup if src==source]
        for batch in pq.ParquetFile(ROOT/f'outputs/external/2026-11-06/{name}.parquet').iter_batches(batch_size=4096,use_threads=False):
            keep=pc.is_in(batch.column(batch.schema.get_field_index(identifier)),value_set=pa.array(wanted))
            if not pc.any(keep).as_py():continue
            for row in batch.filter(keep).to_pylist():
                accession=str(row[identifier]);identity=lookup[(source,accession)];key=mol.key.iloc[identity]
                assert key not in forbidden
                if key not in verified_keys:
                    assert key==features.structure(row['smiles'])[0];verified_keys.add(key)
                mid=selected.loc[identity,'molecule_id'];adduct=normalize_adduct(row.get('precursor_type',row.get('adduct')))
                mz=np.asarray(row['mz'],np.float64);intensity=np.asarray(row['intensity'],np.float64)
                good=np.isfinite(mz)&np.isfinite(intensity)&(mz>0)&(intensity>0);mz=mz[good];intensity=intensity[good]
                assert len(mz)>=3;order=np.argsort(mz,kind='stable');mz=mz[order];intensity=intensity[order]/intensity.max()
                energy=collision_energy(row.get('collision_energy'));precursor=float(row['precursor_mz'])
                _,n=features.peaks(mz,intensity,precursor,96);assert n>=3
                spectrum='C08S_'+hashlib.sha256((str(source)+':'+accession).encode()).hexdigest()[:20]
                if spectrum in seen:continue
                seen.add(spectrum)
                polarity=str(row.get('ion_mode','positive')).lower();polarity='negative' if polarity in ('negative','neg','n','-','-1') or adduct.endswith('-') else 'positive'
                records.append(dict(molecule_id=mid,spectrum_id=spectrum,adduct=adduct,precursor_mz=precursor,
                    ionization_mode=polarity,instrument_type=str(row.get('instrument_type',row.get('instrument','other'))),
                    collision_energy_ev=energy,ms2_mzs=mz,ms2_normalized_intensities=intensity,num_peaks=len(mz)))
                accepted=chosen[(chosen.source==source)&chosen.accession.astype(str).eq(accession)].iloc[0]
                provenance.append(dict(spectrum_id=spectrum,molecule_id=mid,source=name,accession=accession,
                    library=accepted.library,license=accepted.license))
    queries=pd.DataFrame(records);assert set(queries.molecule_id)==set(labels.molecule_id)
    assert not {'key','inchikey','smiles','normalized_smiles','formula','mass','compound_name'}&set(queries.columns)
    queries.to_parquet(out/'validation-queries.parquet',index=False);labels.to_csv(out/'labels.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'provenance.csv',index=False)
    pd.DataFrame(dict(molecule_id=labels.molecule_id,smiles='CCO')).to_csv(out/'validation-sample.csv',index=False)
    contract=dict(version=1,queries=128,spectra=len(queries),development=64,confirmation=64,
        eligible_novel_external_identities=len(eligible),exclusion='Absent from C08 catalogue, raw competition InChIKey14 and full275810 production spectral SIDs key/rawkey/inchikey14',
        production_train_structs_sha256=sha(reference),external_manifest_sha256=sha(external/'manifest.json'),
        query_labels_visible_to_pipeline=False,truth_injection=False,truth_absent_mrr=0,
        caveat='Public pretrained foundation models and unrelated historical ranker training may have broader exposure; not a claim of universal pretraining exclusion',
        files={name:sha(out/name) for name in ['validation-queries.parquet','labels.csv','provenance.csv','validation-sample.csv']})
    write(out/'validation-contract.json',contract)
    write(out/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c08-validation-20261108',title='CASMI26 C08 external validation 20261108',licenses=[dict(name='CC-BY-SA-4.0')]))
    (out/'README.md').write_text('# Private C08 external validation\n\n128 fixed novel identities;64 development/64 reserved. No candidate truth injection.\n'
        'Queries contain no structure labels; labels/provenance are separate sidecars for evaluation only.\n'
        'MassBank2026.03 and directly contributed GNPS spectra, same accepted-record filters as the audited6November cache.\n'
        'Record-level library/accession/license attribution is retained in provenance.csv.\n'
        'Sources: https://github.com/MassBank/MassBank-data/releases/tag/2026.03 and https://external.gnps2.org/gnpslibrary\n',encoding='utf-8')
    print('DONE C08 EXTERNAL VALIDATION PREPARATION',json.dumps(contract),flush=True)


if __name__=='__main__':main()
