"""Freeze all outcome-blind C47 survivors as an auxiliary, explicitly limited panel.

The original352-query complete-acquisition/mixed C39 checks remain mandatory.
These single-record ESI/APCI standards are not a replacement target population.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import hashlib,json,re,zipfile
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from c45_complete_parent import ROOT,sha,read,write,module,descriptor
from legacy_formate_compat import normalize_legacy_adduct
from build_external_cache import collision_energy
OUT=ROOT/'outputs/local/c48-auxiliary-holdout-20261112'
PUBLISH=ROOT/'outputs/publish/c48-auxiliary-holdout-20261112'
C47=ROOT/'outputs/local/c47-remaining-tims-inventory-20261112'


def main():
    RDLogger.DisableLog('rdApp.*');assert not OUT.exists() and not PUBLISH.exists();OUT.mkdir();PUBLISH.mkdir()
    c39=ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv'
    mb_inventory=ROOT/'outputs/local/c38-source-support-20261111/massbank-latest-tims-inventory.parquet'
    remaining=C47/'remaining-chemistry.parquet';apci=C47/'source-research/nist-public-plasticizers/inventory-survivor-acquisitions.parquet'
    archive=ROOT/'outputs/research/c10-massbank-current-primary-20261110/MassBank-data-2026.11-pinned.zip'
    c13=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet'
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert sha(archive)=='84a8edf35f8df0341151da1dec69359f4a4939038d05816628da974169d1aa84'
    cfg=read(ROOT/'configs/c39-source-supported-validation-20261111.json');allowed=set(cfg['allowed_query_adducts'])
    assert sha(remaining)==read(C47/'inventory.json')['remaining_chemistry_sha256']
    assert sha(apci)==read(C47/'source-research/nist-public-plasticizers/inventory.json')['files_sha256'][apci.name]
    inputs=[c39,mb_inventory,remaining,apci,c13,archive,C47/'inventory.json',C47/'source-research/nist-public-plasticizers/inventory.json']
    write(OUT/'protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        rule='Use ALL C47 inventory survivors; no sampling, scoring or performance-based exclusions. Recheck exact official chemistry separation, and retain every usable original exported acquisition for each selected identity/source.',
        original_C39_required='All352queries/1292acquisitions and original source/query-mode gates remain mandatory used-development checks for a future candidate. Do not relabel C39 as fresh.',
        new_panel_role='Auxiliary chemical holdout only. Twelve remaining primary MassBank records plus thirty public experimental APCI consensus records, subject only to source-integrity and original chemistry rules.',
        identity_exposure='Current heads overlap some identities. NO current-head quality evaluation is allowed here; a new fitting boundary must exclude all new holdout chemistry/aliases and preserve original C39 exclusions.',
        reference_exposure='A prospective fresh-panel inference must exclude its complete query chemistry and input aliases from all reference stages, including Engine representatives, Engine2 and P16. The original audited C39 parent/masks stay unchanged for its development comparison.',
        instrument_limitations='Same MassBank archive; separate positive APCI consensus assay; no new mixed query support or all-raw collision-energy coverage. These do not establish ESI target-population calibration.',
        source_quality_read=False,C13_labels_read=False,scoring_entries_used=0))
    features=module('c48_panel_identity','c09e_features.py');old=pd.read_csv(c39,dtype=str)
    olddesc=[descriptor(s,features) for s in old.smiles];oldfps=[v['fp'] for v in olddesc];oldforms=set(old.formula)
    oldscaffolds={v['scaffold'] for v in olddesc if v['scaffold']};reserved=set(pd.read_parquet(c13,columns=['key']).key)
    queries=[];labels=[];provenance=[];described=[];exclusions=[];record_exclusions=[]
    def add(source,key,smiles,records):
        item=descriptor(smiles,features);assert item['key']==key
        maximum=max(DataStructs.BulkTanimotoSimilarity(item['fp'],oldfps));reason=None
        if key in set(old.key)|reserved:reason='used_or_reserved_identity'
        elif item['formula'] in oldforms:reason='used_development_formula'
        elif item['scaffold'] and item['scaffold'] in oldscaffolds:reason='used_development_nonempty_scaffold'
        elif maximum>=.7:reason='used_development_similarity'
        if reason:exclusions.append(dict(source=source,key=key,reason=reason,acquisitions=len(records)));return
        if not records:
            exclusions.append(dict(source=source,key=key,reason='no_exported_acquisition_meets_existing_adduct_peak_rules',acquisitions=0));return
        mid='C48Q_'+hashlib.sha256((source+':'+key).encode()).hexdigest()[:20]
        for record in records:
            record=dict(record);identifier=record.pop('source_record');sid='C48S_'+hashlib.sha256((source+':'+str(identifier)).encode()).hexdigest()[:20]
            record.update(molecule_id=mid,spectrum_id=sid);queries.append(record)
            provenance.append(dict(molecule_id=mid,spectrum_id=sid,source=source,source_record=str(identifier)))
        _,canonical,formula,mass,_=features.structure(smiles)
        labels.append(dict(molecule_id=mid,key=key,smiles=canonical,formula=formula,mass=mass,source=source,partition='auxiliary_source',maximum_used_C39_tanimoto=maximum))
        described.append(item)
    inventory=pd.read_parquet(mb_inventory);chosen=pd.read_parquet(remaining);assert len(chosen)==12
    with zipfile.ZipFile(archive) as zipped:
        for row in chosen.itertuples(index=False):
            group=inventory[inventory.metric_key.eq(row.key)];assert len(group)==row.available_primary_records
            records=[]
            for entry in group.itertuples(index=False):
                raw=zipped.read(entry.archive_member);assert hashlib.sha256(raw).hexdigest()==entry.original_record_sha256
                lines=raw.decode('utf8').splitlines()
                def field(prefix):
                    values=[line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
                    return values[0] if len(values)==1 else None
                adduct=normalize_legacy_adduct(field('MS$FOCUSED_ION: PRECURSOR_TYPE '));precursor=field('MS$FOCUSED_ION: PRECURSOR_M/Z ')
                if adduct not in allowed or precursor is None:
                    record_exclusions.append(dict(source='MassBank2026.11-remaining-ESI',source_record=entry.accession,key=row.key,adduct=adduct,reason='unsupported_existing_target_adduct' if adduct not in allowed else 'missing_measured_precursor'));continue
                peaks=[];active=False
                for line in lines:
                    if line.startswith('PK$PEAK:'):active=True;continue
                    if active and line.startswith('//'):break
                    if active and line.strip():
                        values=line.split()
                        try:peaks.append((float(values[0]),float(values[1])))
                        except (ValueError,IndexError):pass
                values=np.asarray(peaks,np.float64);assert values.ndim==2
                values=values[np.isfinite(values).all(1)&(values[:,0]>0)&(values[:,1]>0)]
                if len(values)<cfg['minimum_peaks']:
                    record_exclusions.append(dict(source='MassBank2026.11-remaining-ESI',source_record=entry.accession,key=row.key,adduct=adduct,reason='too_few_usable_peaks'));continue
                values=values[np.argsort(values[:,0],kind='stable')]
                records.append(dict(source_record=entry.accession,adduct=adduct,precursor_mz=float(precursor),
                    ionization_mode='positive' if adduct.endswith('+') else 'negative',instrument_type='timsTOF',ionization_method='ESI',
                    collision_energy_ev=collision_energy(field('AC$MASS_SPECTROMETRY: COLLISION_ENERGY ')),
                    ms2_mzs=values[:,0],ms2_normalized_intensities=values[:,1]/values[:,1].max(),num_peaks=len(values)))
            add('MassBank2026.11-remaining-ESI',row.key,row.smiles,records)
    primary=pd.read_parquet(apci);assert len(primary)==30
    for key,group in primary.groupby('key',sort=True):
        records=[]
        for row in group.itertuples(index=False):
            records.append(dict(source_record='public-MGF-record:'+str(row.original_record),adduct=row.adduct,precursor_mz=row.precursor_mz,
                ionization_mode=row.ionization_mode,instrument_type=row.instrument_type,ionization_method='APCI',collision_energy_ev=collision_energy(row.collision_energy_source),
                ms2_mzs=row.ms2_mzs,ms2_normalized_intensities=row.ms2_normalized_intensities,num_peaks=len(row.ms2_mzs)))
        add('NIST-public-APCI-timsTOF-standards',key,group.smiles.iloc[0],records)
    frame=pd.DataFrame(queries);lab=pd.DataFrame(labels);assert frame.spectrum_id.is_unique and lab.molecule_id.is_unique
    assert not {'smiles','key','inchikey14','formula','mass'}&set(frame.columns)
    roots=list(range(len(lab)))
    def find(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    for i,a in enumerate(described):
        for j,b in enumerate(described[:i]):
            if a['key']==b['key'] or a['formula']==b['formula'] or bool(a['scaffold'] and a['scaffold']==b['scaffold']) or DataStructs.TanimotoSimilarity(a['fp'],b['fp'])>=.7:roots[find(i)]=find(j)
    groupmap={r:min(lab.key.iloc[i] for i in range(len(lab)) if find(i)==r) for r in {find(i) for i in range(len(lab))}}
    lab['group']=[groupmap[find(i)] for i in range(len(lab))]
    lab.to_csv(OUT/'labels.csv',index=False);frame.to_parquet(PUBLISH/'validation-queries.parquet',index=False)
    pd.DataFrame(provenance).to_csv(OUT/'provenance.csv',index=False)
    pd.DataFrame(dict(molecule_id=lab.molecule_id,smiles='CCO')).to_csv(PUBLISH/'validation-sample.csv',index=False)
    pd.DataFrame(exclusions).to_json(OUT/'source-integrity-exclusions.json',orient='records',indent=2)
    write(OUT/'record-exclusions.json',record_exclusions)
    modes=frame.groupby('molecule_id').ionization_mode.agg(lambda x:'mixed' if x.nunique()>1 else x.iloc[0])
    result=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),
        queries=len(lab),acquisitions=len(frame),official_identities=lab.key.nunique(),chemistry_groups=lab.group.nunique(),
        sources=lab.source.value_counts().to_dict(),actual_query_modes=modes.value_counts().to_dict(),
        original_C39_all352_complete_checks_still_required=True,no_predictions_scored=True,C13_labels_read=False,
        all_usable_exported_selected_source_acquisitions_retained=True,whole_original_raw_acquisitions_claimed=False,
        canonical_integrity_exclusions=exclusions,source_record_exclusions=record_exclusions,maximum_C39_tanimoto=float(lab.maximum_used_C39_tanimoto.max()),
        fit_and_reference_exclusions_complete=False,qualified_for_scoring=False,
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'labels.csv',OUT/'provenance.csv',PUBLISH/'validation-queries.parquet',PUBLISH/'validation-sample.csv',OUT/'source-integrity-exclusions.json',OUT/'record-exclusions.json']})
    write(OUT/'validation-freeze.json',result);print('C48 AUXILIARY PANEL FROZEN',json.dumps(result))


if __name__=='__main__':main()
