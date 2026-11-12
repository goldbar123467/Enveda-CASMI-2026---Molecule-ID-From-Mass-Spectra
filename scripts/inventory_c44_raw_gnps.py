"""Audit residual raw documented-tims GNPS records, beyond admitted views.

This is a conservative identity/mass inventory, not a new validation panel or
authentic-standard qualification. No prediction or reserved outcome is read.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from rdkit import RDLogger
from precision_cache import ROOT,sha,write,features
from legacy_formate_compat import scalar_namespace,normalize_legacy_adduct


def main():
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    out=ROOT/'outputs/local/c44-raw-source-audit-20261111';out.mkdir(exist_ok=True);assert not (out/'inventory.json').exists()
    paths=dict(raw_gnps=ROOT/'outputs/external/2026-11-06/gnps.parquet',witnesses=ROOT/'outputs/local/c41r1-masked-pipeline-validation-20261111/original-structure-identity-witnesses.parquet',
        catalogue=ROOT/'outputs/local/n0/cache/molecules.parquet',external=ROOT/'outputs/external/cache-20261106/molecules.parquet',
        C13_exclusion_inventory=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet')
    libraries=['GNPS-REFRAME-DRUG-CCS-LIBRARY','GNPS-ION-MOBILITY-LIBRARY-FILTERED','GNPS-ION-MOBILITY-LIBRARY']
    allowed=['[M+H]+','[M+Na]+','[M+K]+','[M+NH4]+','[M-H]-','[M+CH2O2-H]-','[M+Cl]-']
    write(out/'protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),libraries=libraries,
        inputs_sha256={k:sha(p) for k,p in paths.items()},primary_instrument_source='https://ccms-ucsd.github.io/GNPSDocumentation/gnpslibraries/',
        criteria=dict(adducts=allowed,minimum_finite_positive_peaks=3,maximum_precursor=1000,maximum_mass_error_ppm=30,MS_level=2),
        exclusion='Block already known supplied identity keys first; canonicalize remaining primary SMILES and block official training/reference/prior-panel identities. Conservative: no attempt to repair conflicting supplied identity annotations.',
        C13_inventory_keys_used_only_for_global_exclusion=True,C13_labels_or_outcomes_read=False,predictions_read=False,
        qualification='Inventory only; primary identity/purity and raw-spectrum audits would still be required for any surviving record.'))
    w=pd.read_parquet(paths['witnesses'],columns=['raw_key','official_key']);known=set(w.raw_key.dropna())|set(w.official_key.dropna())
    for n in ['catalogue','external']:known.update(pd.read_parquet(paths[n],columns=['key']).key)
    for p in ['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv','outputs/local/c10-casmi2022-20261110/labels.csv',
              'outputs/local/c20-negative-development-20261110/labels.csv','outputs/local/c39-source-supported-validation-20261111/labels.csv']:
        known.update(pd.read_csv(ROOT/p,usecols=['key']).key)
    known.update(pd.read_parquet(paths['C13_exclusion_inventory'],columns=['key']).key)
    raw=pd.read_parquet(paths['raw_gnps'],filters=[('library','in',libraries)])
    admitted=pd.read_parquet(ROOT/'outputs/external/cache-20261106/spectra.parquet',columns=['source','accession'])
    admitted=set(admitted.loc[admitted.source.eq(2),'accession'])
    supplied=raw.inchikey.fillna('').astype(str).str[:14];remaining=raw[~supplied.isin(known)]
    chem=ROOT/'outputs/local/c41-masked-pipeline-validation-20261111/static-compat-fixture/patched_code/casmi/chem.py'
    neutral=scalar_namespace(chem.read_text(encoding='utf8'))['neutral_mass']
    RDLogger.DisableLog('rdApp.*');counts=Counter();structures={};records=[]
    for r in remaining.itertuples(index=False):
        if str(r.ms_level).split('.')[0]!='2':counts['not_MS2']+=1;continue
        adduct=normalize_legacy_adduct(r.adduct)
        if adduct not in allowed:counts['outside_prespecified_target_adducts']+=1;continue
        if not np.isfinite(r.precursor_mz) or not 0<float(r.precursor_mz)<=1000:counts['outside_encoder_precursor_support']+=1;continue
        mz,it=np.asarray(r.mz,float),np.asarray(r.intensity,float)
        if len(mz)!=len(it) or (np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0)).sum()<3:counts['insufficient_valid_fragments']+=1;continue
        s=str(r.smiles)
        if s not in structures:
            try:structures[s]=features.structure(s)[:4]
            except (ValueError,TypeError,RuntimeError):structures[s]=None
        value=structures[s]
        if value is None:counts['invalid_primary_smiles']+=1;continue
        key,canonical,formula,mass=value
        if key in known:counts['official_identity_already_known']+=1;continue
        nm=float(neutral(np.asarray([r.precursor_mz],float),np.asarray([adduct],object))[0]);ppm=abs(nm-float(mass))/float(mass)*1e6
        if not np.isfinite(ppm) or ppm>30:counts['primary_structure_precursor_inconsistent']+=1;continue
        counts['inventory_survivor']+=1
        records.append(dict(accession=r.spectrum_id,library=r.library,key=key,smiles=canonical,formula=formula,mass=mass,adduct=adduct,
            precursor_mz=float(r.precursor_mz),mass_error_ppm=ppm,previously_admitted=r.spectrum_id in admitted,library_class=r.library_class,compound_source=r.compound_source))
    pd.DataFrame(records).to_parquet(out/'inventory-survivors.parquet',index=False)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'protocol.json'),raw_records=len(raw),
        raw_records_not_in_previous_admitted_view=int((~raw.spectrum_id.isin(admitted)).sum()),known_supplied_identity_records=int(supplied.isin(known).sum()),
        residual_records_requiring_primary_structure_check=len(remaining),unique_primary_smiles_canonicalized=len(structures),outcomes=dict(counts),
        surviving_records=len(records),surviving_identities=len({r['key'] for r in records}),qualified_validation_panel=False,
        conservative_supplied_key_exclusion=True,C13_reserved_outcomes_read=False,prediction_outcomes_read=False,source_selection_by_model_performance=False,
        limitation='Previously admitted-view inventory was not an audit of every raw record. This residual check uses the cached November6source snapshot; it does not prove absence of later updates, annotation repairs or other libraries. GNPS Crude/class1 metadata alone does not prove authentic-standard purity.')
    write(out/'inventory.json',result);print('C44 RAW SOURCE AUDIT',json.dumps(result),flush=True)


if __name__=='__main__':main()
