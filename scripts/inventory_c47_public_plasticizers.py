"""Audit public NIST experimental APCI-timsTOF standards, not NIST comparators.

This is a census before any new fit/prediction. APCI positive consensus spectra
are explicitly a different assay and cannot stand in for mixed ESI acquisitions.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import re,json
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,sha,read,write,module,descriptor


def parse_mgf(text):
    rows=[]
    for number,block in enumerate(text.split('BEGIN IONS')[1:]):
        fields={};peaks=[]
        if 'END IONS' not in block:fields['_source_missing_END_IONS']=['true']
        for line in block.split('END IONS',1)[0].splitlines():
            line=line.strip()
            if '=' in line and not line.startswith('='):
                name,value=line.lstrip('#').split('=',1);fields.setdefault(name,[]).append(value.strip())
            values=line.split()
            try:
                a,b=float(values[0]),float(values[1])
                if np.isfinite(a) and np.isfinite(b) and a>0 and b>0:peaks.append((a,b))
            except (ValueError,IndexError):pass
        rows.append((number,fields,peaks))
    return rows


def one(fields,name):
    values=set(fields.get(name,[]));return values.pop() if len(values)==1 else None


def main():
    RDLogger.DisableLog('rdApp.*');out=ROOT/'outputs/local/c47-remaining-tims-inventory-20261112/source-research/nist-public-plasticizers'
    assert not (out/'inventory.json').exists()
    manifest=read(out/'download-manifest.json')
    for name,value in manifest['files'].items():assert 'sha256' in value and sha(out/name)==value['sha256']
    c39=ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv'
    c13=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet'
    config=ROOT/'configs/c39-source-supported-validation-20261111.json';cfg=read(config)
    oldlabels=[ROOT/p for p in ['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
        'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']]
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),download_manifest_sha256=sha(out/'download-manifest.json'),
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [c39,c13,config]+oldlabels},
        rule='Census every exported spectrum. Require explicit original Bruker timsTOF AND APCI, a matching primary metadata row, source-supplied SDF structure agreeing with raw InChIKey, existing C39 supported adduct/minimum3peaks, measured precursor present and mass-consistent10ppm+0.002Da. Exclude all non-tims commercial-library comparator spectra. Preserve every remaining original exported usable acquisition.',
        chemistry_rule='Exclude prior used/reserved identity, C39 formula, nonempty scaffold or Morgan2 Tanimoto>=.70. Report training identity overlap; a future fitted component must be refitted with full holdout exclusions.',
        no_performance_sampling=True,prediction_outcomes_read=False,C13_labels_read=False,
        limitation='Public primary APCI-positive consensus library, not ESI/mixed mode or complete raw collision-energy acquisitions. No calibrated target transfer or qualification is claimed.')
    write(out/'inventory-protocol.json',protocol)
    features=module('c47_plasticizer_identity','c09e_features.py');structures={};invalid_sdf=0
    for molecule in Chem.SDMolSupplier(str(out/'Plasticizer_Data.SDF'),removeHs=False):
        if molecule is None:invalid_sdf+=1;continue
        raw_key=Chem.MolToInchiKey(molecule).split('-')[0];smiles=Chem.MolToSmiles(molecule)
        item=descriptor(smiles,features);item['smiles']=smiles
        if raw_key in structures:assert structures[raw_key]['key']==item['key']
        structures[raw_key]=item
    labels=pd.read_csv(c39,usecols=['key','smiles','formula']);used=set(labels.key)|set(pd.read_parquet(c13,columns=['key']).key)
    for path in oldlabels:used.update(pd.read_csv(path,usecols=['key']).key)
    q=[descriptor(s,features) for s in labels.smiles];qfps=[v['fp'] for v in q];forms=set(labels.formula);scaffolds={v['scaffold'] for v in q if v['scaffold']}
    metadata=pd.read_csv(out/'SampleDetails.csv',dtype=str).fillna('');counts=Counter();records=[];survivors=[]
    source=parse_mgf((out/'Plasticizer_Data.mgf').read_text(encoding='utf8'))
    for number,fields,peaks in source:
        instrument=one(fields,'Instrument');kind=one(fields,'Instrument_type');ionization=one(fields,'Ionization')
        raw_key=(one(fields,'InChIKey') or '').split('-')[0];adduct=one(fields,'Precursor_type');reason=None
        record=dict(original_record=number,raw_key=raw_key,instrument=instrument,instrument_type=kind,ionization=ionization,adduct=adduct,positive_finite_peaks=len(peaks))
        if '_source_missing_END_IONS' in fields:reason='malformed_exported_comparator_or_record'
        elif not (kind and kind.strip().lower()=='timstof' and instrument and 'Bruker timsTOF' in instrument and ionization=='APCI'):reason='non_primary_tims_APCI_comparator'
        elif raw_key not in structures:reason='no_source_SDF_structure'
        elif adduct not in cfg['allowed_query_adducts']:reason='unsupported_existing_target_adduct'
        elif len(peaks)<cfg['minimum_peaks']:reason='too_few_usable_peaks'
        if reason is None:
            matching=metadata[metadata.InChIKey.str[:14].eq(raw_key)&metadata['Instrument Type'].str.strip().eq('timsTOF')&metadata.Ionization.str.strip().eq('APCI')&metadata['Precursor Type'].str.strip().eq(adduct)]
            if len(matching)!=1:reason='ambiguous_or_missing_primary_metadata'
        if reason is None:
            measured=set(re.findall(r'(?:^|\s)Parent=([0-9.]+)', ' '.join(fields.get('Comments',[]))))
            if len(measured)!=1:reason='missing_or_ambiguous_measured_precursor'
        if reason is None:
            item=structures[raw_key];precursor=float(next(iter(measured)));_,canonical,formula,mass,_=features.structure(item['smiles'])
            offsets={'[M+H]+':1.007276,'[M-H]-':-1.007276,'[M+Na]+':22.989218,'[M+NH4]+':18.033823,'[M+K]+':38.963158,'[M+Cl]-':34.969402,'[M+CH2O2-H]-':44.998201}
            assert adduct in offsets
            error=abs(precursor-(mass+offsets[adduct]));maximum=max(DataStructs.BulkTanimotoSimilarity(item['fp'],qfps))
            record.update(key=item['key'],smiles=canonical,formula=formula,precursor_mz=precursor,mass_error_da=error,maximum_C39_tanimoto=maximum)
            if error>precursor*10e-6+.002:reason='precursor_mass_inconsistent'
            elif item['key'] in used:reason='used_or_reserved_identity'
            elif formula in forms:reason='C39_formula'
            elif item['scaffold'] and item['scaffold'] in scaffolds:reason='C39_nonempty_scaffold'
            elif maximum>=.7:reason='C39_similarity'
        record['exclusion_reason']=reason;records.append(record);counts[reason or 'inventory_survivor']+=1
        if reason is None:
            a=np.asarray(peaks,np.float64);order=np.argsort(a[:,0],kind='stable');a=a[order]
            survivors.append(dict(original_record=number,key=record['key'],smiles=record['smiles'],adduct=adduct,precursor_mz=precursor,
                ionization_mode='positive' if adduct.endswith('+') else 'negative',instrument_type='timsTOF',ionization_method='APCI',
                collision_energy_source=one(fields,'Collision_energy'),ms2_mzs=a[:,0],ms2_normalized_intensities=a[:,1]/a[:,1].max()))
    pd.DataFrame(records).to_parquet(out/'all-record-audit.parquet',index=False)
    pd.DataFrame(survivors).to_parquet(out/'inventory-survivor-acquisitions.parquet',index=False)
    keys={v['key'] for v in survivors};train=pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14']);known=set(train.inchikey14.astype(str).str[:14])
    cache=pd.read_parquet(ROOT/'outputs/local/long-frozen-20261107-r1/molecules.parquet',columns=['key']);fitted=np.load(ROOT/'outputs/local/c40r1-crossmodal-heads-20261111/fitting-identities.npy',allow_pickle=False)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'inventory-protocol.json'),exported_records=len(source),
        supplied_SDF_structures=len(structures),invalid_SDF_records=invalid_sdf,first_exclusion_reason_counts=dict(counts),
        survivor_acquisitions=len(survivors),survivor_identities=len(keys),supplied_training_raw_key_overlap=len(keys&known),C40R1_fit_identity_overlap=len(keys&set(cache.key.iloc[fitted])),
        files_sha256={n:sha(out/n) for n in ['all-record-audit.parquet','inventory-survivor-acquisitions.parquet']},
        C13_read=False,predictions_read=False,qualified_validation=False,mixed_query_support=False,whole_original_raw_acquisitions_available=False,
        limitation=protocol['limitation'])
    write(out/'inventory.json',result);print('C47 PUBLIC PRIMARY APCI INVENTORY',json.dumps(result))


if __name__=='__main__':main()
