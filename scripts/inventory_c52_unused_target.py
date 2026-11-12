"""Inventory unused target-domain chemistry before selecting or fitting a new pair.

No candidate predictions, quality ranks or reserved C13 query labels are read.
The global C13 identity inventory is used for quarantine only.
"""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import hashlib
from collections import Counter
from legacy_formate_compat import normalize_legacy_adduct
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from c45_complete_parent import ROOT,read,write,sha,stamp,module,descriptor
OUT=ROOT/'outputs/local/c52-unused-target-inventory-20261112'
CACHE=ROOT/'outputs/local/long-frozen-20261107-r1'
FIT=ROOT/'outputs/local/c48-clean-residual-heads-20261112'


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');OUT.mkdir(exist_ok=True);assert not (OUT/'inventory.json').exists()
    source_pin=sha(__file__);target=ROOT/'outputs/local/c37-selection-trace-20261111/unlabelled-target/test.parquet'
    used_paths=[ROOT/p for p in ['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
        'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv',
        'outputs/local/c39-source-supported-validation-20261111/labels.csv','outputs/local/c48-auxiliary-holdout-20261112/labels.csv']]
    reserved=ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet'
    raw=ROOT/'outputs/competition-raw/train.parquet';witness=FIT/'original-structure-identity-witnesses.parquet'
    write(OUT/'protocol.json',dict(frozen_utc=stamp(),source_sha256=source_pin,
        raw_training_sha256=sha(raw),global_reserved_inventory_sha256=sha(reserved),witness_sha256=sha(witness),
        used_sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in used_paths},
        candidate_pool='All NP examples and all39TIMS-Bench plant standards; first5000peractual query-mode Enveda180raw identities under SHA256(C52-unused-target-v1:+rawkey). This is a bounded inventory, not a population-wide eligible count.',
        rule='Exclude all prior raw/official quarantine components, all used/reserved identities, same formula, same nonempty scaffold, Morgan2Tanimoto>=.70 to used chemistry and catalogue-resolved global C13 identities. No outcome-based inclusion.',
        proposed_next_panel='All usable remaining NP/plant standard chemistry, plus384newEnveda180queries allocated265positive/93mixed/26negative from target400mode proportions276/97/27. Preserve all usable acquisitions. No panel selected by quality.',
        prediction_quality_read=False,C13_labels_or_predictions_read=False))
    mol=pd.read_parquet(CACHE/'molecules.parquet');assert len(mol)==273833
    used=pd.concat([pd.read_csv(p,usecols=lambda c:c in {'key','smiles','formula'}) for p in used_paths],ignore_index=True).drop_duplicates('key')
    reserved_keys=set(pd.read_parquet(reserved,columns=['key']).key)
    blocked_keys=set(used.key)|reserved_keys
    known_reserved=mol[mol.key.isin(reserved_keys)][['key','smiles','formula']]
    chemistry=pd.concat([used,known_reserved],ignore_index=True).drop_duplicates('key')
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    cmols=[Chem.MolFromSmiles(s) for s in chemistry.smiles];assert all(m is not None for m in cmols)
    forms=set(chemistry.formula);scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in cmols};scaffolds.discard('')
    fps=[gen.GetFingerprint(m) for m in cmols]
    w=pd.read_parquet(witness);blocked_raw=set(read(FIT/'engine2-raw-key-exclusion.json')['keys'])
    blocked_official=set(w.loc[w.raw_key.isin(blocked_raw),'official_key'].dropna())|blocked_keys
    mapping=w[w.official_key.notna()].drop_duplicates(['raw_key','official_key'])
    ambiguous=set(mapping.groupby('raw_key').official_key.nunique().loc[lambda s:s>1].index)
    raw_to_official=mapping.drop_duplicates('raw_key').set_index('raw_key').official_key.to_dict()
    canonical=mol.drop_duplicates('key').set_index('key')[['smiles','formula']].to_dict('index')
    features=module('c52_inventory_features','c09e_features.py');records=[];reports={}
    def classify(source,key,smiles,formula,mode,acquisitions,rawkey=None):
        reason=None;maximum=None
        if key in blocked_official:reason='used_reserved_or_prior_quarantine_identity'
        elif rawkey in blocked_raw or rawkey in ambiguous:reason='prior_raw_quarantine_or_ambiguous_identity'
        elif formula in forms:reason='used_or_reserved_formula'
        else:
            m=Chem.MolFromSmiles(smiles);assert m is not None
            scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
            if scaffold and scaffold in scaffolds:reason='used_or_reserved_nonempty_scaffold'
            else:
                maximum=max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),fps),default=0.)
                if maximum>=.7:reason='used_or_reserved_near_chemistry'
        records.append(dict(source=source,key=key,smiles=smiles,formula=formula,mode=mode,acquisitions=acquisitions,raw_key=rawkey,
            excluded=reason is not None,reason=reason,maximum_used_reserved_tanimoto=maximum))
    allowed=set(read(ROOT/'configs/c39-source-supported-validation-20261111.json')['allowed_query_adducts'])
    columns=['ingest_lib','inchikey14','adduct','ionization_mode','num_peaks','precursor_mz']
    for source in ['enveda-np-examples','enveda-180']:
        data=pd.read_parquet(raw,columns=columns,filters=[('ingest_lib','==',source)])
        data=data[data.adduct.map(normalize_legacy_adduct).isin(allowed)&data.num_peaks.ge(3)&data.precursor_mz.gt(0)]
        modes=data.groupby('inchikey14').ionization_mode.agg(lambda s:'mixed' if s.nunique()>1 else s.iloc[0]);counts=data.groupby('inchikey14').size()
        candidates=list(modes.index)
        if source=='enveda-180':candidates=sum([sorted(modes.index[modes.eq(mode)],key=lambda k:hashlib.sha256(('C52-unused-target-v1:'+k).encode()).hexdigest())[:5000] for mode in ['positive','mixed','negative']],[])
        unresolved=0
        for key in candidates:
            official=raw_to_official.get(key);s=canonical.get(official)
            if s is None:unresolved+=1;continue
            classify(source,official,s['smiles'],s['formula'],modes.loc[key],int(counts.loc[key]),key)
        reports[source]=dict(original_supported_raw_identities=len(modes),original_supported_acquisitions=len(data),bounded_candidate_raw_identities=len(candidates),unresolved_official_identity=unresolved)
        print('C52 SOURCE INVENTORIED',source,len(candidates),flush=True)
    plant_root=ROOT/'outputs/local/c47-remaining-tims-inventory-20261112/source-research/tims-bench-source-files'
    plant=pd.read_parquet(plant_root/'plant_spikein_lib.pq');smilesmap=read(plant_root/'inchikey_smiles_map.json')
    for rawkey,g in plant.groupby('inchikey_2d',sort=True):
        source_key=rawkey.split('-')[0];text=smilesmap.get(rawkey)
        if text is None:text=canonical[raw_to_official[source_key]]['smiles']
        d=descriptor(text,features)
        assert d['key']==raw_to_official[source_key], 'Plant identity source/canonical-witness mismatch'
        classify('TIMS-Bench-plant-standard-library',d['key'],d['canonical'],d['formula'],'metadata_not_yet_established',len(g),rawkey.split('-')[0])
    frame=pd.DataFrame(records);frame.to_parquet(OUT/'candidate-chemistry-inventory.parquet',index=False)
    for source,g in frame.groupby('source'):
        reports.setdefault(source,{})
        reports[source].update(inventoried_records=len(g),eligible_official_identities=g.loc[~g.excluded,'key'].nunique(),
            eligible_acquisitions=int(g.loc[~g.excluded,'acquisitions'].sum()),eligible_modes=g.loc[~g.excluded].groupby('mode').key.nunique().to_dict(),
            nonexclusive_reason_counts=g.reason.value_counts().to_dict())
    assert sha(__file__)==source_pin
    write(OUT/'inventory.json',dict(completed_utc=stamp(),source_sha256=source_pin,protocol_sha256=sha(OUT/'protocol.json'),
        inventory_sha256=sha(OUT/'candidate-chemistry-inventory.parquet'),reports=reports,used_chemistry_identities=len(used),
        global_C13_identity_quarantine=len(reserved_keys),global_C13_catalogue_resolved_chemistry=len(known_reserved),
        no_panel_selected=True,no_prediction_quality_read=True,C13_labels_or_predictions_read=False,
        limitation='Existing public encoder/parent pretraining exposure persists. Enveda collections are target-domain controlled in-corpus data, not independent laboratories. Plant source metadata/provenance and acquisition fidelity still require verification.'))
    print('C52 INVENTORY',reports,flush=True)


if __name__=='__main__':main()
