"""Metadata-only source/support inventory; does not select candidates or score."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from precision_cache import ROOT,sha,write
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c38-source-support-20261111'
LIBRARIES=['GNPS-REFRAME-DRUG-CCS-LIBRARY','GNPS-ION-MOBILITY-LIBRARY-FILTERED','GNPS-ION-MOBILITY-LIBRARY']


def main():
    pa.set_cpu_count(2);assert system_memory()['free_gib']>=4
    authorization=json.loads((OUT/'authorization.json').read_text())
    assert authorization['active'] and authorization['submissions_approved']==2
    assert not (OUT/'source-inventory.json').exists()
    raw=ROOT/'outputs/external/2026-11-06'
    external=ROOT/'outputs/external/cache-20261106'
    inputs={n:ROOT/p for n,p in dict(
        massbank='outputs/external/2026-11-06/massbank.parquet',gnps='outputs/external/2026-11-06/gnps.parquet',
        accepted='outputs/external/cache-20261106/spectra.parquet',molecules='outputs/external/cache-20261106/molecules.parquet',
        own_catalogue='outputs/local/n0/cache/molecules.parquet',raw_training='outputs/competition-raw/train.parquet',
        production_reference='outputs/research/c08-pipeline/reference/train_structs.parquet').items()}
    # Existing manifest pins avoid repeated hashing of the3GB raw training file.
    previous=json.loads((ROOT/'outputs/local/c36-validation-repair-20261111/input-contract.json').read_text())
    paths={n:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for n,p in inputs.items() if n!='raw_training'}
    paths['raw_training']=dict(path=str(inputs['raw_training'].relative_to(ROOT)),bytes=inputs['raw_training'].stat().st_size,
                             scope='Read existing label-key column only for exclusion; no query outcomes')
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),inputs=paths,
                  libraries=LIBRARIES,primary_instrument_source='https://ccms-ucsd.github.io/GNPSDocumentation/gnpslibraries/',
                  alias_rule='Only the three named GNPS libraries have documented timsTOF provenance; preserve reported ToF separately.',
                  eligibility='Inventory of previously admitted records only. Full-acquisition export, raw/near-spectrum exclusion and primary-record validation remain required.',
                  outcome_metrics_read=False,C13_read=False,source_selection_based_on_query_performance=False)
    write(OUT/'source-inventory-protocol.json',protocol)
    mol=pd.read_parquet(inputs['molecules'])
    admitted=pd.read_parquet(inputs['accepted'],columns=['mol_idx','source','accession','library'])
    admitted['metric_key']=mol.key.to_numpy()[admitted.mol_idx]
    forbidden=set(pd.read_parquet(inputs['own_catalogue'],columns=['key']).key)
    forbidden.update(pd.read_parquet(inputs['raw_training'],columns=['inchikey14']).inchikey14.dropna().astype(str).str[:14])
    ref=pd.read_parquet(inputs['production_reference'],columns=['key','rawkey','inchikey14'])
    for c in ref:forbidden.update(ref[c].dropna().astype(str).str[:14])
    used=set()
    for path in ['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
                 'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']:
        frame=pd.read_csv(ROOT/path,usecols=['key']);used.update(frame.key)
    g=pd.read_parquet(inputs['gnps'],columns=['spectrum_id','library','instrument','compound_source','library_class','ion_mode','inchikey'],filters=[('library','in',LIBRARIES)])
    g=g.rename(columns={'spectrum_id':'accession'});g['source']=2
    m=pd.read_parquet(inputs['massbank'],columns=['accession','instrument','instrument_type','spectrum_type','ion_mode','precursor_type','inchikey','license'])
    m=m[m.instrument.str.contains('tims',case=False,na=False)].copy();m['source']=1;m['library']='MassBank-timsTOF'
    reports={};witnesses=[]
    for name,f in [('gnps_documented_timsTOF',g),('massbank_explicit_timsTOF',m)]:
        joined=f.merge(admitted[['source','accession','metric_key']],on=['source','accession'],how='left',validate='one_to_one')
        joined['previously_admitted']=joined.metric_key.notna()
        joined['in_training_or_reference']=joined.metric_key.isin(forbidden)
        joined['in_consumed_panel']=joined.metric_key.isin(used)
        joined['inventory_eligible']=joined.previously_admitted&~joined.in_training_or_reference&~joined.in_consumed_panel
        def summary(t):
            eligible=t[t.inventory_eligible]
            return dict(raw_records=len(t),raw_supplied_keys=int(t.inchikey.dropna().astype(str).str[:14].nunique()),
                        previously_admitted_records=int(t.previously_admitted.sum()),admitted_unique_identities=int(t.metric_key.nunique()),
                        excluded_training_or_reference_records=int((t.previously_admitted&t.in_training_or_reference).sum()),
                        excluded_consumed_panel_records=int((t.previously_admitted&t.in_consumed_panel).sum()),
                        inventory_eligible_records=len(eligible),inventory_eligible_identities=int(eligible.metric_key.nunique()),
                        raw_modes=t.ion_mode.value_counts().to_dict(),eligible_modes=eligible.ion_mode.value_counts().to_dict())
        reports[name]=dict(all=summary(joined),by_library={n:summary(t) for n,t in joined.groupby('library')})
        joined.to_parquet(OUT/(name+'-inventory.parquet'),index=False)
        witnesses.append(joined[['source','accession','metric_key','inventory_eligible']])
    combined=pd.concat(witnesses,ignore_index=True);eligible=combined[combined.inventory_eligible]
    eligible.to_parquet(OUT/'source-eligible-records.parquet',index=False)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'source-inventory-protocol.json'),
                reports=reports,unique_eligible_identities=int(eligible.metric_key.nunique()),forbidden_identity_count=len(forbidden),consumed_identity_count=len(used),
                reserved_C13_labels_or_predictions_read=False,quality_metrics_computed=False,new_inference=False,
                primary_source_checks_remaining=True,full_acquisitions_not_yet_exported=True,
                known_limitation='GNPS metadata says Crude and only positive mode for these libraries; source documentation alone does not establish every identity as an authentic-standard truth. Same-lab GNPS collections are not independent laboratories. Exact source and purity/annotation review is required.',
                qualified_validation_population=False)
    write(OUT/'source-inventory.json',result);print('DONE C38 SOURCE INVENTORY',json.dumps(result),flush=True)


if __name__=='__main__':main()
