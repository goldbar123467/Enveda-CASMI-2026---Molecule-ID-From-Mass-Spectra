"""Scan the pinned latest primary MassBank archive for new timsTOF records."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json,re,zipfile,sys
from collections import Counter
from datetime import datetime,timezone
import pandas as pd
import pyarrow as pa
from rdkit import RDLogger
from precision_cache import ROOT,sha,write,features
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c38-source-support-20261111'


def main():
    pa.set_cpu_count(2);RDLogger.DisableLog('rdApp.*');assert system_memory()['free_gib']>=4
    assert json.loads((OUT/'authorization.json').read_text())['active']
    assert not (OUT/'massbank-latest-inventory.json').exists()
    folder=ROOT/'outputs/research/c10-massbank-current-primary-20261110'
    archive=folder/'MassBank-data-2026.11-pinned.zip';audit=json.loads((folder/'audit.json').read_text())
    assert sha(archive)==audit['archive_sha256']
    write(OUT/'massbank-latest-protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        archive_sha256=audit['archive_sha256'],official_tag=audit['official_tag'],official_commit=audit['official_commit'],
        selection='All original MS2 records whose own instrument field explicitly identifies tims/tims-tof; no candidate or outcome filter.',
        exclusions='Official canonical identity outside own training catalogue, raw competition labels and production reference, and already consumed panels.',
        prediction_files_read=False,reserved_C13_read=False))
    forbidden=set(pd.read_parquet(ROOT/'outputs/local/n0/cache/molecules.parquet',columns=['key']).key)
    forbidden.update(pd.read_parquet(ROOT/'outputs/competition-raw/train.parquet',columns=['inchikey14']).inchikey14.dropna())
    f=pd.read_parquet(ROOT/'outputs/research/c08-pipeline/reference/train_structs.parquet',columns=['key','rawkey','inchikey14'])
    for c in f:forbidden.update(f[c].dropna().astype(str).str[:14])
    used=set()
    for p in ['outputs/publish/c08-validation-20261108/labels.csv','outputs/publish/c09-validation-20261109/labels.csv',
              'outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']:
        used.update(pd.read_csv(ROOT/p,usecols=['key']).key)
    counts=Counter();records=[];cache={}
    with zipfile.ZipFile(archive) as z:
        for i,name in enumerate(z.namelist()):
            if not name.endswith('.txt'):continue
            counts['text_records_scanned']+=1
            text=z.read(name).decode('utf8',errors='replace')
            fields={}
            for line in text.splitlines():
                if line.startswith(('ACCESSION:','AC$INSTRUMENT:','AC$INSTRUMENT_TYPE:','CH$SMILES:','LICENSE:','AUTHORS:','PUBLICATION:')):
                    k,v=line.split(':',1);fields[k]=v.strip()
            instrument=fields.get('AC$INSTRUMENT','')
            if not re.search('tims',instrument,re.I):
                if counts['text_records_scanned']%10000==0:print('C38 MASSBANK SCAN',counts['text_records_scanned'],'TIMS',len(records),flush=True)
                continue
            counts['explicit_tims_records']+=1
            if 'AC$MASS_SPECTROMETRY: MS_TYPE MS2' not in text:counts['not_ms2']+=1;continue
            smiles=fields.get('CH$SMILES','')
            if smiles not in cache:
                try:cache[smiles]=features.structure(smiles)[0] if smiles and '.' not in smiles else None
                except (ValueError,TypeError,RuntimeError):cache[smiles]=None
            key=cache[smiles]
            mode=re.search(r'^AC\$MASS_SPECTROMETRY: ION_MODE (.+)$',text,re.M)
            record=dict(accession=fields.get('ACCESSION'),archive_member=name,source='MassBank2026.11',instrument=instrument,
                instrument_type=fields.get('AC$INSTRUMENT_TYPE',''),mode=mode.group(1).strip() if mode else '',
                metric_key=key,smiles=smiles,license=fields.get('LICENSE',''),authors=fields.get('AUTHORS',''),publication=fields.get('PUBLICATION',''),
                valid_identity=key is not None,in_training_or_reference=key in forbidden,in_consumed_panel=key in used,
                eligible=key is not None and key not in forbidden and key not in used,
                original_record_sha256=__import__('hashlib').sha256(text.encode()).hexdigest())
            records.append(record)
    f=pd.DataFrame(records);assert len(f)>0
    f.to_parquet(OUT/'massbank-latest-tims-inventory.parquet',index=False)
    good=f[f.eligible]
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'massbank-latest-protocol.json'),
        counts=dict(counts),tims_ms2_records=len(f),tims_identities=int(f.metric_key.nunique()),
        eligible_records=len(good),eligible_identities=int(good.metric_key.nunique()),eligible_modes=good['mode'].value_counts().to_dict(),
        instruments=f.instrument.value_counts().to_dict(),eligible_instruments=good.instrument.value_counts().to_dict(),
        eligible_licenses=good.license.value_counts().to_dict(),inventory_sha256=sha(OUT/'massbank-latest-tims-inventory.parquet'),
        quality_metrics_read=False,reserved_C13_read=False,qualified_validation=False,
        limitation='Primary-record novelty inventory only. Acquisition export, exact/near spectrum tests, chemistry/source partitioning and metadata support checks remain required.')
    write(OUT/'massbank-latest-inventory.json',result);print('DONE C38 MASSBANK LATEST',json.dumps(result),flush=True)


if __name__=='__main__':main()
