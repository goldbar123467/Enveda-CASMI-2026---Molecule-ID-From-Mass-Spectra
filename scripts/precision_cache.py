"""Precision-preserving compatible-scorer input cache. No GPU or model training.

Uses the existing credited feature/identity catalogue; unmapped raw identities
are counted explicitly. The old FP16 cache is never rewritten.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
from datetime import datetime,timezone
import gc,hashlib,json,sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/local/kit'))
import features


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while b:=stream.read(4*1024*1024):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def prepare(spec,notify=lambda **kw:None):
    out=ROOT/spec['cache'];out.mkdir(parents=True,exist_ok=True);pa.set_cpu_count(2)
    cfg=spec['training'];contract=dict(raw_train_sha256=sha(ROOT/spec['raw_training']),
        catalogue_sha256=sha(ROOT/spec['base_cache']/'molecules.parquet'),feature_sha256=sha(ROOT/'outputs/local/kit/features.py'),
        cache_builder_sha256=sha(__file__),views_per_identity=spec['views_per_identity'],max_peaks=cfg['max_peaks'],min_peaks=cfg['min_peaks'],dtype='float32',sorting='legacy features.peaks uses stable intensity and m/z ordering')
    if (out/'prepared.json').exists():
        receipt=json.loads((out/'prepared.json').read_text());assert receipt['contract']==contract,'Cache contract changed'
        return receipt
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==contract
    else:write(out/'contract.json',contract)
    mol=pd.read_parquet(ROOT/spec['base_cache']/'molecules.parquet');lookup=dict(zip(mol.key,range(len(mol))))
    if not (out/'selected.parquet').exists():
        notify(stage='selecting precise acquisitions')
        cols=['inchikey14','adduct','ionization_mode','precursor_mz','num_peaks','ingest_lib']
        raw=pd.read_parquet(ROOT/spec['raw_training'],columns=cols);mapped=raw.inchikey14.map(lookup)
        good=mapped.notna()&raw.precursor_mz.gt(0)&np.isfinite(raw.precursor_mz)&raw.num_peaks.ge(cfg['min_peaks'])
        selected=raw.loc[good].copy();selected['mol_idx']=mapped[good].astype(np.int32);selected['row']=selected.index.to_numpy()
        selected['rank']=(selected.row.to_numpy(np.uint64)*np.uint64(11400714819323198485))^np.uint64(20261108)
        diverse=selected.sort_values('rank',kind='stable').drop_duplicates(['mol_idx','adduct','ionization_mode','ingest_lib'])
        selected=pd.concat([diverse,selected.sort_values('rank',kind='stable')]).drop_duplicates('row').groupby('mol_idx',sort=False).head(spec['views_per_identity'])
        selected=selected[['row','mol_idx','adduct','ionization_mode','precursor_mz','ingest_lib']].sort_values('row',kind='stable').reset_index(drop=True)
        selected.to_parquet(out/'selected.parquet',index=False);mol.to_parquet(out/'molecules.parquet',index=False)
        write(out/'selection.json',dict(raw_rows=len(raw),unmapped_raw_rows=int(mapped.isna().sum()),selected_rows=len(selected),identities=int(selected.mol_idx.nunique()),
             limitation='Catalogue mapping uses existing canonical InChIKey14; unmatched raw records are excluded, never guessed'))
        del raw,mapped,diverse;gc.collect()
    selected=pd.read_parquet(out/'selected.parquet');total=len(selected)
    def array(name,dtype,shape):
        p=out/(name+'.npy');exists=p.exists()
        value=np.lib.format.open_memmap(p,mode='r+' if exists else 'w+',dtype=dtype,shape=shape)
        if name=='valid' and not exists:value[:]=False;value.flush()
        return value
    peaks=array('peaks',np.float32,(total,cfg['max_peaks'],4));meta=array('metadata',np.float32,(total,4))
    adduct=array('adduct',np.int64,(total,));lengths=array('lengths',np.int16,(total,));valid=array('valid',np.bool_,(total,))
    n=pq.ParquetFile(ROOT/spec['raw_training']).metadata.num_rows;mapping=np.full(n,-1,np.int32);mapping[selected.row.to_numpy()]=np.arange(total,dtype=np.int32)
    cursor=json.loads((out/'progress.json').read_text()).get('raw_rows_seen',0) if (out/'progress.json').exists() else 0
    offset=0;errors=0
    cols=['ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','ionization_mode','collision_energy_ev']
    for batch in pq.ParquetFile(ROOT/spec['raw_training']).iter_batches(batch_size=2048,columns=cols):
        end=offset+len(batch)
        if end<=cursor:offset=end;continue
        positions=mapping[offset:end];keep=positions>=0
        if keep.any():
            frame=batch.filter(pa.array(keep)).to_pandas()
            for index,row in zip(positions[keep],frame.itertuples(index=False)):
                valid[index]=False
                try:
                    x,k=features.peaks(row.ms2_mzs,row.ms2_normalized_intensities,row.precursor_mz,cfg['max_peaks'])
                    if k<cfg['min_peaks'] or not np.isfinite(x).all():continue
                    m,a=features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev)
                    assert np.isfinite(m).all()
                    peaks[index]=x;lengths[index]=k;meta[index]=m;adduct[index]=a;valid[index]=True
                except (ValueError,TypeError):errors+=1
        offset=end
        if offset%32768<2048 or offset==n:
            for value in [peaks,meta,adduct,lengths,valid]:value.flush()
            write(out/'progress.json',dict(raw_rows_seen=offset,raw_rows_total=n,updated_utc=datetime.now(timezone.utc).isoformat()))
            notify(stage='building FP32 peak cache',stage_completed=offset,stage_total=n)
    usable=int(valid.sum());assert usable>=.99*total,'More than1% unusable selected views'
    report=dict(contract=contract,selected_spectra=total,usable_spectra=usable,invalid_selected_spectra=total-usable,
        identities=int(selected.loc[np.asarray(valid),'mol_idx'].nunique()),valid_sha256=sha(out/'valid.npy'),
        arrays={name:dict(bytes=(out/(name+'.npy')).stat().st_size,sha256=sha(out/(name+'.npy'))) for name in ['peaks','metadata','adduct','lengths']},
        input_errors_seen_in_last_preparation_process=errors,completed_utc=datetime.now(timezone.utc).isoformat())
    write(out/'prepared.json',report);notify(stage='precision cache ready',stage_completed=usable,stage_total=total)
    return report
