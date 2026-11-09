"""Cloud CPU audit: query/reference peak overlap, ignoring acquisition metadata.

Uses actual raw competition spectra and the sealed C09 queries, never query
structure labels. Narrow precursor windows bound work; report that limitation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import collections,hashlib,json,sys
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def signatures(row,features):
    mz,it=features.array_value(row.ms2_mzs),features.array_value(row.ms2_normalized_intensities)
    if len(mz)!=len(it):return None
    keep=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0);mz,it=mz[keep],it[keep]
    if len(mz)<3:return None
    order=np.argsort(mz,kind='stable');mz,it=mz[order],it[order];it=it/it.max()
    full=np.column_stack([np.rint(mz*1000),np.rint(it*10000)]).astype('<i4')
    x,n=features.peaks(row.ms2_mzs,row.ms2_normalized_intensities,0.,96)
    return hashlib.sha256(full.tobytes()).hexdigest(),hashlib.sha256(x.astype('<f2').tobytes()+int(n).to_bytes(2,'little')).hexdigest()


def main():
    import features
    pa.set_cpu_count(2)
    input_root=Path('/kaggle/input');out=Path('/kaggle/working');contract_hits=list(input_root.rglob('validation-contract.json'))
    assert len(contract_hits)==1;contract=contract_hits[0];spec=json.loads(contract.read_text());assert spec['queries']==80
    source=contract.parent/'validation-queries.parquet';assert digest(source)==spec['files']['validation-queries.parquet']
    q=pd.read_parquet(source);assert q.molecule_id.nunique()==80
    assert not {'key','inchikey14','smiles','formula'}&set(q.columns)
    references=list(input_root.rglob('train.parquet'));assert len(references)==1;reference=references[0]
    table=pd.read_parquet(reference,columns=['precursor_mz']);prec=table.precursor_mz.to_numpy(dtype=np.float64)
    order=np.argsort(prec,kind='stable');sorted_prec=prec[order]
    wanted=set();query_signatures=[collections.defaultdict(set),collections.defaultdict(set)]
    for row in q.itertuples(index=False):
        values=signatures(row,features);assert values is not None
        for i,value in enumerate(values):query_signatures[i][value].add(str(row.molecule_id))
        center=float(row.precursor_mz);width=center*20e-6+.003
        lo,hi=np.searchsorted(sorted_prec,[center-width,center+width]);wanted.update(order[lo:hi].tolist())
    matches=[];offset=0
    columns=['precursor_mz','ms2_mzs','ms2_normalized_intensities','inchikey14','ingest_lib','instrument_type']
    available=pq.ParquetFile(reference).schema_arrow.names;columns=[c for c in columns if c in available]
    checked=0
    for batch in pq.ParquetFile(reference).iter_batches(batch_size=4096,columns=columns,use_threads=False):
        positions=[i for i in range(len(batch)) if offset+i in wanted]
        if positions:
            selected=batch.take(pa.array(positions)).to_pandas()
            for pos,row in zip(positions,selected.itertuples(index=False)):
                values=signatures(row,features);checked+=1
                if values is None:continue
                for i,value in enumerate(values):
                    for mid in query_signatures[i].get(value,[]):
                        matches.append(dict(molecule_id=mid,reference_row=offset+pos,
                            signature='full_peaks_1mDa_intensity1e-4' if i==0 else 'model_top96_fp16_ignore_metadata',
                            reference_key=str(getattr(row,'inchikey14','')),library=str(getattr(row,'ingest_lib','')),
                            instrument=str(getattr(row,'instrument_type',''))))
        offset+=len(batch)
        if offset%262144<4096:print('C09 RAW SPECTRUM AUDIT',offset,'/',len(prec),'candidate rows checked',checked,flush=True)
    frame=pd.DataFrame(matches,columns=['molecule_id','reference_row','signature','reference_key','library','instrument'])
    frame.to_csv(out/'c09_spectrum_matches.csv',index=False)
    report=dict(kind='raw_competition_peak_overlap_audit',queries=80,query_spectra=len(q),raw_reference_rows=len(prec),
        precursor_window='20ppm +0.003Da; peak signatures ignore CE/adduct/instrument metadata',candidate_rows=len(wanted),checked_rows=checked,
        overlap_queries_by_signature={name:int(g.molecule_id.nunique()) for name,g in frame.groupby('signature')},
        any_reference_overlap_queries=int(frame.molecule_id.nunique()),match_records=len(frame),query_identity_labels_used=False,
        within_query_panel_cross_identity_peak_aliases=[sum(len(v)>1 for v in values.values()) for values in query_signatures],
        validation_contract_sha256=digest(contract),query_sha256=digest(source),raw_reference_sha256=digest(reference),
        limitation='Bounded precursor window; rounded/full-peak and quantized model-input matches are conservative warnings, not proof of label leakage. Does not exhaustively test arbitrary tolerance/cross-instrument near-duplicates.')
    (out/'c09_spectrum_audit.json').write_text(json.dumps(report,indent=2));print('DONE C09 SPECTRUM AUDIT',json.dumps(report),flush=True)


if __name__=='__main__':main()
