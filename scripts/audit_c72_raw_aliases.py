"""Outcome-blind raw acquisition alias census for every auxiliary query.

Matched reference identities are quarantined from future fitting/reference use;
queries are never removed because of an overlap. This bounded equality audit
does not establish universal provenance or arbitrary near-spectrum exclusion.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from c45_complete_parent import ROOT,read,write,sha,module
OUT=ROOT/'outputs/local/c72-clean-target-holdout-20261112'


def main():
    pa.set_cpu_count(2);target=OUT/'raw-source-alias-audit.json';assert not target.exists()
    freeze=read(OUT/'validation-freeze.json');query_path=ROOT/'outputs/publish/c72-clean-target-holdout-20261112/validation-queries.parquet'
    assert freeze['files_sha256'][str(query_path.relative_to(ROOT))]==sha(query_path)
    query=pd.read_parquet(query_path);assert len(query)==freeze['acquisitions'] and query.molecule_id.nunique()==freeze['queries']
    assert not {'key','smiles','formula','inchikey14'}&set(query.columns)
    raw=ROOT/'outputs/competition-raw/train.parquet';pin=read(ROOT/'outputs/local/c39-source-supported-validation-20261111/preparation-protocol.json')['raw_training_sha256']
    assert sha(raw)==pin;source_sha=sha(__file__)
    write(OUT/'raw-source-alias-protocol.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_sha,
        query_sha256=sha(query_path),raw_training_sha256=pin,panel_freeze_sha256=sha(OUT/'validation-freeze.json'),
        rule='All query acquisitions; raw precursor windows20ppm+0.003Da inclusive. Original full1mDa/intensity1e-4 and FP16top96 signatures, independent of adduct/CE/instrument. No prediction labels matched. Keep every query; use matched raw keys/official aliases only for future fit/reference exclusions.',
        C13_labels_read=False,quality_outcomes_read=False))
    features=module('c48_alias_features','c09e_features.py');signatures=module('c48_alias_signature','c09_spectrum_audit.py').signatures
    lookup=[defaultdict(set),defaultdict(set)]
    for row in query.itertuples(index=False):
        values=signatures(row,features);assert values is not None
        for kind,value in enumerate(values):lookup[kind][value].add(row.molecule_id)
    precursor=pd.read_parquet(raw,columns=['precursor_mz']).precursor_mz.to_numpy(np.float64)
    order=np.argsort(precursor,kind='stable');sorted_values=precursor[order];wanted=np.zeros(len(precursor),bool)
    for center in query.precursor_mz:
        width=center*20e-6+.003;low=np.searchsorted(sorted_values,center-width,side='left');high=np.searchsorted(sorted_values,center+width,side='right')
        wanted[order[low:high]]=True
    parquet=pq.ParquetFile(raw);assert parquet.metadata.num_rows==len(precursor)
    records=[];offset=0;checked=0
    for group in range(parquet.num_row_groups):
        count=parquet.metadata.row_group(group).num_rows;positions=np.flatnonzero(wanted[offset:offset+count])
        if len(positions):
            table=parquet.read_row_group(group,columns=['precursor_mz','ms2_mzs','ms2_normalized_intensities','inchikey14','ingest_lib'],use_threads=False)
            frame=table.take(pa.array(positions)).to_pandas();del table
            for position,row in zip(positions,frame.itertuples(index=False)):
                checked+=1;values=signatures(row,features)
                if values is None:continue
                for kind,value in enumerate(values):
                    for mid in lookup[kind].get(value,[]):records.append(dict(molecule_id=mid,raw_row=offset+int(position),raw_key=str(row.inchikey14),
                        ingest_lib=str(row.ingest_lib),signature='full_1mDa_1e-4' if kind==0 else 'FP16_top96'))
            del frame
        offset+=count
        write(OUT/'raw-source-alias-progress.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage='raw_acquisition_census',
            original_rows_completed=offset,original_rows=len(precursor),candidate_rows_checked=checked,match_records=len(records)))
    assert checked==int(wanted.sum()) and offset==len(precursor) and sha(__file__)==source_sha
    frame=pd.DataFrame(records,columns=['molecule_id','raw_row','raw_key','ingest_lib','signature']);frame.to_parquet(OUT/'raw-source-alias-matches.parquet',index=False)
    keys=sorted(set(frame.raw_key));write(OUT/'additional-raw-alias-quarantine.json',dict(keys=keys,scope='Global fit/reference quarantine; no per-query truth map or cohort removal.'))
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_sha,protocol_sha256=sha(OUT/'raw-source-alias-protocol.json'),
        queries=freeze['queries'],query_acquisitions=len(query),raw_rows=len(precursor),raw_precursor_candidate_rows=checked,
        match_records=len(frame),query_identities_with_any_signature_match=frame.molecule_id.nunique(),matched_raw_keys=len(keys),
        overlap_queries_by_signature={name:int(group.molecule_id.nunique()) for name,group in frame.groupby('signature')},
        files_sha256={name:sha(OUT/name) for name in ['raw-source-alias-matches.parquet','additional-raw-alias-quarantine.json']},
        queries_removed=0,C13_labels_read=False,prediction_outcomes_read=False,
        limitation='Bounded precursor equality signatures; not universal exposure exclusion. Frozen public encoders/parent historical exposure remains known or unknown.')
    write(target,result);print('C72 ALL QUERY RAW ALIAS AUDIT',__import__('json').dumps(result))


if __name__=='__main__':main()
