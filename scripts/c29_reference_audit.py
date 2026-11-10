"""Bounded query/reference acquisition signatures, without any structure labels."""
import hashlib
import numpy as np


def signatures(mz,it,features):
    mz,it=np.asarray(mz,dtype=np.float64),np.asarray(it,dtype=np.float64)
    keep=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0);mz,it=mz[keep],it[keep]
    if len(mz)<3:return ()
    order=np.argsort(mz,kind='stable');mz,it=mz[order],it[order];it=it/it.max()
    full=np.column_stack([np.rint(mz*1000),np.rint(it*10000)]).astype('<i4')
    x,n=features.peaks(mz,it,0.,96)
    return hashlib.sha256(full.tobytes()).hexdigest(),hashlib.sha256(x.astype('<f2').tobytes()+int(n).to_bytes(2,'little')).hexdigest()


def reference_view(refs,queries,features,exclude,raw_queries=None):
    order=np.argsort([r[1] for r in refs],kind='stable');ordered=[refs[i] for i in order]
    precursor=np.array([r[1] for r in ordered]);wanted=set();sigs=set()
    if raw_queries is not None:
        for row in raw_queries.itertuples(index=False):
            sigs.update(signatures(row.ms2_mzs,row.ms2_normalized_intensities,features))
    for group in queries.values():
        for q in group:
            sigs.update(signatures(q['mz'],q['it'],features));w=q['prec']*10e-6+.002
            lo=np.searchsorted(precursor,q['prec']-w,'left');hi=np.searchsorted(precursor,q['prec']+w,'right')
            wanted.update(range(lo,hi))
    aliases={i for i in wanted if set(signatures(ordered[i][2],ordered[i][3],features))&sigs}
    result=[r for i,r in enumerate(ordered) if not exclude or i not in aliases]
    audit=dict(reference_rows=len(refs),precursor_candidate_rows=len(wanted),input_alias_rows=len(aliases),
        excluded=bool(exclude),retained_rows=len(result),labels_used=False,
        limitation='Only precursor10ppm+.002Da candidate rows; rounded full peaks and FP16top96 signatures. No universal provenance or pretraining exclusion claim.')
    return result,audit
