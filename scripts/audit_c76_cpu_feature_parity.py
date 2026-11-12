"""Locate CPU preprocessing/prediction differences; no truth or quality data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import torch
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c76-numerical-portability-20261112/cpu-feature-parity'
    out=base/'independent-comparison.json';assert not out.exists()
    a=read(base/'local/feature-probe.json');b=read(base/'cloud/feature-probe.json')
    assert read(base/'cloud-collection-audit.json')['remote_source_exact']
    assert a['jobs']==b['jobs'] and len(a['jobs'])==4
    aa=np.load(base/'local/feature-arrays.npz');bb=np.load(base/'cloud/feature-arrays.npz')
    assert set(aa.files)==set(bb.files)
    fields={}
    for key in aa.files:
        x,y=aa[key],bb[key]
        entry=dict(shape_equal=x.shape==y.shape,dtype_equal=x.dtype==y.dtype)
        if x.shape==y.shape:
            entry['exact']=bool(np.array_equal(x,y,equal_nan=True))
            finite=np.isfinite(x)&np.isfinite(y)
            entry['maximum_absolute_difference']=float(np.abs(x.astype(float)[finite]-y.astype(float)[finite]).max(initial=0))
            entry['outside_1e5_tolerance']=int((~np.isclose(x,y,atol=1e-5,rtol=1e-5,equal_nan=True)).sum())
        fields[key]=entry
    source_names=set(a['package_files_sha256'])|set(b['package_files_sha256'])
    changed=[n for n in source_names if a['package_files_sha256'].get(n)!=b['package_files_sha256'].get(n)]
    result=dict(completed_utc=stamp(),source_sha256=sha(__file__),
        local_receipt_sha256=sha(base/'local/feature-probe.json'),cloud_receipt_sha256=sha(base/'cloud/feature-probe.json'),
        canonical_strings_exact=a['canonical']==b['canonical'],checkpoint_exact=a['checkpoint_sha256']==b['checkpoint_sha256'],
        checkpoint_sha256=a['checkpoint_sha256'],different_package_files=changed,
        array_comparisons=fields,CPU_only=True,USED_jobs=4,query_truth_or_quality_read=False,C13_read=False,scoring_entries=0)
    write(out,result)
    print('CPU FEATURE CHECKPOINT/CANONICAL/SOURCES',result['checkpoint_exact'],result['canonical_strings_exact'],changed)
    print('CPU DIFFERENT ARRAYS',{k:v for k,v in fields.items() if not v.get('exact')})
    print('Exact arrays',sum(v.get('exact',False) for v in fields.values()),'/',len(fields))


if __name__=='__main__':main()
