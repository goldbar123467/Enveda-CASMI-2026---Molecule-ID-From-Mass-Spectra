"""Compare identical channel vectors across local and actual original-image CPUs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import torch
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112/cpu-sort-parity'
    dest=base/'independent-comparison.json';assert not dest.exists()
    pin=read(base/'freeze.json');assert read(base/'cloud-collection-audit.json')['remote_source_exact']
    a=read(base/'local/sort-receipt.json');z=read(base/'cloud/sort-receipt.json')
    assert a['source_sha256']==z['source_sha256']==pin['probe_source_sha256']
    assert a['payload_sha256']==z['payload_sha256']==pin['payload_sha256']
    assert a['numpy']==z['numpy']=='2.0.2'
    aa=np.load(base/'local/sort-arrays.npz');zz=np.load(base/'cloud/sort-arrays.npz');assert set(aa.files)==set(zz.files)
    cases={}
    for key in aa.files:
        x=aa[key];y=zz[key];assert x.shape==y.shape and x.dtype==y.dtype
        cases[key]=dict(exact=bool(np.array_equal(x,y)),different_entries=int((x!=y).sum()),
            maximum_absolute_difference=float(np.abs(x.astype(float)-y.astype(float)).max(initial=0)))
    result=dict(completed_utc=stamp(),source_sha256=sha(__file__),freeze_sha256=sha(base/'freeze.json'),
        actual_cloud_audit_sha256=sha(base/'cloud-collection-audit.json'),
        identical_original_inputs=True,identical_source=True,identical_numpy_version=True,
        comparisons=cases,arrays_exact=sum(v['exact'] for v in cases.values()),arrays_compared=len(cases),
        fixed_USED_queries=12,query_truth_read=False,quality_read=False,C13_read=False,scoring_entries=0)
    write(dest,result)
    print('C75 CROSS-HOST RAW SORT EXACT ARRAYS',result['arrays_exact'],'/',len(cases),flush=True)
    print('C75 CROSS-HOST RAW SORT DIFFERENCES',{k:v for k,v in cases.items() if not v['exact']},flush=True)
    print('LOCAL NUMPY RUNTIME',a['runtime'],flush=True);print('CLOUD NUMPY RUNTIME',z['runtime'],flush=True)


if __name__=='__main__':main()
