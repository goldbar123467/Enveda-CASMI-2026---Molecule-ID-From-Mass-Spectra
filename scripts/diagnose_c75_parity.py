"""Locate engineering parity divergence using USED predictions only; no truth."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'2')
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    port=base/'fixture-port';working=Path(read(port/'run-complete.json')['working'])
    cloud=ROOT/'outputs/forks/c57-full-masked-validation-v1'
    dest=port/'parity-divergence-diagnosis.json';assert not dest.exists()
    from c74_complete_target import module
    identity=module('c75_diagnosis_identity','c09e_features.py').structure
    from functools import lru_cache
    key=lru_cache(maxsize=10000)(lambda s:identity(s)[0])
    lists={}
    for name in ['c41_p01.csv','c41_p14.csv']:
        a=pd.read_csv(working/name,dtype=str).set_index('molecule_id')
        b=pd.read_csv(cloud/name,dtype=str).set_index('molecule_id').loc[a.index]
        av=[x.split(';') for x in a.smiles];bv=[x.split(';') for x in b.smiles]
        ak=[[key(s) for s in row] for row in av];bk=[[key(s) for s in row] for row in bv]
        lists[name]=dict(queries=len(a),exact_strings=sum(x==y for x,y in zip(av,bv)),
            exact_official_key_order=sum(x==y for x,y in zip(ak,bk)),
            identical_official_key_membership=sum(set(x)==set(y) for x,y in zip(ak,bk)),
            identical_leader=sum(x[0]==y[0] for x,y in zip(ak,bk)),
            changed_slots=sum(sum(x!=y for x,y in zip(u,v)) for u,v in zip(ak,bk)))
    frames={}
    for name in ['c41_neural_p01.parquet','neural_public.parquet','neural_ours.parquet']:
        a=pd.read_parquet(working/name);ids=set(a.molecule_id)
        b=pd.read_parquet(cloud/name);b=b[b.molecule_id.isin(ids)]
        ix=['molecule_id','metric_key'];a=a.set_index(ix);b=b.set_index(ix)
        assert a.index.is_unique and b.index.is_unique
        common=a.index.intersection(b.index);columns={}
        for col in sorted(set(a)&set(b)):
            u=a.loc[common,col];v=b.loc[common,col]
            if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
                x=u.to_numpy(float);y=v.to_numpy(float);finite=np.isfinite(x)&np.isfinite(y)
                delta=np.abs(x[finite]-y[finite])
                columns[col]=dict(equal_including_nan=int(((x==y)|(np.isnan(x)&np.isnan(y))).sum()),
                    nan_mismatches=int((np.isnan(x)!=np.isnan(y)).sum()),
                    maximum_absolute_difference=float(delta.max(initial=0)),
                    outside_frozen_tolerance=int((~np.isclose(x,y,atol=1e-5,rtol=1e-5,equal_nan=True)).sum()))
            else:columns[col]=dict(exact=int(u.fillna('MISSING').eq(v.fillna('MISSING')).sum()))
        frames[name]=dict(local_rows=len(a),cloud_rows=len(b),common_candidates=len(common),
            local_only_candidates=len(a.index.difference(b.index)),cloud_only_candidates=len(b.index.difference(a.index)),columns=columns)
    write(dest,dict(completed_utc=stamp(),source_sha256=sha(__file__),
        run_complete_sha256=sha(port/'run-complete.json'),lists=lists,features=frames,
        USED_only=True,query_truth_or_quality_read=False,new_source_quality_read=False,C13_read=False,
        scoring_entries=0,full_fresh_inference_held=True,parity_passed=False))
    print('C75 USED ENGINEERING LIST COMPARISON',lists)
    for name,data in frames.items():
        print(name,{k:v for k,v in data.items() if k!='columns'})
        print('DIFFERENT FIELDS',{c:v for c,v in data['columns'].items() if v.get('outside_frozen_tolerance',0) or v.get('exact',len(common))<len(common)})


if __name__=='__main__':main()
