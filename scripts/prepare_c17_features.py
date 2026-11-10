"""Label-blind C17 generator features; seal before constructing training targets."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import rdBase
from precision_cache import ROOT,sha,write,features

OUT=ROOT/'outputs/local/c17-generator-fusion-20261110'
POOL=ROOT/'outputs/local/c12-deep-candidates-20261110/all'
GEN=ROOT/'outputs/forks/c10-candidate-pools-v1'
SCORES=ROOT/'outputs/local/c16-learned-pool-20261110'
CONFIG=ROOT/'configs/c17-generator-fusion-20261110.json'


def stamp():return datetime.now(timezone.utc).isoformat()


def relative(values):
    """No labels or fitted normalization. Missing/constant scores have no rank signal."""
    values=np.asarray(values,float);valid=np.isfinite(values)
    present=valid.astype(float);z=np.zeros(len(values));rank=np.zeros(len(values))
    if valid.sum()>1 and np.ptp(values[valid])>1e-8:
        v=values[valid];z[valid]=np.clip((v-v.mean())/max(v.std(),1e-8),-3,3)/3
        rank[valid]=1/pd.Series(v).rank(method='min',ascending=False).to_numpy()
    return present,z,rank


def main():
    pa.set_cpu_count(2);assert rdBase.rdkitVersion=='2026.03.3'
    OUT.mkdir(exist_ok=True);assert not (OUT/'feature-freeze.json').exists()
    assert not (OUT/'protocol.json').exists(),'Inspect interrupted preparation before any recovery'
    prep=json.loads((POOL/'preparation.json').read_text())
    for name,pin in prep['files'].items():assert sha(POOL/name)==pin,name
    files=[POOL/'pool.parquet',POOL/'parent.csv',POOL/'fingerprint-smiles.parquet',POOL/'scores.parquet',
        SCORES/'cosine-control-scores.parquet',SCORES/'C15NP-scores.parquet']
    names=['eng_lists.json','pc_lists.json','ice_work/ice_out.json','gl_work/gl_out.json','n1_work/n1_out.json']
    files += [GEN/n for n in names]
    write(OUT/'protocol.json',dict(prepared_utc=stamp(),config_sha256=sha(CONFIG),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in ['prepare_c17_features.py','train_c17_fusion.py','expected_mrr_expansion.py']},
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        labels_read=False,fresh_C13_prediction_metrics_read=False,automatic_submission=False,
        prior_preparation_protocol_sha256=sha(OUT/'protocol-v1.json') if (OUT/'protocol-v1.json').exists() else None,
        rules='Official identity joins; score alias=max, rank alias=min; missing/null0pluspresence; constant channel has zero rank/z signal. Query-only normalization, no learned preprocessing. N1 only one spectrum supported here.'))
    pool=pd.read_parquet(POOL/'pool.parquet').sort_values(['molecule_id','prior_rank'],kind='stable').reset_index(drop=True)
    index=pd.read_parquet(POOL/'fingerprint-smiles.parquet');lookup=dict(zip(index.smiles,index.metric_key));invalid=set();computed=0
    def key(s):
        nonlocal computed
        if s not in lookup:
            try:lookup[s]=features.structure(s)[0];computed+=1
            except (ValueError,TypeError,RuntimeError):lookup[s]=None;invalid.add(s)
        return lookup[s]
    data={n:json.loads((GEN/n).read_text(encoding='utf-8')) for n in names}
    maps={}
    for channel,name,field in [('engine2',names[0],'smiles'),('pubchem',names[1],'pc')]:
        maps[channel]={}
        for mid,item in data[name].items():
            values={}
            for rank,s in enumerate(item[field],1):
                k=key(s)
                if k is not None:values[k]=min(values.get(k,rank),rank)
            maps[channel][mid]=values
    for channel,name in [('ice',names[2]),('gl',names[3]),('n1',names[4])]:
        maps[channel]={}
        for mid,item in data[name].items():
            if channel=='n1':
                assert len(item['spec'])==1,'Multi-spectrum aggregation requires a separate source audit'
                spec=next(iter(item['spec'].values()));assert spec['nb']>0
                assert len(item['cands'])==len(spec['mean']);items=zip(item['cands'],spec['mean'])
            else:items=item.items()
            values={}
            for s,v in items:
                if v is None:continue
                assert np.isfinite(v);k=key(s)
                if k is not None:values[k]=max(values.get(k,-np.inf),float(v))
            maps[channel][mid]=values
    for channel in maps:
        pool[channel]=[maps[channel].get(mid,{}).get(k,np.nan) for mid,k in zip(pool.molecule_id,pool.metric_key)]
    for channel,path,column in [('p01pair',POOL/'scores.parquet','neural_score'),
            ('c09',SCORES/'cosine-control-scores.parquet','c08_score'),('c15',SCORES/'C15NP-scores.parquet','c08_score')]:
        frame=pd.read_parquet(path).set_index(['molecule_id','smiles'])
        pool[channel]=frame.loc[pd.MultiIndex.from_frame(pool[['molecule_id','smiles']]),column].to_numpy()
        assert np.isfinite(pool[channel]).all()
    feature_names=['parent','parent_rank1','inverse_parent_rank','inverse_prior_rank','library','library_parent','inverse_engine2','inverse_pubchem']
    for c in ['ice','gl','n1','p01pair','c09','c15']:feature_names += [c+'_present',c+'_z',c+'_inverse_rank']
    feature_names+=['log_pool_size','generator_count']
    mids=[];blocks=[];folds=[]
    import hashlib
    cfg=json.loads(CONFIG.read_text())
    for mid,g in pool.groupby('molecule_id',sort=False):
        n=len(g);assert n<=128 and g.metric_key.is_unique
        parent=g.is_parent.to_numpy(bool);r=g.prior_rank.to_numpy(float);lib=g.library_max.to_numpy(float)
        assert np.isfinite(lib).all() and len(set(lib))==1
        columns=[parent.astype(float),(r==1).astype(float),parent/r,1/r,lib,lib*parent]
        for c in ['engine2','pubchem']:columns.append(np.nan_to_num(1/g[c].to_numpy(float),nan=0.))
        for c in ['ice','gl','n1','p01pair','c09','c15']:columns.extend(relative(g[c].to_numpy(float)))
        columns.extend([np.full(n,np.log(n)/np.log(128)),g[['engine2','pubchem','ice','gl','n1']].notna().sum(axis=1).to_numpy()/5])
        x=np.stack(columns,axis=1).astype(np.float32);assert x.shape==(n,len(feature_names)) and np.isfinite(x).all()
        mids.append(mid);blocks.append(x);folds.append(int(hashlib.sha256((cfg['split_salt']+mid).encode()).hexdigest(),16)%cfg['folds'])
    assert len(mids)==160
    matrix=np.zeros((160,128,len(feature_names)),np.float32);mask=np.zeros((160,128),bool)
    for i,b in enumerate(blocks):matrix[i,:len(b)]=b;mask[i,:len(b)]=True
    np.savez(OUT/'features.npz',features=matrix,mask=mask,folds=np.asarray(folds,dtype=np.int64),molecule_ids=np.asarray(mids))
    pool.to_parquet(OUT/'evidence.parquet',index=False)
    write(OUT/'feature-freeze.json',dict(completed_utc=stamp(),protocol_sha256=sha(OUT/'protocol.json'),
        features_sha256=sha(OUT/'features.npz'),evidence_sha256=sha(OUT/'evidence.parquet'),feature_names=feature_names,
        queries=160,rows=len(pool),shape=list(matrix.shape),fold_sizes={str(i):folds.count(i) for i in range(cfg['folds'])},
        extra_official_identities_computed=computed,invalid_generator_structures=len(invalid),
        channel_rows={c:int(pool[c].notna().sum()) for c in maps},
        labels_read=False,fresh_C13_prediction_metrics_read=False,automatic_submission=False))
    print('C17 FEATURES FROZEN',sha(OUT/'feature-freeze.json'),len(feature_names),'FEATURES',flush=True)


if __name__=='__main__':main()
