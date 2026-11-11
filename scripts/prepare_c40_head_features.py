"""Aggregate existing frozen features under new C39 fit exclusions, no scoring."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
from precision_cache import ROOT,sha,write
from training_monitor import system_memory

CONFIG=ROOT/'configs/c40-crossmodal-heads-20261111.json'


def main():
    import argparse
    global CONFIG
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=str,default=str(CONFIG));args=parser.parse_args()
    CONFIG=__import__('pathlib').Path(args.plan)
    pa.set_cpu_count(2);assert system_memory()['free_gib']>=4
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    cfg=json.loads(CONFIG.read_text());out=ROOT/cfg['output'];cache=ROOT/cfg['cache']
    assert not (out/'head-feature-freeze.json').exists()
    boundary=json.loads((out/'fit-boundary-freeze.json').read_text())
    for p,h in boundary['files_sha256'].items():assert sha(out/p)==h,p
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),config_sha256=sha(CONFIG),
        fit_boundary_sha256=sha(out/'fit-boundary-freeze.json'),query_encoding_sha256=sha(out/'query-encoding-freeze.json'),
        encoder_manifest_sha256=sha(ROOT/cfg['frozen_encoders']/'manifest.json'),
        cached_manifest_sha256=sha(cache/'manifest.json'),cached_preparation_sha256=sha(cache/'prepared.json'),
        no_old_head_weights_loaded=True,no_quality_metrics_read=True,C13_read=False)
    write(out/'head-feature-protocol.json',protocol)
    ids=np.load(out/'fitting-identities.npy',allow_pickle=False)
    rows=np.load(out/'fitting-spectrum-rows.npy',allow_pickle=False)
    selected=pd.read_parquet(cache/'selected.parquet',columns=['mol_idx','valid'])
    order=np.argsort(selected.mol_idx.to_numpy()[rows],kind='stable');rows=rows[order]
    row_ids=selected.mol_idx.to_numpy()[rows]
    counts=np.bincount(row_ids,minlength=273833)[ids]
    assert np.all(counts>0) and np.all(np.diff(ids)>0)
    offsets=np.r_[0,counts.cumsum()];assert offsets[-1]==len(rows)
    x=np.load(cache/'spectrum_embeddings.npy',mmap_mode='r');meta=np.load(cache/'metadata.npy',mmap_mode='r');ad=np.load(cache/'adduct.npy',mmap_mode='r')
    target=np.load(cache/'molecule_embeddings.npy',mmap_mode='r')
    assert x.shape==(len(selected),1024) and target.shape==(273833,768)
    dest=np.lib.format.open_memmap(out/'fit-features.npy',mode='w+',dtype=np.float32,shape=(len(ids),1039))
    targets=np.lib.format.open_memmap(out/'fit-targets.npy',mode='w+',dtype=np.float32,shape=(len(ids),768))
    for start in range(0,len(ids),1024):
        stop=min(len(ids),start+1024);lo,hi=int(offsets[start]),int(offsets[stop]);r=rows[lo:hi]
        assert np.array_equal(np.unique(selected.mol_idx.to_numpy()[r]),ids[start:stop])
        position=offsets[start:stop]-lo;div=counts[start:stop,None]
        dest[start:stop,:1024]=np.add.reduceat(np.asarray(x[r],np.float32),position,axis=0)/div
        dest[start:stop,1024:1028]=np.add.reduceat(np.asarray(meta[r],np.float32),position,axis=0)/div
        adduct=np.asarray(ad[r],np.int64);assert np.all((adduct>=0)&(adduct<11))
        dest[start:stop,1028:]=np.add.reduceat(np.eye(11,dtype=np.float32)[adduct],position,axis=0)/div
        targets[start:stop]=np.asarray(target[ids[start:stop]],np.float32)
        assert np.isfinite(dest[start:stop]).all() and np.isfinite(targets[start:stop]).all()
        if start%16384==0:print('C40 AGGREGATED FIT IDENTITIES',stop,'of',len(ids),flush=True)
    dest.flush();targets.flush()
    mean=np.asarray(dest,dtype=np.float64).mean(0);std=np.maximum(np.asarray(dest,dtype=np.float64).std(0),1e-6)
    center=np.asarray(targets,dtype=np.float64).mean(0)
    np.savez(out/'fit-scaling.npz',input_mean=mean.astype(np.float32),input_std=std.astype(np.float32),molecule_center=center.astype(np.float32))
    for start in range(0,len(ids),4096):
        stop=min(len(ids),start+4096)
        dest[start:stop]=np.clip((dest[start:stop]-mean)/std,-5,5)
        y=np.asarray(targets[start:stop],np.float32)-center
        norm=np.linalg.norm(y,axis=1,keepdims=True);assert np.all(norm>1e-6)
        targets[start:stop]=y/norm
    dest.flush();targets.flush()
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/'head-feature-protocol.json'),
        fitting_identities=len(ids),acquisitions=len(rows),input_dimension=1039,target_dimension=768,
        files_sha256={n:sha(out/n) for n in ['fit-features.npy','fit-targets.npy','fit-scaling.npz']},
        stats_fitted_on_training_only=True,equal_identity_weights=True,quality_metrics_read=False,C13_read=False,
        full_pipeline_validation_still_required=True)
    write(out/'head-feature-freeze.json',result);print('DONE C40 HEAD FEATURE PREPARATION',json.dumps(result),flush=True)


if __name__=='__main__':main()
