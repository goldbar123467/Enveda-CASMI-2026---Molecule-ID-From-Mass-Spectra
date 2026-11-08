"""Read-only cohort and mass-precision measurements for the C08 cache."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
from pathlib import Path
import argparse,json
from precision_cache import ROOT,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=ROOT/'configs/compatible-training-20261108.json');args=parser.parse_args()
    spec=json.loads(args.plan.read_text());cache=ROOT/spec['cache'];frame=pd.read_parquet(cache/'selected.parquet')
    groups=np.load(ROOT/spec['split_groups']);frame['fold']=groups['folds'][frame.mol_idx.to_numpy()]
    for column in ['ingest_lib','ionization_mode','adduct']:
        report=frame.groupby(['fold',column],dropna=False).agg(spectra=('row','size'),identities=('mol_idx','nunique')).reset_index()
        report.to_csv(cache/f'cohorts-{column}.csv',index=False)
    peaks=np.load(cache/'peaks.npy',mmap_mode='r');lengths=np.load(cache/'lengths.npy',mmap_mode='r')
    rows=np.random.default_rng(20261108).choice(len(frame),min(8192,len(frame)),replace=False)
    x=np.array(peaks[rows]);keep=np.arange(96)[None,:]<lengths[rows,None]
    error=np.abs((x.astype(np.float16).astype(np.float32)-x)[:,:,[0,3]])*1000
    value=error[keep]
    write(cache/'precision-audit.json',dict(spectra=len(rows),peaks=int(keep.sum()),
        normalized_mz_and_loss_fp16_round_trip_error_Th={label:dict(median=float(np.median(value[:,i])),p95=float(np.quantile(value[:,i],.95)),maximum=float(value[:,i].max())) for i,label in enumerate(['mz','neutral_loss'])},
        precise_peaks_sha256=sha(cache/'peaks.npy'),
        interpretation='Measured representation rounding only; no evidence that FP32 storage improves ranking. AMP remains FP16 in both arms.'))
    print('DONE COMPATIBLE INPUT AUDIT',len(frame),flush=True)


if __name__=='__main__':main()
