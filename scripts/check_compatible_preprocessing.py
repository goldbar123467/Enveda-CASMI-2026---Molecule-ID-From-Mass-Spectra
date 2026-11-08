"""CPU-only train/serve stable peak/precision fixture across NumPy environments."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
from pathlib import Path
import argparse,json,hashlib,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'outputs/local/kit'))
import features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--queries',type=Path,default=ROOT/'outputs/publish/c08-validation-20261108/validation-queries.parquet');args=parser.parse_args()
    torch.set_num_threads(2);pa.set_cpu_count(2);rows=[]
    for row in pd.read_parquet(args.queries).itertuples(index=False):
        x,n=features.peaks(row.ms2_mzs,row.ms2_normalized_intensities,row.precursor_mz,96)
        m,a=features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev);rows.append((x,n,m,a))
    x=np.stack([r[0] for r in rows]);meta=np.stack([r[2] for r in rows])
    arrays=dict(precise32=x,legacy16=x.astype(np.float16).astype(np.float32),metadata=meta,
        lengths=np.asarray([r[1] for r in rows],np.int64),adduct=np.asarray([r[3] for r in rows],np.int64))
    np.savez(args.out,**arrays)
    report=dict(numpy=np.__version__,spectra=len(rows),feature_source_sha256=hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest(),
        arrays={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in arrays.items()})
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
