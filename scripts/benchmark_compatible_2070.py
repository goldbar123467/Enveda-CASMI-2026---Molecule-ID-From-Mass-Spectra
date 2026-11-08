"""Measure real RTX2070 training shapes before choosing the next long-run plan."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,hashlib,json,sys,time
from compatible_training import accumulated_step

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/local/kit'))
from model import SpectrumCandidateModel
import features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--logical-batches',type=int,nargs='+',default=[96])
    parser.add_argument('--hard',type=int,nargs='+',default=[16,48]);parser.add_argument('--full-batches',action='store_true')
    parser.add_argument('--receipt',default='benchmark.json');args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.70)
    cfg=json.loads((ROOT/'outputs/local/kit/config.json').read_text());cfg.update(width=320,heads=8,layers=3,embedding=320)
    records=[]
    cols=['ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','ionization_mode','collision_energy_ev']
    for b in pq.ParquetFile(ROOT/'outputs/competition-raw/train.parquet').iter_batches(batch_size=1024,columns=cols):
        for row in b.to_pandas().itertuples(index=False):
            try:
                x,n=features.peaks(row.ms2_mzs,row.ms2_normalized_intensities,row.precursor_mz,96)
                if n<3:continue
                meta,adduct=features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev)
                records.append((x,n,meta,adduct))
            except (ValueError,TypeError):continue
            if len(records)==96:break
        if len(records)==96:break
    assert len(records)==96
    original=(np.stack([r[0] for r in records]),np.array([r[1] for r in records]),np.stack([r[2] for r in records]),np.array([r[3] for r in records]))
    fps=np.load(ROOT/'outputs/local/n0/cache/fingerprints.npy',mmap_mode='r');rng=np.random.default_rng(20261108)
    out=ROOT/'outputs/local/compatible-20261108';out.mkdir(exist_ok=True);results=[]
    for logical in args.logical_batches:
      batch=tuple(torch.tensor(value[np.arange(logical)%96],device='cuda') for value in original)
      positive=torch.tensor(np.array(fps[rng.choice(len(fps),logical,replace=False)]),device='cuda',dtype=torch.float32)
      for hard in args.hard:
        negative=torch.tensor(np.array(fps[rng.choice(len(fps),logical*hard,replace=False)]),device='cuda',dtype=torch.float32).reshape(logical,hard,-1)
        for micro in ([logical] if args.full_batches else [n for n in [16,24,32,48,96] if n<=logical]):
            model=optimizer=scaler=None
            try:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(20261108)
                model=SpectrumCandidateModel(cfg).cuda().train()
                optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01,fused=True)
                scaler=torch.amp.GradScaler('cuda')
                for _ in range(5):accumulated_step(model,optimizer,scaler,batch,positive,negative,micro,cfg)
                torch.cuda.synchronize();start=time.perf_counter();skipped=0
                for _ in range(12):skipped+=int(accumulated_step(model,optimizer,scaler,batch,positive,negative,micro,cfg)['amp_skipped'])
                torch.cuda.synchronize();seconds=(time.perf_counter()-start)/12
                assert skipped==0,'AMP skipped a measured optimizer update'
                result=dict(hard_negatives=hard,micro_batch=micro,effective_batch=logical,seconds_per_optimizer_step=seconds,
                    spectra_per_second=logical/seconds,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,amp_skipped_steps=skipped)
            except torch.cuda.OutOfMemoryError:
                result=dict(hard_negatives=hard,micro_batch=micro,effective_batch=logical,status='oom_within70percent_cap')
            results.append(result);print('BENCH',json.dumps(result),flush=True)
            model=optimizer=scaler=None;gc.collect();torch.cuda.empty_cache()
    successful=[r for r in results if 'seconds_per_optimizer_step' in r]
    choices={str(h):max([r for r in successful if r['hard_negatives']==h and r['gpu_peak_reserved_gib']<=5.4],key=lambda r:r['spectra_per_second']) for h in args.hard}
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,architecture=cfg,
                 real_spectra=96,results=results,recommendations=choices,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                    ['scripts/compatible_training.py','scripts/benchmark_compatible_2070.py','outputs/local/kit/model.py','outputs/local/kit/features.py']},
                 limitation='Training throughput/resource benchmark with real spectra and randomized candidate identities; not a quality experiment')
    (out/args.receipt).write_text(json.dumps(receipt,indent=2));print('DONE COMPATIBLE BENCHMARK',json.dumps(choices),flush=True)


if __name__=='__main__':main()
