"""Fit one frozen CPU ridge alignment; no validation or scoring submission."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,time
from datetime import datetime,timezone
import numpy as np
from precision_cache import ROOT,sha,write
from crossmodal_heads import fit_ridge
from training_monitor import system_memory


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=str,default='configs/c40-crossmodal-heads-20261111.json');args=parser.parse_args()
    config=ROOT/args.plan;cfg=json.loads(config.read_text());out=ROOT/cfg['output']
    directory=out/'cpu-ridge';directory.mkdir(exist_ok=True)
    assert not (directory/'complete.json').exists()
    frozen=json.loads((out/'head-feature-freeze.json').read_text())
    for n,h in frozen['files_sha256'].items():assert sha(out/n)==h,n
    contract=dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),helper_sha256=sha(ROOT/'scripts/crossmodal_heads.py'),
        config_sha256=sha(config),feature_freeze_sha256=sha(out/'head-feature-freeze.json'),kind='cpu_ridge_alignment',validation_metrics_read=False)
    write(directory/'source-contract.json',contract)
    def report(**kw):
        assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active'],'Goal stopped'
        assert system_memory()['free_gib']>=4,'RAM reserve reached'
        row=dict(updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw)
        write(directory/'status.json',row)
        if kw.get('completed',0)%16384<cfg['cpu']['block_size']:print('C40 CPU',json.dumps(row),flush=True)
    x=np.load(out/'fit-features.npy',mmap_mode='r');y=np.load(out/'fit-targets.npy',mmap_mode='r')
    start=time.perf_counter();weights=fit_ridge(x,y,cfg['cpu']['ridge_l2'],cfg['cpu']['block_size'],report)
    np.savez(directory/'model.npz',weights=weights.astype(np.float32),ridge_l2=cfg['cpu']['ridge_l2'])
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),model_sha256=sha(directory/'model.npz'),
        source_contract_sha256=sha(directory/'source-contract.json'),fitting_identities=len(x),seconds=time.perf_counter()-start,
        feature_freeze_sha256=sha(out/'head-feature-freeze.json'),validation_metrics_read=False,C13_read=False,
        head_fitted_on_cpu=True,qualified_for_submission=False)
    write(directory/'complete.json',result);report(state='complete',stage='CPU alignment fitted')
    print('DONE C40 CPU TRAINING',json.dumps(result),flush=True)


if __name__=='__main__':main()
