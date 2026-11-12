"""Isolate Torch2.10 using the exact prior USED cloud-nine GL input bytes."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['PYTHONHASHSEED']='0';os.environ['CUDA_VISIBLE_DEVICES']='0'
import subprocess,sys
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    assert sys.prefix=='/home/imsa/.venvs/enveda-c75-torch210'
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'gl-torch210-probe-v1';assert not out.exists();out.mkdir()
    source=base/'gl-engineering-probe-v1/original-cloud-nine-input.json'
    previous=read(base/'gl-engineering-probe-v1/intent.json')
    assert sha(source)==previous['cloud_nine_input_sha256']
    assert read(base/'gl-engineering-probe-v1/results.json')['USED_only']
    package=base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-glacier'
    runner=package/'gl_runner.py';assert sha(runner)==previous['runner_sha256']
    assert 'enveda-c75-gl-probe-torch210' in __import__('pathlib').Path('/proc/self/cgroup').read_text()
    write(out/'intent.json',dict(started_utc=stamp(),pid=os.getpid(),source_sha256=sha(__file__),
        input_sha256=sha(source),runner_sha256=sha(runner),expected_torch='2.10.0+cu128',
        new_runtime_only=True,original_Torch211_environment_unchanged=True,
        USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    dest=out/'cloud-nine.json'
    command=[sys.executable,str(runner),str(package),str(source),str(dest),
        '--device','cuda','--budget','4000.0','--site','/home/imsa/.local/share/enveda-c75/fixture/working/ice_site',
        '--wheels',str(base/'input/casmi26-attributed-v4g-reproduction-assets/casmi26-iceberg/wheels'),'--any-rdkit']
    with (out/'replay.log').open('w',encoding='utf8') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
    assert result.returncode==0
    meta=read(str(dest)+'.meta.json')
    assert meta['torch']=='2.10.0+cu128' and meta['rdkit']=='2025.03.6' and meta['device']=='cuda'
    assert meta['status']=='ok' and meta['n_mols_scored']==meta['n_mols_covered']==9
    assert meta['n_pred_fail']==meta['n_feat_fail']==0
    import torch
    import numpy as np
    import pandas as pd
    assert torch.__version__=='2.10.0+cu128' and str(torch.__file__).startswith(sys.prefix)
    values=read(dest)
    cloud=pd.read_parquet(ROOT/'outputs/forks/c57-full-masked-validation-v1/c41_neural_p01.parquet')
    pairs=[(values[r.molecule_id].get(r.smiles),r.gl) for r in cloud.itertuples(index=False)
           if r.molecule_id in values and np.isfinite(r.gl) and values[r.molecule_id].get(r.smiles) is not None]
    a,b=np.asarray(pairs,float).T
    comparison=dict(common_finite_scores=len(a),maximum_absolute_difference=float(np.abs(a-b).max(initial=0)),
        outside_frozen_tolerance=int((~np.isclose(a,b,atol=1e-5,rtol=1e-5)).sum()))
    write(out/'results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),
        intent_sha256=sha(out/'intent.json'),comparison_to_original_T4=comparison,
        Torch211_comparison=read(base/'gl-engineering-probe-v1/results.json')['results']['original_cloud_candidate_input_vs_T4'],
        runtime=torch.__version__,metadata_sha256=sha(str(dest)+'.meta.json'),
        USED_only=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 TORCH210 IDENTICAL-INPUT GL DIAGNOSIS',comparison,flush=True)


if __name__=='__main__':main()
