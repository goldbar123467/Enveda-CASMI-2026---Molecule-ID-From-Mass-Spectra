"""Record isolated Linux dependencies and actual RTX2070 CUDA execution."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[n]='2'
import torch
import hashlib,json,platform,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow,scipy,sklearn,lightgbm,numba
from rdkit import rdBase
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'


def main():
    assert platform.system()=='Linux' and platform.python_version_tuple()[:2]==('3','12')
    assert sys.prefix=='/home/imsa/.venvs/enveda-c75'
    assert torch.__version__=='2.11.0+cu128' and rdBase.rdkitVersion=='2026.03.3'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and '2070' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    with torch.inference_mode():
        x=torch.ones((16,16),device='cuda:0');y=x@x
        assert torch.equal(y,torch.full_like(y,16))
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],check=True,capture_output=True,text=True).stdout
    (OUT/'wsl-runtime-pip-freeze.txt').write_text(freeze,encoding='utf8')
    versions=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,numba=numba.__version__,rdkit=rdBase.rdkitVersion)
    value=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),python=platform.python_version(),interpreter=sys.executable,isolated_prefix=sys.prefix,versions=versions,
        actual_GPU=torch.cuda.get_device_name(0),GPU_count=1,VRAM_bytes=torch.cuda.get_device_properties(0).total_memory,
        actual_CUDA_matmul_passed=True,peak_tensor_allocation_bytes=torch.cuda.max_memory_allocated(),
        pip_freeze_sha256=hashlib.sha256((OUT/'wsl-runtime-pip-freeze.txt').read_bytes()).hexdigest(),
        scientific_pipeline_parity_not_yet_established=True,new_source_outcomes_read=False,C13_read=False,scoring_entries=0)
    target=OUT/'wsl-runtime-probe.json';assert not target.exists();target.write_text(json.dumps(value,indent=2),encoding='utf8')
    print('C75 ISOLATED LINUX RTX2070 CUDA PROBE PASS',json.dumps(versions),flush=True)


if __name__=='__main__':main()
