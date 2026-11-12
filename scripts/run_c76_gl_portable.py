"""Run the pinned original GL CLI with the explicit C76 numerical adapter."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[name]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.set_num_threads(2)
import hashlib,importlib.util,json,sys
from pathlib import Path
from c76_gl_portability import install


def main():
    assert len(sys.argv)>3
    pkg=Path(sys.argv[1]);runner=pkg/'gl_runner.py'
    assert hashlib.sha256(runner.read_bytes()).hexdigest()=='5c4f74313ac54263d05d987c39d012d81ec57c5f35e1095ecd49a5bcb6440fca'
    assert hashlib.sha256((pkg/'ckpt/glacier.pt').read_bytes()).hexdigest()=='9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12'
    spec=importlib.util.spec_from_file_location('c76_pinned_original_gl',runner)
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    original=g.load_model
    def load(package,device,log,ckpt=None):
        if ckpt is not None:
            assert hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()=='9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12'
        model=original(package,device,log,ckpt);cpu=original(package,'cpu',log,ckpt)
        receipt=install(model,cpu)
        Path(str(sys.argv[3])+'.portability.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
        log('C76 EXPLICIT NUMERICAL ADAPTER',json.dumps(receipt));return model
    g.load_model=load
    return g.main()


if __name__=='__main__':sys.exit(main())
