"""Private short dual-T4 recovery probe; no data, training, or submission."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    worker="""import json,sys,time,torch
torch.set_num_threads(2)
assert torch.cuda.is_available()
torch.cuda.set_per_process_memory_fraction(.3)
torch.manual_seed(20261109)
start=time.perf_counter()
x=torch.randn(1024,1024,device='cuda',dtype=torch.float16)
for _ in range(5):y=x@x.T
torch.cuda.synchronize();assert torch.isfinite(y).all()
report=dict(logical_device=0,name=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,finite=True,seconds=time.perf_counter()-start)
open(sys.argv[1],'w').write(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
"""
    source="""import os,json,subprocess,sys
from pathlib import Path
import torch
root=Path('/kaggle/working');torch.set_num_threads(2)
devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]
print('ACTUAL ALLOCATION',json.dumps(devices),flush=True)
assert len(devices)==2 and all('T4' in d['name'] for d in devices),'Expected two requested T4 devices'
(root/'worker.py').write_text(WORKER,encoding='utf-8')
children=[]
for d in devices:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d['index']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    children.append(subprocess.Popen([sys.executable,str(root/'worker.py'),str(root/('gpu'+str(d['index'])+'.json'))],env=env))
for child in children:assert child.wait(timeout=180)==0
reports=[json.loads((root/('gpu'+str(d['index'])+'.json')).read_text()) for d in devices]
(root/'t4_recovery_report.json').write_text(json.dumps(dict(devices=devices,workers=reports,finite=True,scoring_submission=False),indent=2))
print('DONE DUAL T4 RECOVERY PROBE',flush=True)
""".replace('WORKER',repr(worker))
    compile(source,'c09_t4_probe','exec')
    out=ROOT/'kaggle/forks/c09-t4-recovery-probe';out.mkdir(parents=True,exist_ok=True)
    (out/'probe.py').write_text(source,encoding='utf-8')
    meta=dict(id='clarkkitchen/casmi26-c09-t4-recovery-probe',title='casmi26 c09 t4 recovery probe',code_file='probe.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (out/'probe-pin.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),training=False,scoring_submission=False),indent=2),encoding='utf-8')
    print('BUILT DUAL T4 RECOVERY PROBE')


if __name__=='__main__':main()
