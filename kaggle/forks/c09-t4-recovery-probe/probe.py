import os,json,subprocess,sys
from pathlib import Path
import torch
root=Path('/kaggle/working');torch.set_num_threads(2)
devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]
print('ACTUAL ALLOCATION',json.dumps(devices),flush=True)
assert len(devices)==2 and all('T4' in d['name'] for d in devices),'Expected two requested T4 devices'
(root/'worker.py').write_text("import json,sys,time,torch\ntorch.set_num_threads(2)\nassert torch.cuda.is_available()\ntorch.cuda.set_per_process_memory_fraction(.3)\ntorch.manual_seed(20261109)\nstart=time.perf_counter()\nx=torch.randn(1024,1024,device='cuda',dtype=torch.float16)\nfor _ in range(5):y=x@x.T\ntorch.cuda.synchronize();assert torch.isfinite(y).all()\nreport=dict(logical_device=0,name=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,finite=True,seconds=time.perf_counter()-start)\nopen(sys.argv[1],'w').write(json.dumps(report,indent=2))\nprint(json.dumps(report),flush=True)\n",encoding='utf-8')
children=[]
for d in devices:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d['index']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    children.append(subprocess.Popen([sys.executable,str(root/'worker.py'),str(root/('gpu'+str(d['index'])+'.json'))],env=env))
for child in children:assert child.wait(timeout=180)==0
reports=[json.loads((root/('gpu'+str(d['index'])+'.json')).read_text()) for d in devices]
(root/'t4_recovery_report.json').write_text(json.dumps(dict(devices=devices,workers=reports,finite=True,scoring_submission=False),indent=2))
print('DONE DUAL T4 RECOVERY PROBE',flush=True)
