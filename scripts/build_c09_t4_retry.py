"""Retry cancelled C09 clean validation on T4x2, with a real GPU1 P14 lane."""
import hashlib,json
from pathlib import Path
from build_general_anchor import ROOT,guarded_source
from build_c09_controls import unwrap


def main():
    folder=ROOT/'kaggle/forks/c09-clean-validation';path=folder/'notebook.ipynb'
    original=json.loads(path.read_text(encoding='utf-8'))
    assert not (folder/'t4-retry-pin.json').exists(),'Retry already built; do not mutate it while running'
    # Preserve the accepted but cancelled v1 as exact source evidence.
    archive=ROOT/'outputs/research/c09-cancelled-v1';archive.mkdir(parents=True,exist_ok=True)
    for name in ['notebook.ipynb','kernel-metadata.json','clean-pin.json']:
        (archive/name).write_bytes((folder/name).read_bytes())
    worker="""import os,json,hashlib,gc,torch
from pathlib import Path
torch.set_num_threads(2)
hits=list(Path('/kaggle/input').rglob('frozen-manifest.json'));assert len(hits)==1
bundle=hits[0].parent;manifest=json.loads(hits[0].read_text())
for name in ['dreams.ts','molformer.ts']:
    assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==manifest['files'][name]
graph=torch.jit.load(str(bundle/'dreams.ts'),map_location='cuda').eval()
x=torch.zeros((2,101,2),device='cuda');x[:,0,0]=300;x[:,0,1]=1.1;x[:,1,0]=100;x[:,1,1]=1
with torch.inference_mode():assert torch.isfinite(graph(x)).all()
del graph,x;gc.collect();torch.cuda.empty_cache()
graph=torch.jit.load(str(bundle/'molformer.ts'),map_location='cuda').eval()
ids=torch.full((2,202),2,device='cuda',dtype=torch.long);ids[:,0]=0;ids[:,1]=1
mask=torch.zeros_like(ids);mask[:,:2]=1
with torch.inference_mode():assert torch.isfinite(graph(ids,mask)).all()
report=dict(physical_device_requested=os.environ['CUDA_VISIBLE_DEVICES'],name=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),cuda_version=torch.version.cuda,manifest_sha256=hashlib.sha256(hits[0].read_bytes()).hexdigest(),dreams_finite=True,molformer_finite=True,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
Path('/kaggle/working/c09_gpu1_preflight.json').write_text(json.dumps(report,indent=2))
print('P14 GPU1 REAL ENCODER PREFLIGHT',json.dumps(report),flush=True)
"""
    preamble="""import os,json,sys,subprocess,torch
from pathlib import Path
_c09t4_devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]
assert len(_c09t4_devices)==2 and all('T4' in d['name'] for d in _c09t4_devices),'Actual dual-T4 allocation required'
Path('/kaggle/working/c09_t4_allocation.json').write_text(json.dumps(dict(devices=_c09t4_devices,allocation={'GPU0':'complete P01 clean80 pipeline','GPU1':'real P14 encoder preflight and complete final-list negative-control scoring'}),indent=2))
Path('/kaggle/working/c09_gpu1_preflight.py').write_text(WORKER,encoding='utf-8')
_c09t4_log=open('/kaggle/working/c09_gpu1_preflight.log','w')
_c09t4_child=subprocess.Popen([sys.executable,'/kaggle/working/c09_gpu1_preflight.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=_c09t4_log,stderr=subprocess.STDOUT)
os.environ['CUDA_VISIBLE_DEVICES']='0'
print('C09 RETRY ACTUAL DEVICES',json.dumps(_c09t4_devices),flush=True)
""".replace('WORKER',repr(worker))
    needle="run_stage([sys.executable,CODE/'score_frozen_channel.py',_frozen_bundle,PUB_BUNDLE,ROOT/'c09_clean_neural_p01.parquet',ROOT/'effective_query_spectra.parquet',ROOT/'neural_candidate_features.parquet','cuda'],ROOT/'c09_clean_p14.log','C09 clean exact P14 negative control')"
    replacement="assert _c09t4_child.wait(timeout=300)==0,'GPU1 encoder preflight failed; inspect saved log'\n_c09t4_log.close()\nos.environ['CUDA_VISIBLE_DEVICES']='1'\ntry:\n    "+needle+"\nfinally:\n    os.environ['CUDA_VISIBLE_DEVICES']='0'"
    changed=0
    for i,cell in enumerate(original['cells']):
        if cell['cell_type']!='code':continue
        source=unwrap(''.join(cell['source']))
        if needle in source:
            assert source.count(needle)==1;source=source.replace(needle,replacement);changed+=1
        compile(source,'c09_t4_retry','exec')
        cell['source']=guarded_source(source,f'c09_t4_retry_{i}').splitlines(keepends=True);cell['outputs']=[];cell['execution_count']=None
    assert changed==1
    original['cells'].insert(0,dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(preamble,'c09_t4_lane_allocation').splitlines(keepends=True),id='c09-t4-lanes'))
    path.write_text(json.dumps(original,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((folder/'kernel-metadata.json').read_text());meta.update(machine_shape='NvidiaTeslaT4',docker_image_pinning_type='original')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    receipt=dict(campaign='C09',retry_of='cancelled v1; no outputs/GPU quota consumed',parent_v1_sha256=hashlib.sha256((archive/'notebook.ipynb').read_bytes()).hexdigest(),
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gpu0='Unchanged full P01 clean80 validation',gpu1='Pinned P14 real encoder preflight and negative-control inference',
        changes='Device allocation/recovery only; no model, query, coefficient or metric change',scoring_submission=False)
    (folder/'t4-retry-pin.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print('BUILT DUAL-T4 CLEAN RETRY',json.dumps(receipt))


if __name__=='__main__':main()
