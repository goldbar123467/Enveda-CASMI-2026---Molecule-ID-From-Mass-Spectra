"""Freeze a CPU-only raw-order probe from original USED Engine2 return values.

The private executed source embeds unlabeled parent channel probabilities only.
Raw arrays and query identities stay out of public Git evidence. No fitting.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import torch
import base64,io,pickle
import numpy as np
from c75_port_utils import ROOT,read,write,sha,stamp


SOURCE='''import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='2'
import base64,hashlib,io,json,platform,sys
from pathlib import Path
import numpy as np
PAYLOAD = {payload!r}
data=np.load(io.BytesIO(base64.b85decode(PAYLOAD)))
arrays={{}}
def rank(x):
    order=np.argsort(-x);r=np.empty(len(x));r[order]=np.arange(len(x));return r/max(1,len(x)-1)
for i in range(12):
    a=data['pv.'+str(i)];b=data['ours.'+str(i)]
    arrays['pv.order.'+str(i)]=np.argsort(-a)
    arrays['ours.order.'+str(i)]=np.argsort(-b)
    blended=.88*(1-rank(a))+.12*(1-rank(b))+1e-6*a
    arrays['blend.order.'+str(i)]=np.argsort(-blended,kind='mergesort')
    arrays['blend.score.'+str(i)]=blended
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/kaggle/working')
out.mkdir(parents=True,exist_ok=True);assert not (out/'sort-receipt.json').exists()
np.savez_compressed(out/'sort-arrays.npz',**arrays)
capture=io.StringIO()
import contextlib
with contextlib.redirect_stdout(capture):np.show_runtime()
result=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    payload_sha256=hashlib.sha256(base64.b85decode(PAYLOAD)).hexdigest(),numpy=np.__version__,python=platform.python_version(),
    runtime=capture.getvalue(),arrays_sha256=hashlib.sha256((out/'sort-arrays.npz').read_bytes()).hexdigest(),
    fixed_USED_queries=12,no_query_identities_or_labels_embedded=True,training_performed=False,scoring_entries=0)
(out/'sort-receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('C75 CPU SORT PROBE COMPLETE',len(arrays),'arrays')
'''


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112';out=base/'cpu-sort-parity'
    assert not out.exists();out.mkdir()
    assert read(base/'retrieval-runtime-probe-v1/results.json')['comparison']['original_local_runtime']['exact_lists']==12
    snapshot=base/'retrieval-runtime-probe-v1/original-engine-return.pkl'
    with snapshot.open('rb') as stream:recs,scores=pickle.load(stream)
    ids=sorted(scores['pv']);assert len(ids)==12 and set(ids)==set(scores['ours'])
    values={}
    for i,mid in enumerate(ids):
        for channel in ['pv','ours']:values[channel+'.'+str(i)]=scores[channel][mid]
    buf=io.BytesIO();np.savez_compressed(buf,**values);payload=base64.b85encode(buf.getvalue()).decode('ascii')
    folder=ROOT/'outputs/publish/c75-cpu-sort-parity-20261112';assert not folder.exists();folder.mkdir()
    source=folder/'sort_probe.py';source.write_text(SOURCE.format(payload=payload),encoding='utf8',newline='\n')
    compile(source.read_text(),str(source),'exec')
    image=read(base/'cloud-runtime-metadata/actual-status.json')['actual_docker_image']
    meta=dict(id='clarkkitchen/casmi26-c75-cpu-sort-parity',title='casmi26 c75 cpu sort parity',code_file='sort_probe.py',
        language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        dataset_sources=[],competition_sources=[],kernel_sources=[],docker_image=image,docker_image_pinning_type='original')
    write(folder/'kernel-metadata.json',meta)
    write(out/'freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),probe_source_sha256=sha(source),
        metadata_sha256=sha(folder/'kernel-metadata.json'),engine_snapshot_sha256=sha(snapshot),payload_sha256=__import__('hashlib').sha256(buf.getvalue()).hexdigest(),
        fixed_USED_queries=12,channel_arrays=24,query_identities_or_truth_embedded=False,training_performed=False,scoring_entries=0))
    print('C75 PRIVATE CPU SORT PROBE FROZEN',source.stat().st_size,'bytes; no GPU or fitting',flush=True)


if __name__=='__main__':main()
