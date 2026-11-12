"""Run a private metadata-only CPU probe in C57's actual immutable image.

No datasets, competition inputs, labels, GPU allocation or scoring submission.
One durable intent, no blind retry. Used to identify exact parent dependencies.
"""
import json
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    auth=read(ROOT/'.local/c38-active-goal.json')
    assert auth['active'] and auth['submissions_used']==0 and auth['free_local_and_kaggle_compute_authorized']
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'cloud-runtime-metadata';out.mkdir(exist_ok=True)
    intent=out/'push-intent.json';assert not intent.exists(),'Reconcile original metadata probe; no duplicate push'
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    api=KaggleApi();api.authenticate()
    quota=json.loads(api.quota_view().to_json());write(out/'actual-quota.json',dict(checked_utc=stamp(),quota=quota))
    req=ApiGetKernelRequest();req.user_name='clarkkitchen';req.kernel_slug='casmi26-c57-full-masked-validation'
    with api.build_kaggle_client() as client:
        original=json.loads(client.kernels.kernels_api_client.get_kernel(req).to_json())['metadata']
    image=original['dockerImage']
    assert original['currentVersionNumber']==1 and image=='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
    folder=ROOT/'kaggle/forks/c75-original-runtime-metadata';assert not folder.exists();folder.mkdir()
    source='''import importlib.metadata as m,json,platform,sys
from pathlib import Path
packages=['torch','numpy','pandas','scipy','scikit-learn','lightgbm','numba','llvmlite','pyarrow','rdkit','nvidia-cudnn-cu12','nvidia-cublas-cu12','triton']
versions={}
for name in packages:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
result=dict(python=platform.python_version(),platform=platform.platform(),versions=versions,GPU_requested=False,query_data_mounted=False,quality_read=False,C13_read=False,scoring_entries=0)
Path('/kaggle/working/runtime-packages.json').write_text(json.dumps(result,indent=2))
print('C75 ORIGINAL IMAGE RUNTIME METADATA',json.dumps(result),flush=True)
'''
    (folder/'runtime_probe.py').write_text(source,encoding='utf8',newline='\n')
    metadata=dict(id='clarkkitchen/casmi26-c75-original-runtime-metadata',title='casmi26 c75 original runtime metadata',
        code_file='runtime_probe.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=[],competition_sources=[],kernel_sources=[],
        docker_image=image,docker_image_pinning_type='original')
    write(folder/'kernel-metadata.json',metadata)
    write(out/'original-image-metadata.json',dict(checked_utc=stamp(),kernel=original['ref'],version=1,docker_image=image,machine_shape=original['machineShape']))
    write(intent,dict(created_utc=stamp(),source_sha256=sha(__file__),kernel=metadata['id'],expected_version=1,
        code_sha256=sha(folder/'runtime_probe.py'),metadata_sha256=sha(folder/'kernel-metadata.json'),
        image=image,private=True,CPU_only=True,GPU_requested=False,competition_inputs=False,scoring_entries=0))
    response=api.kernels_push(str(folder),timeout='600')
    payload=json.loads(response.to_json())
    write(out/'push-response.json',dict(received_utc=stamp(),response=payload))
    result=dict(received_utc=stamp(),version=getattr(response,'version_number',None),error=getattr(response,'error',None),url=getattr(response,'url',None),kernel=metadata['id'],scoring_entries=0)
    write(out/'push-result.json',result)
    assert result['version']==1 and not result['error'],result
    print('C75 PRIVATE CPU METADATA PROBE ACCEPTED',json.dumps(result),flush=True)
    print('ACTUAL ACCELERATOR QUOTA',json.dumps(quota),flush=True)


if __name__=='__main__':main()
