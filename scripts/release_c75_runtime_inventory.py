"""Private CPU-only full package inventory in the exact original image."""
import json
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    assert read(ROOT/'.local/c38-active-goal.json')['active']
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=base/'cloud-runtime-inventory';assert not out.exists();out.mkdir()
    image=read(base/'cloud-runtime-metadata/actual-status.json')['actual_docker_image']
    folder=ROOT/'kaggle/forks/c75-original-runtime-inventory';assert not folder.exists();folder.mkdir()
    source='''import importlib.metadata as m,json,platform
from pathlib import Path
versions={d.metadata['Name']:d.version for d in m.distributions() if d.metadata.get('Name')}
result=dict(python=platform.python_version(),versions=versions,GPU_requested=False,query_data_mounted=False,quality_read=False,C13_read=False,scoring_entries=0)
Path('/kaggle/working/runtime-inventory.json').write_text(json.dumps(result,indent=2,sort_keys=True))
print('C75 CPU ORIGINAL IMAGE INVENTORY',len(versions),'packages',flush=True)
'''
    (folder/'runtime_inventory.py').write_text(source,encoding='utf8',newline='\n')
    metadata=dict(id='clarkkitchen/casmi26-c75-original-runtime-inventory',title='casmi26 c75 original runtime inventory',
        code_file='runtime_inventory.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=False,dataset_sources=[],competition_sources=[],kernel_sources=[],
        docker_image=image,docker_image_pinning_type='original')
    write(folder/'kernel-metadata.json',metadata)
    write(out/'push-intent.json',dict(created_utc=stamp(),source_sha256=sha(__file__),kernel=metadata['id'],expected_version=1,
        code_sha256=sha(folder/'runtime_inventory.py'),metadata_sha256=sha(folder/'kernel-metadata.json'),
        image=image,private=True,CPU_only=True,no_data_attached=True,scoring_entries=0))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();response=api.kernels_push(str(folder),timeout='600')
    write(out/'push-response.json',dict(received_utc=stamp(),response=json.loads(response.to_json())))
    result=dict(received_utc=stamp(),kernel=metadata['id'],version=getattr(response,'version_number',None),error=getattr(response,'error',None),url=getattr(response,'url',None),scoring_entries=0)
    write(out/'push-result.json',result);assert result['version']==1 and not result['error'],result
    print('C75 ORIGINAL IMAGE CPU INVENTORY ACCEPTED',json.dumps(result),flush=True)


if __name__=='__main__':main()
