"""Verify actual private CPU source/image and collect the completed sort probe."""
import hashlib,json
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest,ApiListKernelSessionOutputRequest
    import requests
    base=ROOT/'outputs/local/c75-local-core-port-20261112/cpu-sort-parity';out=base/'cloud'
    assert not out.exists();pin=read(base/'freeze.json')
    folder=ROOT/'outputs/publish/c75-cpu-sort-parity-20261112';meta=read(folder/'kernel-metadata.json')
    api=KaggleApi();api.authenticate();state=json.loads(str(api.kernels_status(meta['id'])))
    assert state['status']=='COMPLETE',state
    with api.build_kaggle_client() as client:
        req=ApiGetKernelRequest();req.user_name='clarkkitchen';req.kernel_slug=meta['id'].split('/')[1]
        remote=client.kernels.kernels_api_client.get_kernel(req)
        m=remote.metadata;raw=json.loads(remote.to_json())
        assert m.current_version_number==1 and m.is_private and not m.enable_gpu and not m.enable_internet
        assert m.docker_image==meta['docker_image']
        source=remote.blob.source;assert hashlib.sha256(source.encode()).hexdigest()==pin['probe_source_sha256']
        req=ApiListKernelSessionOutputRequest();req.user_name='clarkkitchen';req.kernel_slug=meta['id'].split('/')[1];req.page_size=100
        response=client.kernels.kernels_api_client.list_kernel_session_output(req)
    assert not response.next_page_token
    wanted={'sort-receipt.json','sort-arrays.npz'};found={f.file_name:f for f in response.files if f.file_name in wanted};assert set(found)==wanted
    out.mkdir()
    for name,item in found.items():
        download=requests.get(item.url,timeout=60);download.raise_for_status();(out/name).write_bytes(download.content)
    if response.log:(out/'run.log').write_text(response.log,encoding='utf8')
    result=read(out/'sort-receipt.json');assert result['source_sha256']==pin['probe_source_sha256'] and result['payload_sha256']==pin['payload_sha256']
    assert sha(out/'sort-arrays.npz')==result['arrays_sha256']
    write(base/'cloud-collection-audit.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),kernel=meta['id'],version=1,
        remote_source_exact=True,actual_GPU_disabled=True,actual_original_image_exact=True,private=True,
        files_sha256={n:sha(out/n) for n in wanted},scoring_entries=0))
    print('C75 CPU SORT CLOUD SOURCE/IMAGE/OUTPUT VERIFIED',flush=True)


if __name__=='__main__':main()
