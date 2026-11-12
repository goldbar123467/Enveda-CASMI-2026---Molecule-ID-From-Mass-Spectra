"""Collect and independently verify the actual private CPU portability probe."""
import hashlib,json
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest,ApiListKernelSessionOutputRequest
    import requests
    base=ROOT/'outputs/local/c76-numerical-portability-20261112/cpu-feature-parity';out=base/'cloud'
    assert not out.exists();pin=read(base/'freeze.json')
    meta=read(ROOT/'kaggle/forks/c76-cpu-feature-parity/kernel-metadata.json')
    api=KaggleApi();api.authenticate();state=json.loads(str(api.kernels_status(meta['id'])))
    assert state['status']=='COMPLETE',state
    with api.build_kaggle_client() as client:
        req=ApiGetKernelRequest();req.user_name='clarkkitchen';req.kernel_slug=meta['id'].split('/')[1]
        remote=client.kernels.kernels_api_client.get_kernel(req);m=remote.metadata
        assert m.current_version_number==1 and m.is_private and not m.enable_gpu and not m.enable_internet
        assert m.docker_image==meta['docker_image']
        assert hashlib.sha256(remote.blob.source.encode()).hexdigest()==pin['probe_source_sha256']
        req=ApiListKernelSessionOutputRequest();req.user_name='clarkkitchen';req.kernel_slug=meta['id'].split('/')[1];req.page_size=100
        wanted={'feature-probe.json','feature-arrays.npz'};found={};pages=0;log=''
        for _ in range(20):
            response=client.kernels.kernels_api_client.list_kernel_session_output(req);pages+=1
            if pages==1:log=response.log
            found.update({f.file_name:f for f in response.files if f.file_name in wanted})
            if set(found)==wanted:break
            assert response.next_page_token,'Required output files absent'
            req.page_token=response.next_page_token
        assert set(found)==wanted,'Bounded output listing ended before required files'
    out.mkdir()
    for name,item in found.items():
        download=requests.get(item.url,timeout=60);download.raise_for_status();(out/name).write_bytes(download.content)
    if log:(out/'run.log').write_text(log,encoding='utf8')
    result=read(out/'feature-probe.json');assert result['source_sha256']==pin['probe_source_sha256']
    assert result['c76_candidate_freeze_sha256']==pin['c76_candidate_freeze_sha256'] and result['c76_helper_sha256']==pin['c76_helper_sha256']
    assert not result['GPU_initialized'] and result['c76_adapter']['weights_exact']
    assert sha(out/'feature-arrays.npz')==result['arrays_sha256']
    write(base/'cloud-collection-audit.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),kernel=meta['id'],version=1,
        remote_source_exact=True,actual_GPU_disabled=True,actual_original_image_exact=True,private=True,
        files_sha256={n:sha(out/n) for n in wanted},listing_pages=pages,
        original_collector_source_sha256=sha(base/'collector-source-before-pagination-fix.py'),
        c76_helper_sha256=pin['c76_helper_sha256'],scoring_entries=0))
    print('C76 CPU CLOUD SOURCE/IMAGE/ADAPTER/OUTPUT VERIFIED',flush=True)


if __name__=='__main__':main()
