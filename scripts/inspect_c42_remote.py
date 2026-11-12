"""SDK-environment source inspection for the numerical production auditor."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from submit_c09_external_once import scientific_sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('trial',choices=['cpu','gpu']);a=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiGetKernelRequest
    slot='p20-c42-cpu-alignment' if a.trial=='cpu' else 'p21-c42-gpu-alignment'
    folder=ROOT/'kaggle/forks'/slot;pin=json.loads((folder/'production-pin.json').read_text())
    assert hashlib.sha256((folder/'notebook.ipynb').read_bytes()).hexdigest()==pin['notebook_sha256']
    api=KaggleApi();api.authenticate();kernel='clarkkitchen/casmi26-'+slot
    state=json.loads(str(api.kernels_status(kernel)));assert state['status']=='COMPLETE',state
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name='clarkkitchen';request.kernel_slug=kernel.split('/')[1]
        remote=json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    m=remote['metadata'];assert m['currentVersionNumber']==1 and m['isPrivate'] and not m['enableInternet']
    source=scientific_sha(json.loads(remote['blob']['source']));assert source==pin['scientific_sha256']
    out=ROOT/'outputs/forks'/(slot+'-v1');out.mkdir(parents=True,exist_ok=True)
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),kernel=kernel,version=1,trial=a.trial,
        private=True,internet=False,remote_source_exact=True,scientific_sha256=source,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'remote-source-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('C42 REMOTE SOURCE EXACT',a.trial,source,flush=True)


if __name__=='__main__':main()
