"""Publish/push only the frozen private production previews, never score them."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from submit_c09_external_once import scientific_sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)
def stamp():return datetime.now(timezone.utc).isoformat()
def payload(response):
    value=response.to_json() if hasattr(response,'to_json') else str(response)
    try:return json.loads(value)
    except (ValueError,TypeError):return dict(response=str(value))


def verified(trial):
    authorization=read(ROOT/'.local/c38-active-goal.json');assert authorization['active'] and authorization['submissions_used']==0
    assert authorization['free_local_and_kaggle_compute_authorized'] and not authorization['new_paid_compute_authorized']
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    family=read(OUT/'complete-candidate-freeze.json');assert family['policy_sha256']==sha(ROOT/'configs/c56-bounded-semantic-pair-20261112.json')
    assert read(OUT/'used-auxiliary-results.json')['combined_used_development_pair_passed']
    assert read(OUT/'independent-fitted-model-audit.json')['C39_C51_C53_and_global_C13_inventory_fit_identity_overlaps']==0
    assert not (ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json').exists()
    entry=family['candidates'][trial];folder=ROOT/entry['path'];bundle=ROOT/entry['bundle']
    assert sha(folder/'notebook.ipynb')==entry['notebook_sha256'] and sha(folder/'production-pin.json')==entry['pin_sha256']
    assert scientific_sha(read(folder/'notebook.ipynb'))==entry['scientific_sha256']
    manifest=read(bundle/'head-manifest.json');assert sha(bundle/'head-manifest.json')==entry['head_manifest_sha256']
    for name,pin in manifest['files'].items():assert sha(bundle/name)==pin
    assert set(p.name for p in bundle.iterdir() if p.is_file())==set(manifest['files'])|{'head-manifest.json','dataset-metadata.json'}
    assert all(p.name=='__pycache__' for p in bundle.iterdir() if p.is_dir())
    dataset=read(bundle/'dataset-metadata.json')['id'];meta=read(folder/'kernel-metadata.json')
    assert dataset in meta['dataset_sources'] and meta['id']==entry['kernel'] and meta['is_private'] and not meta['enable_internet']
    assert meta['machine_shape']=='NvidiaTeslaT4'
    return entry,folder,bundle,dataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('trial',choices=['cpu','gpu']);parser.add_argument('phase',choices=['publish','status','push']);a=parser.parse_args()
    import msvcrt
    lock=(OUT/(a.trial+'-private-release.lock')).open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    entry,folder,bundle,dataset=verified(a.trial)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if a.phase=='status':
        result=dict(checked_utc=stamp(),trial=a.trial,dataset=dataset,dataset_status=str(api.dataset_status(dataset)))
        if (OUT/(a.trial+'-private-push-result.json')).exists():result['kernel_status']=json.loads(str(api.kernels_status(entry['kernel'])))
        write(OUT/(a.trial+'-private-actual-status.json'),result);print('C56 PRIVATE ACTUAL STATUS',json.dumps(result));return
    if a.phase=='publish':
        intent=OUT/'model-dataset-upload-intent.json';assert not intent.exists(),'Reconcile existing exact upload; never duplicate it'
        write(intent,dict(created_utc=stamp(),dataset=dataset,private=True,head_manifest_sha256=entry['head_manifest_sha256'],source_sha256=sha(__file__)))
        response=api.dataset_create_new(str(bundle.resolve()),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
        write(OUT/'model-dataset-upload-response.json',dict(received_utc=stamp(),dataset=dataset,response=payload(response)))
        print('C56 PRIVATE MODEL DATASET UPLOAD RETURNED',a.trial,dataset);return
    assert str(api.dataset_status(dataset))=='ready'
    intent=OUT/(a.trial+'-private-push-intent.json');assert not intent.exists(),'Reconcile existing exact kernel; never requeue blindly'
    write(intent,dict(created_utc=stamp(),kernel=entry['kernel'],version=1,notebook_sha256=entry['notebook_sha256'],source_sha256=sha(__file__),scored_entry=False))
    response=api.kernels_push(str(folder.resolve()),timeout='32400',acc='NvidiaTeslaT4')
    write(OUT/(a.trial+'-private-push-response.json'),dict(received_utc=stamp(),response=payload(response)))
    result=dict(submitted_utc=stamp(),trial=a.trial,kernel=entry['kernel'],version=getattr(response,'version_number',None),url=getattr(response,'url',None),
        error=getattr(response,'error',None),notebook_sha256=entry['notebook_sha256'],head_manifest_sha256=entry['head_manifest_sha256'],scored_entry=False)
    write(OUT/(a.trial+'-private-push-result.json'),result);assert result['version']==1 and not result['error'],result
    print('C56 PRIVATE PRODUCTION PREVIEW RELEASED',json.dumps(result))


if __name__=='__main__':main()
