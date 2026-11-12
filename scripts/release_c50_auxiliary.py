"""Durable release of one private35-query inference. Cannot score or open labels."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c50-auxiliary-validation-20261112'
PACKAGE=ROOT/'outputs/publish/c50-full-masked-validation-20261111'
FOLDER=ROOT/'kaggle/forks/c50-full-masked-validation'
DATASET='clarkkitchen/casmi26-c50-full-masked-validation-20261111'
KERNEL='clarkkitchen/casmi26-c50-full-masked-validation'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)
def stamp():return datetime.now(timezone.utc).isoformat()
def payload(response):
    value=response.to_json() if hasattr(response,'to_json') else str(response)
    try:return json.loads(value)
    except (ValueError,TypeError):return dict(response=str(value))


def verified():
    authorization=read(ROOT/'.local/c38-active-goal.json')
    assert authorization['active'] and authorization['submissions_approved']==2 and authorization['submissions_used']==0
    assert authorization['free_local_and_kaggle_compute_authorized'] and not authorization['new_paid_compute_authorized']
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    pin=read(FOLDER/'validation-pin.json');build=read(OUT/'validation-build.json');contract=read(PACKAGE/'validation-contract.json')
    assert pin['notebook_sha256']==build['notebook_sha256']==sha(FOLDER/'notebook.ipynb')
    assert pin['query_contract_sha256']==build['query_contract_sha256']==sha(PACKAGE/'validation-contract.json')
    assert pin['queries']==pin['spectra']==contract['queries']==contract['spectra']==35
    assert pin['same_scientific_payload_as_completed_C41R3'] and not pin['labels_mounted']
    assert pin['private'] and not pin['internet'] and not pin['scoring_submission']
    assert pin['wrapper_source_sha256']==sha(ROOT/'scripts/build_c50_auxiliary_validation.py')
    for name,value in contract['files'].items():assert sha(PACKAGE/name)==value
    allowed=set(contract['files'])|{'validation-contract.json','dataset-metadata.json'}
    assert set(p.name for p in PACKAGE.iterdir())==allowed
    assert read(PACKAGE/'dataset-metadata.json')['id']==DATASET
    metadata=read(FOLDER/'kernel-metadata.json');assert metadata['id']==KERNEL and metadata['is_private'] and not metadata['enable_internet']
    assert DATASET in metadata['dataset_sources'] and metadata['machine_shape']=='NvidiaTeslaT4'
    return pin


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['publish','status','push']);args=parser.parse_args()
    import msvcrt
    lock=(OUT/'private-release.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pin=verified()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if args.phase=='status':
        dataset=str(api.dataset_status(DATASET));result=dict(checked_utc=stamp(),dataset=DATASET,dataset_status=dataset)
        if (OUT/'kernel-push-result.json').exists():result['kernel_status']=json.loads(str(api.kernels_status(KERNEL)))
        write(OUT/'actual-status.json',result);print('C50 ACTUAL STATUS',json.dumps(result));return
    if args.phase=='publish':
        intent=OUT/'dataset-upload-intent.json';assert not intent.exists(),'Reconcile existing exact upload intent; never create a duplicate version'
        write(intent,dict(created_utc=stamp(),dataset=DATASET,private=True,convert_to_csv=False,
            contract_sha256=sha(PACKAGE/'validation-contract.json'),source_sha256=sha(__file__)))
        response=api.dataset_create_new(str(PACKAGE.resolve()),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
        write(OUT/'dataset-upload-response.json',dict(received_utc=stamp(),response=payload(response),dataset=DATASET))
        print('C50 PRIVATE QUERY DATASET UPLOAD RETURNED',DATASET);return
    assert str(api.dataset_status(DATASET))=='ready'
    intent=OUT/'kernel-push-intent.json';assert not intent.exists(),'Reconcile existing exact notebook handle; never requeue blindly'
    write(intent,dict(created_utc=stamp(),kernel=KERNEL,version=1,notebook_sha256=pin['notebook_sha256'],source_sha256=sha(__file__),scored_entry=False))
    response=api.kernels_push(str(FOLDER.resolve()),timeout='32400',acc='NvidiaTeslaT4')
    write(OUT/'kernel-push-response.json',dict(received_utc=stamp(),response=payload(response)))
    result=dict(submitted_utc=stamp(),kernel=KERNEL,version=getattr(response,'version_number',None),url=getattr(response,'url',None),
        error=getattr(response,'error',None),notebook_sha256=pin['notebook_sha256'],source_sha256=sha(__file__),scored_entry=False)
    write(OUT/'kernel-push-result.json',result);assert result['version']==1 and not result['error'],result
    print('C50 PRIVATE35 INFERENCE RELEASED',json.dumps(result))


if __name__=='__main__':main()
