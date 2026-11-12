"""Release query-only compatibility inference only after all prior gates pass."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from submit_c09_external_once import scientific_sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c59-c13-compatibility-20261112'
FOLDER=ROOT/'kaggle/forks/c59-c13-main-compatibility'
KERNEL='clarkkitchen/casmi26-c59-c13-main-compatibility'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    p=Path(path);temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf8');temporary.replace(p)


def verified():
    auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_approved']==2 and auth['submissions_used']==0
    assert auth['free_local_and_kaggle_compute_authorized'] and not auth['new_paid_compute_authorized']
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    source=ROOT/'outputs/local/c58-complete-target-20261112';quality=read(source/'source-results.json');audit=read(source/'independent-quality-audit.json')
    assert quality['passed_pair'] and audit['passed_pair'] and audit['quality_results_sha256']==sha(source/'source-results.json')
    assert quality['evaluation_freeze_sha256']==audit['evaluation_freeze_sha256']==sha(source/'evaluation-freeze.json')
    assert audit['source_sha256']==sha(ROOT/'scripts/audit_c58_quality.py') and not quality['C13_read']
    frozen=read(source/'evaluation-freeze.json')
    for name,pin in frozen['files_sha256'].items():assert sha(ROOT/name)==pin
    assert quality['consumption_sha256']==sha(ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json')
    model=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112';family=read(model/'complete-candidate-freeze.json')
    assert read(model/'independent-used-quality-audit.json')['passed_pair']
    for entry in family['candidates'].values():
        p=ROOT/'outputs/forks'/(Path(entry['path']).name+'-v1')/'independent-production-audit.json';a=read(p)
        assert a['execution_qualified'] and a['local_cloud_final_lists_exact'] and a['notebook_sha256']==entry['notebook_sha256']
        assert a['head_manifest_sha256']==entry['head_manifest_sha256'] and a['source_sha256']==sha(ROOT/'scripts/audit_c56_production.py')
        for name,pin in a['artifact_sha256'].items():assert sha(ROOT/name)==pin
    pin=read(FOLDER/'compatibility-pin.json');assert pin['notebook_sha256']==sha(FOLDER/'notebook.ipynb')
    assert pin['scientific_sha256']==scientific_sha(read(FOLDER/'notebook.ipynb'))
    assert pin['source_sha256']==sha(ROOT/'scripts/build_c59_c13_compatibility.py') and not pin['query_tokens_or_selection_changed']
    assert pin['queries']==120 and pin['query_modes']=={'positive':60,'negative':60} and not pin['query_labels_mounted']
    assert pin['PV_v2_repair_unchanged'] and pin['all_other_scientific_cells_exact'] and pin['changed_inner_cells']==[27]
    assert pin['query_sha256']==sha(ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet')
    assert pin['query_contract_sha256']==sha(ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-contract.json')
    exposure=read(OUT/'global-reference-preflight.json')
    assert exposure['global_inventory_keys']==231 and exposure['all_reference_raw_identity_overlap']==exposure['all_reference_official_identity_overlap']==0
    assert exposure['global_inventory_sha256']==sha(ROOT/'outputs/local/c13-confirmation-20261110/identity-inventory.parquet')
    assert exposure['reference_witness_sha256']==sha(model/'original-structure-identity-witnesses.parquet')
    metadata=read(FOLDER/'kernel-metadata.json');assert metadata['id']==KERNEL and metadata['is_private'] and not metadata['enable_internet'] and metadata['machine_shape']=='NvidiaTeslaT4'
    return pin


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['push','status']);a=p.parse_args()
    import msvcrt
    lock=(OUT/'private-release.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pin=verified()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if a.phase=='status':
        state=json.loads(str(api.kernels_status(KERNEL)));write(OUT/'actual-status.json',dict(checked_utc=stamp(),kernel=KERNEL,state=state));print('C59 ACTUAL STATUS',state);return
    assert not (OUT/'kernel-push-intent.json').exists(),'Reconcile the existing exact handle; never requeue blindly'
    write(OUT/'kernel-push-intent.json',dict(created_utc=stamp(),kernel=KERNEL,version=1,notebook_sha256=pin['notebook_sha256'],source_sha256=sha(__file__),
        C58_quality_sha256=sha(ROOT/'outputs/local/c58-complete-target-20261112/source-results.json'),scored_entry=False,C13_quality_read=False))
    response=api.kernels_push(str(FOLDER.resolve()),timeout='32400',acc='NvidiaTeslaT4')
    value=response.to_json() if hasattr(response,'to_json') else str(response)
    write(OUT/'kernel-push-response.json',dict(received_utc=stamp(),response=value))
    result=dict(submitted_utc=stamp(),kernel=KERNEL,version=getattr(response,'version_number',None),url=getattr(response,'url',None),error=getattr(response,'error',None),
        notebook_sha256=pin['notebook_sha256'],scientific_sha256=pin['scientific_sha256'],scored_entry=False,C13_quality_read=False)
    write(OUT/'kernel-push-result.json',result);assert result['version']==1 and not result['error'],result
    print('C59 QUERY-ONLY COMPATIBILITY INFERENCE RELEASED',result,flush=True)


if __name__=='__main__':main()
