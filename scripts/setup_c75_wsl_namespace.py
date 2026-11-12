"""Create a task-owned Kaggle-shaped Linux namespace without changing assets.

Only file links and fresh task directories are created. Root mode may switch
ONLY this task's verified /kaggle links; unknown existing namespaces are refused.
An unfinished asset plan is never permission to start inference.
"""
import argparse,json,os,sys
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'
NATIVE=Path('/home/imsa/.local/share/enveda-c75')
OWNER='enveda-c75-local-core-port-20261112'


def file_link(source,target):
    source=source.resolve(strict=False)
    assert source.is_relative_to(ROOT),source
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():assert target.resolve(strict=False)==source
    else:
        assert not target.exists(),('Refusing non-task file overwrite',target)
        target.symlink_to(source)


def plan(mode):
    assert sys.platform.startswith('linux') and os.geteuid()!=0
    port=OUT/(mode+'-port');pin=read(port/'port-freeze.json');audit=read(port/'port-source-audit.json')
    assert audit['source_and_fixture_audit_passed'] and audit['port_freeze_sha256']==sha(port/'port-freeze.json')
    home=NATIVE/mode;input_dir=home/'input';working=home/'working'
    input_dir.mkdir(parents=True,exist_ok=True);working.mkdir(exist_ok=True)
    owner=home/'owner.json'
    if owner.exists():assert read(owner)['owner']==OWNER and read(owner)['workspace']==str(ROOT)
    else:write(owner,dict(owner=OWNER,workspace=str(ROOT),mode=mode,created_utc=stamp()))
    census=read(OUT/'asset-protocol.json');old=read(ROOT/'outputs/research/c20-engine-recovery-20261110/asset-files.json')['files']
    assert census['files']==len(old)==237
    links=0
    for row in old:
        rel=Path('casmi26-attributed-v4g-reproduction-assets')/row['name'];file_link(OUT/'input'/rel,input_dir/rel);links+=1
    auxiliary=read(OUT/'aux-asset-inventory.json')
    for dataset,files in auxiliary['datasets'].items():
        for name in files:
            rel=Path(dataset.split('/')[1])/name;file_link(OUT/'input'/rel,input_dir/rel);links+=1
    neural=read(OUT/'cached-neural-bundles.json')
    for name,pin in neural['public4pins'].items():
        source=OUT/'input/casmi-spectrum-model-v1/pinned-bundle'/name;assert sha(source)==pin
        file_link(source,input_dir/'casmi-spectrum-model-v1/pinned-bundle'/name);links+=1
    ours=OUT/'input/v80-source/casmi_spectrum_model_ours_v80/final_model.pt'
    assert sha(ours)==neural['ours_checkpoint_sha256'];file_link(ours,input_dir/'v80-source/casmi_spectrum_model_ours_v80/final_model.pt');links+=1
    frozen=ROOT/'outputs/publish/frozen-p14-20261107';manifest=read(frozen/'frozen-manifest.json')
    assert sha(frozen/'frozen-manifest.json')=='22c5300c20a942fdd346aa26ca5f57237664e6a251e4c48ba18daaace16eac19'
    for name in set(manifest['files'])|{'frozen-manifest.json'}:
        source=frozen/name;assert source.is_file();file_link(source,input_dir/'casmi26-frozen-p14-20261107'/name);links+=1
    package=port/'query-package';contract=read(package/'validation-contract.json')
    for name in set(contract['files'])|{'validation-contract.json'}:
        file_link(package/name,input_dir/'c75-query-only'/name);links+=1
    competition=input_dir/'competitions/enveda-CASMI26-molecule-id-mass-spectra'
    file_link(ROOT/'outputs/competition-raw/train.parquet',competition/'train.parquet')
    file_link(package/'validation-queries.parquet',competition/'test.parquet')
    file_link(package/'validation-sample.csv',competition/'sample_submission.csv');links+=3
    assert len(list(input_dir.rglob('validation-contract.json')))==1
    assert len(list(input_dir.rglob('frozen-manifest.json')))==1
    write(port/'namespace-plan.json',dict(prepared_utc=stamp(),source_sha256=sha(__file__),owner=OWNER,workspace=str(ROOT),mode=mode,
        native_root=str(home),input=str(input_dir),working=str(working),file_links=links,
        complete_asset_freezes_still_required=True,port_freeze_sha256=sha(port/'port-freeze.json'),new_source_outcomes_read=False,C13_read=False))
    print('C75 TASK NAMESPACE PLANNED',mode,links,'file links; no inference started',flush=True)


def activate(mode):
    assert sys.platform.startswith('linux') and os.geteuid()==0
    port=OUT/(mode+'-port');p=read(port/'namespace-plan.json');home=NATIVE/mode
    assert p['owner']==OWNER and p['workspace']==str(ROOT) and p['native_root']==str(home)
    assert read(home/'owner.json')['owner']==OWNER
    namespace=Path('/kaggle');marker=namespace/'.c75-owner.json'
    if namespace.exists():
        assert marker.is_file() and read(marker)['owner']==OWNER and read(marker)['workspace']==str(ROOT),'Unknown existing /kaggle namespace'
        assert set(x.name for x in namespace.iterdir())<={'input','working','.c75-owner.json'}
    else:
        namespace.mkdir();write(marker,dict(owner=OWNER,workspace=str(ROOT),created_utc=stamp()))
    for name in ['input','working']:
        link=namespace/name;target=home/name
        assert target.resolve().is_relative_to(NATIVE.resolve()) and target.is_dir()
        if link.is_symlink():
            old=link.resolve(strict=False);assert old.is_relative_to(NATIVE.resolve()),('Unknown existing task link',link,old)
            if old==target.resolve():continue
            link.unlink()  # Unlink this verified task pointer only; retain ALL old data.
        else:assert not link.exists(),('Refusing to replace non-link',link)
        link.symlink_to(target,target_is_directory=True)
    write(port/'namespace-activation.json',dict(activated_utc=stamp(),source_sha256=sha(__file__),owner=OWNER,workspace=str(ROOT),mode=mode,
        input_target=str((namespace/'input').resolve()),working_target=str((namespace/'working').resolve()),only_owned_links_switched=True,
        no_existing_data_deleted_or_moved=True,new_source_outcomes_read=False,C13_read=False,inference_started=False))
    print('C75 VERIFIED TASK NAMESPACE ACTIVATED',mode,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['plan','activate']);parser.add_argument('mode',choices=['fixture','full']);args=parser.parse_args()
    (plan if args.phase=='plan' else activate)(args.mode)
