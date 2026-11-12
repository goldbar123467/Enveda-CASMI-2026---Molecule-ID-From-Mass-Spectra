"""Cache the remaining public frozen-pipeline dataset mounts, without scoring."""
import contextlib,hashlib,io,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'
DATASETS=['dmitriigluzdov/casmi26-fold-safe-fpnet','dmitriigluzdov/casmi26-pubchem-popularity-prior',
    'dmitriigluzdov/casmi26-natural-product-knowledge-table','dmitriigluzdov/casmi26-natural-product-panel-benchmark',
    'sanpier/catalogue-experiment-saner']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(4<<20):h.update(b)
    return h.hexdigest()


def stamp():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def main():
    import msvcrt
    lock=(OUT/'aux-assets.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'aux-asset-freeze.json').exists()
    auth=json.loads((ROOT/'.local/c38-active-goal.json').read_text());assert auth['active'] and auth['free_local_and_kaggle_compute_authorized']
    with contextlib.redirect_stdout(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    inventory=OUT/'aux-asset-inventory.json'
    if inventory.exists():
        census=json.loads(inventory.read_text());assert census['source_sha256']==sha(__file__)
    else:
        datasets={}
        for dataset in DATASETS:
            files={};token=None
            for page in range(40):
                j=json.loads(api.dataset_list_files(dataset,page_token=token,page_size=1000).to_json())
                files.update({r['name']:r.get('totalBytes',0) for r in j['datasetFiles']});token=j.get('nextPageToken')
                if not token:break
            assert not token and files;datasets[dataset]=files
        # The frozen PC scorer computes its shortlist fingerprints itself.
        # The optional64GiB native fingerprint index is not imported anywhere
        # by the frozen pipeline, its PC runner or helper modules. Preserve the
        # complete census, but do not download that unused index's12binary shards.
        public_source=OUT/'probe_core2.py';assert 'fingerprint_index' not in public_source.read_text()
        for path in list(OUT.glob('inner-cell-*.py'))+list((OUT/'modules').glob('*.py')):
            assert 'fingerprint_index' not in path.read_text(encoding='utf8'),path
        selected={d:{n:b for n,b in f.items() if not (n.startswith('fingerprints-') and n.endswith('.bin'))} for d,f in datasets.items()}
        omitted={d:{n:b for n,b in f.items() if n not in selected[d]} for d,f in datasets.items()}
        total=sum(n for f in selected.values() for n in f.values());assert total<40*2**30
        census=dict(created_utc=stamp(),source_sha256=sha(__file__),remote_complete_census=datasets,datasets=selected,
            unused_native_index_binary_shards_not_read_by_frozen_source=omitted,bytes=total,quality_outcomes_read=False,C13_read=False)
        write(inventory,census)
    assert shutil.disk_usage(ROOT).free>census['bytes']+25*2**30
    pins={};count=0
    for dataset,files in census['datasets'].items():
        dest=OUT/'input'/dataset.split('/')[1];dest.mkdir(parents=True,exist_ok=True)
        for name,size in sorted(files.items(),key=lambda r:(r[1],r[0])):
            assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
            target=dest/name;assert target.resolve().is_relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                try:
                    with contextlib.redirect_stdout(io.StringIO()):api.dataset_download_file(dataset,name,path=str(target.parent),quiet=True)
                except Exception as error:raise RuntimeError('Auxiliary asset download failed: '+dataset+'/'+name+' ('+type(error).__name__+')') from None
                if not target.exists():
                    archives=list(target.parent.glob('*.zip'));assert len(archives)==1
                    with zipfile.ZipFile(archives[0]) as archive:
                        members=[n for n in archive.namelist() if Path(n).name==target.name];assert len(members)==1
                        with archive.open(members[0]) as source,target.open('xb') as output:shutil.copyfileobj(source,output)
            assert target.stat().st_size==size,(dataset,name)
            pins[dataset+'/'+name]=sha(target);count+=1
            write(OUT/'aux-asset-status.json',dict(checked_utc=stamp(),pid=__import__('os').getpid(),completed=count,dataset=dataset,last_file=name,
                expected_files=sum(len(f) for f in census['datasets'].values()),bytes=census['bytes'],files_sha256=pins,
                quality_outcomes_read=False,C13_read=False,scoring_entries=0))
        print('C75 AUX DATASET COMPLETE',dataset,len(files),round(sum(files.values())/2**30,2),'GiB',flush=True)
    write(OUT/'aux-asset-freeze.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),inventory_sha256=sha(inventory),files_sha256=pins,
        quality_outcomes_read=False,C13_read=False,scoring_entries=0))


if __name__=='__main__':main()
