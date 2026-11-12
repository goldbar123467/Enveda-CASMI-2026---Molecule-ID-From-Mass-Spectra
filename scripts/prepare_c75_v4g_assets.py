"""Cache the original public full-core assets for the free local GPU fallback.

No inference, labels, prediction metrics or scoring. Reuse only receipt-pinned
old copies. Every remote file name/size must match the original frozen census.
"""
import contextlib,hashlib,io,json,shutil,time,zipfile
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'
DATASET='dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'
DEST=OUT/'input/casmi26-attributed-v4g-reproduction-assets'


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
    lock=(OUT/'asset-worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'v4g-asset-freeze.json').exists()
    original=ROOT/'outputs/research/c20-engine-recovery-20261110/asset-files.json'
    expected={r['name']:r['bytes'] for r in json.loads(original.read_text())['files']}
    assert len(expected)==237
    with contextlib.redirect_stdout(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    token=None;remote={}
    for page in range(20):
        j=json.loads(api.dataset_list_files(DATASET,page_token=token,page_size=1000).to_json())
        remote.update({r['name']:r.get('totalBytes',0) for r in j['datasetFiles']})
        token=j.get('nextPageToken')
        if not token:break
    assert not token and remote==expected,'Remote asset census changed; inspect before selecting any model version'
    intent=OUT/'asset-protocol.json'
    if intent.exists():
        protocol=json.loads(intent.read_text());assert protocol['source_sha256']==sha(__file__)
        deadline=datetime.fromisoformat(protocol['deadline_utc'])
    else:
        deadline=datetime.now(timezone.utc)+timedelta(hours=8)
        write(intent,dict(started_utc=stamp(),source_sha256=sha(__file__),deadline_utc=deadline.isoformat(),dataset=DATASET,
            original_census_sha256=sha(original),remote_names_sizes_exact=True,files=237,bytes=sum(expected.values()),
            purpose='Original full-core local RTX2070fallback after actual free Kaggle weekly30hourGPUquota rejection',
            quality_outcomes_read=False,C13_read=False,scoring_entries=0,new_paid_compute=False))
    assert shutil.disk_usage(ROOT).free>sum(expected.values())+10*2**30
    authorization=json.loads((ROOT/'.local/c38-active-goal.json').read_text())
    assert authorization['active'] and authorization['free_local_and_kaggle_compute_authorized']
    prior=ROOT/'outputs/research/c20-engine-recovery-20261110'
    old=json.loads((prior/'asset-status.json').read_text());reuse=old['files_sha256']
    pins={};reused=0;DEST.mkdir(parents=True,exist_ok=True)
    # Read code/manifests before the large arrays so dependency/runtime review
    # can proceed while all original data assets are cached.
    names=sorted(expected,key=lambda n:(expected[n]>2**20,expected[n],n))
    for name in names:
        assert datetime.now(timezone.utc)<deadline
        legacy_pause=ROOT/'.local/compute-paused.json'
        if legacy_pause.exists():
            paused=json.loads(legacy_pause.read_text())
            # Preserve the historical marker to keep old campaigns OFF. The
            # explicit later user goal authorizes this isolated fallback.
            assert datetime.fromisoformat(paused['paused_utc'])<datetime.fromisoformat(authorization['authorized_utc'])
        target=DEST/name;assert target.resolve().is_relative_to(DEST.resolve());target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            oldfile=prior/'assets'/name
            if name in reuse and oldfile.exists():
                assert sha(oldfile)==reuse[name] and oldfile.stat().st_size==expected[name]
                shutil.copyfile(oldfile,target);reused+=1
            else:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):api.dataset_download_file(DATASET,name,path=str(target.parent),quiet=True)
                except Exception as error:raise RuntimeError('Asset download failed: '+name+' ('+type(error).__name__+')') from None
                if not target.exists():
                    archives=list(target.parent.glob('*.zip'));assert len(archives)==1,('Inspect download layout',name)
                    with zipfile.ZipFile(archives[0]) as archive:
                        members=[n for n in archive.namelist() if Path(n).name==target.name];assert len(members)==1
                        with archive.open(members[0]) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest)
        assert target.stat().st_size==expected[name],('Wrong original asset byte count',name)
        pins[name]=sha(target)
        write(OUT/'asset-status.json',dict(checked_utc=stamp(),pid=__import__('os').getpid(),stage='caching_original_v4g',
            completed=len(pins),total=237,completed_bytes=sum(expected[n] for n in pins),total_bytes=sum(expected.values()),last_file=name,reused_files=reused,
            files_sha256=pins,quality_outcomes_read=False,C13_read=False,scoring_entries=0))
        if len(pins)%25==0:print('C75 ORIGINAL ASSETS',len(pins),'/237',round(sum(expected[n] for n in pins)/2**30,2),'GiB',flush=True)
    write(OUT/'v4g-asset-freeze.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),protocol_sha256=sha(intent),dataset=DATASET,
        files=237,bytes=sum(expected.values()),files_sha256=pins,original_remote_census_exact=True,quality_outcomes_read=False,C13_read=False,scoring_entries=0))
    print('C75 ORIGINAL V4G ASSETS COMPLETE',round(sum(expected.values())/2**30,2),'GiB',flush=True)


if __name__=='__main__':main()
