"""Cache exact Engine2 dependencies as a bounded local recovery contingency."""
import contextlib,hashlib,io,json,os,time,zipfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research/c20-engine-recovery-20261110'
DATASET='dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets'


def stamp():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def write(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)


def main():
    import msvcrt
    lock=(OUT/'asset-worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'asset-intent.json').exists(),'Inspect existing asset worker before any recovery'
    listing=json.loads((OUT/'asset-files.json').read_text());items={f['name']:f['bytes'] for f in listing['files']}
    names=['coconut-casmi26-candidates/'+n for n in ['coco_fp.npy','coco_mass.npy','coco_meta.pkl','fp_bits.npy']]
    names+=['casmi-chebi-lmsd-structures/'+n for n in ['bio_fp.npy','bio_mass.npy','bio_meta.pkl','ATTRIBUTION.md']]
    names+=['casmi26-pubchemlite-cc0-candidate-fingerprints/'+n for n in ['pubchemlite_fp.npy','pubchemlite_mass.npy','pubchemlite_meta.pkl','MANIFEST.json']]
    names+=['casmi26-fp-models-v2/'+n for n in ['fp_merged_m1.pt','fp_single_s2.pt']]
    names+=['casmi26-simulated-ranker-rows/sim_rank_rows_nofp.npz','casmi26-ranker-features/rank_train.npz']
    assert all(n in items for n in names)
    deadline=datetime.now(timezone.utc)+timedelta(hours=2)
    write('asset-intent.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),dataset=DATASET,
        requested_files={n:items[n] for n in names},bytes=sum(items[n] for n in names),purpose='Prepare local contingency only; no inference/labels/metrics/submission; leave queued T4 run untouched.'))
    with contextlib.redirect_stdout(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    pins={}
    for name in names:
        assert datetime.now(timezone.utc)<deadline
        assert not (ROOT/'.local/compute-paused.json').exists()
        folder=OUT/'assets'/Path(name).parent;folder.mkdir(parents=True,exist_ok=True)
        path=folder/Path(name).name
        if not path.exists():
            with contextlib.redirect_stdout(io.StringIO()):api.dataset_download_file(DATASET,name,path=str(folder),quiet=True)
            if not path.exists():
                archives=list(folder.glob('*.zip'));assert len(archives)==1,'Inspect unrecognized downloaded file'
                with zipfile.ZipFile(archives[0]) as archive:
                    candidates=[n for n in archive.namelist() if Path(n).name==path.name]
                    assert len(candidates)==1
                    with archive.open(candidates[0]) as source,path.open('xb') as target:
                        import shutil
                        shutil.copyfileobj(source,target)
        assert path.stat().st_size==items[name],name
        pins[name]=sha(path)
        write('asset-status.json',dict(checked_utc=stamp(),pid=os.getpid(),state='downloading',completed=len(pins),total=len(names),files_sha256=pins,
            labels_read=False,inference_started=False,automatic_submission=False))
        print('C20 LOCAL ASSET',len(pins),len(names),name,flush=True)
    write('asset-status.json',dict(completed_utc=stamp(),pid=os.getpid(),state='complete',completed=len(pins),total=len(names),dataset=DATASET,
        files_sha256=pins,labels_read=False,inference_started=False,automatic_submission=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write('asset-attention.json',dict(checked_utc=stamp(),pid=os.getpid(),error=type(error).__name__+': '+str(error),automatic_retry=False))
        raise
