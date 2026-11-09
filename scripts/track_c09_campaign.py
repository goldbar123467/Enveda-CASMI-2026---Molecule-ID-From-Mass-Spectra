"""Read-only C09 account/cloud tracker and hourly health evidence; never submits."""
import csv,json,os,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
from c09_submission_guard import normalize
from training_monitor import build_snapshot
from finish_compatible_campaign import status_name

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c09-20261109'
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    import msvcrt
    lock=(OUT/'tracker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();last_hour=0;previous=None
    deadline=datetime.now(timezone.utc)+timedelta(hours=24)
    while datetime.now(timezone.utc)<deadline and not (ROOT/'.local/c09-tracker-stop.json').exists():
        try:
            submissions=[normalize(r) for r in api.competition_submissions('enveda-CASMI26-molecule-id-mass-spectra')]
            pending=[r for r in submissions if r['status'] not in ('COMPLETE','ERROR','CANCELLED','CANCELED')]
            with (ROOT/'runs.csv').open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            refs=list(dict.fromkeys('clarkkitchen/casmi26-'+r['ref'].split()[0] for r in rows if r['id'].startswith('C09') and r['platform'].startswith('kaggle')))
            cloud={}
            for ref in refs:
                value=api.kernels_status(ref);cloud[ref]=dict(status=status_name(value))
            active=OUT
            pointer=ROOT/'outputs/local/c09-2070-queue-20261109/active-run.json'
            if pointer.exists():
                selected=json.loads(pointer.read_text());candidate=(ROOT/selected.get('run','')).resolve()
                if candidate.is_relative_to((ROOT/'outputs/local').resolve()) and (candidate/'manifest.json').exists():active=candidate
            local=build_snapshot(active);report=dict(checked_utc=stamp(),runner_pid=os.getpid(),local=local,cloud=cloud,pending=pending,latest_submissions=submissions[:7],
                maximum_pending=1,submission_limit_respected=len(pending)<=1,next_check_utc=(datetime.now(timezone.utc)+timedelta(seconds=180)).isoformat())
            write(OUT/'campaign-tracker.json',report)
            change=json.dumps(dict(cloud=cloud,submissions=submissions[:3],local_state=local['state']),sort_keys=True)
            if change!=previous:
                print('C09 STATE CHANGE',json.dumps(dict(checked_utc=report['checked_utc'],cloud=cloud,pending=pending,local_state=local['state'])),flush=True);previous=change
            if time.monotonic()-last_hour>=3600:
                path=OUT/'hourly'/('campaign-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ.json'));write(path,report)
                write(OUT/'hourly/status.json',dict(checked_utc=stamp(),runner_pid=os.getpid(),next_check_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
                last_hour=time.monotonic();print('C09 HOURLY',json.dumps(dict(checked_utc=report['checked_utc'],progress=local['progress'],cloud=cloud,pending=pending)),flush=True)
        except Exception as exc:
            write(OUT/'tracker-error.json',dict(checked_utc=stamp(),error=repr(exc)));print('C09 TRACKER ERROR',type(exc).__name__,flush=True)
        for _ in range(36):
            if (ROOT/'.local/c09-tracker-stop.json').exists():return
            time.sleep(5)
    print('C09 TRACKER EXIT',stamp(),flush=True)


if __name__=='__main__':main()
