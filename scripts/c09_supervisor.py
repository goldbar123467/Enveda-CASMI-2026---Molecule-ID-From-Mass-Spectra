"""Own one C09 continuation trainer and hourly health reports; never submits."""
import csv,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
from training_monitor import build_snapshot,system_memory
from c09_submission_guard import normalize

ROOT=Path(__file__).resolve().parents[1]
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    spec=json.loads((ROOT/'configs/c09-training-20261109.json').read_text());out=ROOT/spec['output']
    import msvcrt
    lock=(out/'supervisor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    deadline=datetime.fromisoformat(json.loads((out/'launch.json').read_text())['started_utc'])+timedelta(hours=spec['maximum_hours'])
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();last_remote=last_hour=0;remote={};child=None;restarts=0;previous_error=None
    micro=[192,96];index=0
    try:
        while datetime.now(timezone.utc)<deadline and not (ROOT/'.local/compute-paused.json').exists():
            if child is None or child.poll() is not None:
                if child is not None:
                    if child.returncode==0 and (out/'results.json').exists():break
                    error=(out/f'train-{restarts}.err').read_text(errors='replace')[-6000:]
                    if restarts>=3 or error==previous_error:raise RuntimeError('Bounded recovery exhausted/repeated failure: '+error[-1000:])
                    if 'out of memory' in error.lower() and index+1<len(micro):index+=1
                    if system_memory().get('free_gib',0)<spec['minimum_free_ram_gib']:
                        write(out/'watchdog/status.json',dict(action='resource_wait',checked_utc=stamp(),pid=os.getpid(),restarts=restarts))
                        time.sleep(10);continue
                    previous_error=error;restarts+=1
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                with (out/f'train-{restarts}.log').open('w') as stdout,(out/f'train-{restarts}.err').open('w') as stderr:
                    child=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-u','scripts/c09_train.py','--micro-batch',str(micro[index])],
                        cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            now=time.monotonic()
            if now-last_remote>=180:
                try:
                    submissions=[normalize(s) for s in api.competition_submissions('enveda-CASMI26-molecule-id-mass-spectra')]
                    status=api.kernels_status('clarkkitchen/casmi26-c09-coverage-controls')
                    status=json.loads(status.to_json()) if hasattr(status,'to_json') else status
                    remote=dict(checked_utc=stamp(),submissions=submissions[:7],cloud=status)
                    write(out/'remote-status.json',remote)
                except Exception as exc:remote=dict(checked_utc=stamp(),error=type(exc).__name__)
                last_remote=now
            if now-last_hour>=3600:
                report=dict(checked_utc=stamp(),local=build_snapshot(out),remote=remote,restarts=restarts)
                write(out/'hourly'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ.json'),report)
                write(out/'hourly/status.json',dict(checked_utc=stamp(),runner_pid=os.getpid(),next_check_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
                print('HOURLY C09',json.dumps(dict(checked_utc=report['checked_utc'],progress=report['local']['progress'],remote=remote)),flush=True);last_hour=now
            write(out/'watchdog/status.json',dict(action='healthy',pid=os.getpid(),trainer_pids=[child.pid],checked_utc=stamp(),
                next_check_utc=(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat(),restarts=restarts,micro_batch=micro[index],deadline_utc=deadline.isoformat()))
            time.sleep(10)
        if child and child.poll() is None:
            subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=30)
            raise RuntimeError('C09 deadline/pause reached')
        assert (out/'results.json').exists(),'Training did not complete'
        write(out/'watchdog/status.json',dict(action='complete',pid=os.getpid(),trainer_pids=[],checked_utc=stamp(),restarts=restarts))
        print('DONE C09 SUPERVISOR',flush=True)
    except BaseException as exc:
        write(out/'watchdog/status.json',dict(action='needs_attention',pid=os.getpid(),checked_utc=stamp(),restarts=restarts,error=repr(exc)))
        raise


if __name__=='__main__':main()
