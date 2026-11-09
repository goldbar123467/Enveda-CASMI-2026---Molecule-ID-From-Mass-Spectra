"""Bounded supervisor for an explicit continuation plan; no network/submission."""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
from training_monitor import build_snapshot,system_memory

ROOT=Path(__file__).resolve().parents[1]
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--trainer',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.plan.read_text());out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'supervisor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (out/'results.json').exists():print('Already complete; no trainer launched',flush=True);return
    assert args.trainer.resolve().is_relative_to(ROOT/'scripts')
    assert (out/'source-contract.json').exists(),'Pilot/checkpoint recovery must precede the supervisor'
    deadline=datetime.fromisoformat(json.loads((out/'launch.json').read_text())['started_utc'])+timedelta(hours=spec['maximum_hours'])
    old=json.loads((out/'watchdog/status.json').read_text()) if (out/'watchdog/status.json').exists() else {}
    restarts=int(old.get('restarts',0));previous_error=None;child=None;last_hour=0;micro=[192,96];index=int(old.get('micro_index',0))
    try:
        while datetime.now(timezone.utc)<deadline and not (ROOT/'.local/compute-paused.json').exists():
            if child is None or child.poll() is not None:
                if child is not None:
                    if child.returncode==0 and (out/'results.json').exists():break
                    error=(out/f'train-{restarts}.err').read_text(errors='replace')[-6000:]
                    if restarts>=3 or error==previous_error:raise RuntimeError('Recovery cap or repeated failure: '+error[-1000:])
                    if 'out of memory' in error.lower() and index+1<len(micro):index+=1
                    if system_memory().get('free_gib',0)<spec['minimum_free_ram_gib']:
                        write(out/'watchdog/status.json',dict(action='resource_wait',checked_utc=stamp(),pid=os.getpid(),restarts=restarts));time.sleep(10);continue
                    previous_error=error;restarts+=1
                with (out/f'train-{restarts}.log').open('w') as stdout,(out/f'train-{restarts}.err').open('w') as stderr:
                    child=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-u',str(args.trainer),'--plan',str(args.plan),'--micro-batch',str(micro[index])],
                        cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1'),stdout=stdout,stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            if time.monotonic()-last_hour>=3600:
                report=dict(checked_utc=stamp(),local=build_snapshot(out),restarts=restarts)
                write(out/'hourly'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ.json'),report)
                write(out/'hourly/status.json',dict(checked_utc=stamp(),runner_pid=os.getpid(),next_check_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
                print('HOURLY',json.dumps(dict(checked_utc=report['checked_utc'],progress=report['local']['progress'])),flush=True);last_hour=time.monotonic()
            write(out/'watchdog/status.json',dict(action='healthy',pid=os.getpid(),trainer_pids=[child.pid],checked_utc=stamp(),
                next_check_utc=(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat(),restarts=restarts,micro_index=index,micro_batch=micro[index],deadline_utc=deadline.isoformat()))
            time.sleep(5)
        if child and child.poll() is None:
            subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=30);raise RuntimeError('Pause or wall budget reached')
        assert (out/'results.json').exists(),'Training did not complete'
        results=json.loads((out/'results.json').read_text())['models']
        path=ROOT/'runs.csv'
        with path.open(newline='',encoding='utf-8') as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
        row=next(r for r in rows if r['id']==spec['run_id'])
        row.update(status='complete',ended_utc=max(r['completed_utc'] for r in results)[:16].replace('T',' '),
            result=f"{len(results)} fits x{spec['training']['steps']} continuation updates={sum(r['continuation_steps'] for r in results)};{sum(r['amp_skips'] for r in results)} AMP skips;{restarts} restarts;peak reserved{max(r['gpu_peak_reserved_gib'] for r in results):.4f}GiB; validation pending, no submission")
        temporary=path.with_suffix('.continuation.tmp')
        with temporary.open('w',newline='',encoding='utf-8') as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temporary.replace(path)
        write(out/'watchdog/status.json',dict(action='complete',pid=os.getpid(),trainer_pids=[],checked_utc=stamp(),restarts=restarts))
        subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/update_runs.py'],cwd=ROOT,check=True,timeout=300)
        print('DONE SUPERVISED CONTINUATION',flush=True)
    except BaseException as exc:
        write(out/'watchdog/status.json',dict(action='needs_attention',pid=os.getpid(),checked_utc=stamp(),restarts=restarts,error=repr(exc)));raise


if __name__=='__main__':main()
