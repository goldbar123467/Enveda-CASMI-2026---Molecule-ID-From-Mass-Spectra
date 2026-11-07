"""Bounded30-minute health supervisor for the new frozen-feature run."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def terminate_tree(child):
    if child.poll() is None:
        # The venv redirector has a Python child on Windows. Kill only this owned tree.
        subprocess.run(['taskkill.exe','/PID',str(child.pid),'/T','/F'],capture_output=True,timeout=30)
        child.wait(timeout=30)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=ROOT/'configs/frozen-training-20261107.json')
    args=parser.parse_args();spec=json.loads(args.plan.read_text());out=ROOT/spec['output'];out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'supervisor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    from long_run_watchdog import inventory
    existing=[p for p in inventory() if str(ROOT/'scripts/frozen_retrieval.py').lower() in (p.get('CommandLine') or '').lower()]
    assert not existing,'An existing frozen trainer must be inspected before restarting its supervisor'
    state_dir=out/'watchdog';state_dir.mkdir(exist_ok=True)
    launch=out/'launch.json'
    if not launch.exists():write(launch,dict(started_utc=datetime.now(timezone.utc).isoformat()))
    deadline=datetime.fromisoformat(json.loads(launch.read_text())['started_utc'])+timedelta(hours=spec['maximum_hours'])
    pins={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/frozen_retrieval.py','scripts/frozen_model.py','outputs/local/kit/features.py']}
    write(out/'supervisor-pins.json',dict(source_sha256=pins,plan=spec,deadline_utc=deadline.isoformat()))
    restarts=0;previous_error=None;child=None
    while True:
        if (ROOT/'.local/compute-paused.json').exists():break
        if datetime.now(timezone.utc)>=deadline:break
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value for p,value in pins.items()),'Scientific source changed; manual review required'
        assert json.loads(args.plan.read_text())==spec,'Training plan changed'
        if child is None or child.poll() is not None:
            if child is not None:
                status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
                if child.returncode==0 and status.get('state')=='complete':
                    write(state_dir/'status.json',dict(action='complete',pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),restarts=restarts));return
                error=(out/f'train-{restarts}.err').read_text(errors='replace')
                signature=hashlib.sha256(error[-6000:].encode()).hexdigest()
                if restarts>=3 or signature==previous_error:
                    write(state_dir/'status.json',dict(action='needs_attention',reason='Repeated error or restart cap',restarts=restarts));return
                previous_error=signature;restarts+=1
            stdout=(out/f'train-{restarts}.log').open('w',encoding='utf-8');stderr=(out/f'train-{restarts}.err').open('w',encoding='utf-8')
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            child=subprocess.Popen([str(ROOT/'.venv-encoders/Scripts/python.exe'),'-u',str(ROOT/'scripts/frozen_retrieval.py'),'--plan',str(args.plan)],
                cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            stdout.close();stderr.close()
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        updated=status.get('updated_utc')
        stale=bool(updated and (datetime.now(timezone.utc)-datetime.fromisoformat(updated)).total_seconds()>900)
        if stale and child.poll() is None:
            # This Popen object exclusively owns the child; no global process lookup/kill.
            terminate_tree(child)
        result=dict(action='restart_stalled' if stale else 'healthy',pid=os.getpid(),trainer_pids=[child.pid],restarts=restarts,
                    checked_utc=datetime.now(timezone.utc).isoformat(),next_check_utc=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat(),deadline_utc=deadline.isoformat())
        write(state_dir/'status.json',result);print(json.dumps(result),flush=True)
        # Frequent parent wakeups observe completion/pause; the health audit remains30 minutes.
        for _ in range(1800):
            if child.poll() is not None or (ROOT/'.local/compute-paused.json').exists() or datetime.now(timezone.utc)>=deadline:break
            time.sleep(1)
    if child is not None and child.poll() is None:terminate_tree(child)
    write(state_dir/'status.json',dict(action='paused' if (ROOT/'.local/compute-paused.json').exists() else 'budget_reached',pid=os.getpid(),restarts=restarts))


if __name__=='__main__':main()
