"""Bounded C29collection -> fixed development -> C30selection, never confirm."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
from update_runs import process_running

OUT=ROOT/'outputs/local/c29-public-replay-20261110'


def main():
    import msvcrt
    lock=(OUT/'development-worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'development.json').exists(),'Fixed C29development already complete'
    deadline=datetime.fromisoformat('2026-11-10T22:08:41+00:00');child=None
    def guard():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Campaign worker deadline')
        if (ROOT/'.local/compute-paused.json').exists():raise RuntimeError('Project paused')
    def report(state,stage,**extra):
        write(OUT/'worker-status.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            deadline_utc=deadline.isoformat(),state=state,stage=stage,C13_confirmation_metrics_read=False,automatic_submission=False,**extra))
    def run(path,*args):
        nonlocal child
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/path),*args],cwd=ROOT)
        while child.poll() is None:guard();time.sleep(10)
        assert child.returncode==0,path+' failed; inspect exact artifacts before recovery'
    try:
        while True:
            guard()
            if (OUT/'collector-attention.json').exists():raise RuntimeError('Collector needs attention; no quality rejection inferred')
            if (OUT/'collector-complete.json').exists():
                collected=json.loads((OUT/'collector-complete.json').read_text())
                if not process_running(collected['pid']):break
            report('waiting','waiting exact C29collector terminal and exit');time.sleep(30)
        report('running','fixed full240 C29development')
        run('develop_c29_public.py','evaluate')
        prior=ROOT/'outputs/local/c28-evaluation-20261110'
        while not (prior/'development.json').exists():
            guard()
            if (prior/'worker-attention.json').exists():raise RuntimeError('C28 development needs attention; outer selection waits')
            report('waiting','C29complete; waiting C28fixed development');time.sleep(30)
        report('running','both separate development cycles complete; C30outer selector')
        run('select_c30_global.py')
        selection=ROOT/'outputs/local/c30-global-selection-20261110/selection-freeze.json'
        result=json.loads(selection.read_text())
        report('complete','both cycles selected once; reserved120 untouched',selected_cycle=result['selected_cycle'],selection_sha256=sha(selection))
        print('DONE C29 DEVELOPMENT WORKER',flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,text=True)
        report('error','inspect before recovery',error_type=type(exc).__name__)
        write(OUT/'worker-attention.json',dict(error_type=type(exc).__name__,reason=str(exc),automatic_retry=False))
        raise


if __name__=='__main__':main()
