"""Bounded C28 development worker; waits exact trainer exit, never submits."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

from precision_cache import ROOT,sha,write
from update_runs import process_running

RUN=ROOT/'outputs/local/c28-multiview-20261110'
OUT=ROOT/'outputs/local/c28-evaluation-20261110'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--until-utc',required=True);args=parser.parse_args()
    deadline=datetime.fromisoformat(args.until_utc.replace('Z','+00:00'))
    OUT.mkdir(exist_ok=True)
    import msvcrt
    lock=(OUT/'worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'development.json').exists(),'Development already complete'
    protocol=sha(OUT/'development-protocol.json');child=None
    def report(state,stage,**extra):
        write(OUT/'worker-status.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=__import__('os').getpid(),
            deadline_utc=deadline.isoformat(),state=state,stage=stage,protocol_sha256=protocol,
            C13_confirmation_metrics_read=False,automatic_submission=False,**extra))
    try:
        while True:
            if datetime.now(timezone.utc)>=deadline:raise TimeoutError('C28 development deadline')
            if (ROOT/'.local/compute-paused.json').exists():raise RuntimeError('Project paused')
            assert sha(OUT/'development-protocol.json')==protocol
            status=json.loads((RUN/'status.json').read_text())
            live=process_running(status['pid'])
            if not live:
                assert status['state']=='validation_pending' and (RUN/'results.json').exists(),'Trainer exited without both final fits; inspect before recovery'
                break
            report('waiting','waiting exact C28 trainer exit',trainer_pid=status['pid'],trainer_state=status['state'])
            time.sleep(30)
        report('running','full240 final-model inference/development')
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/develop_c28_multiview.py'),'evaluate'],cwd=ROOT)
        while child.poll() is None:
            if datetime.now(timezone.utc)>=deadline:raise TimeoutError('C28 development deadline')
            if (ROOT/'.local/compute-paused.json').exists():raise RuntimeError('Project paused')
            time.sleep(10)
        assert child.returncode==0,'C28 development process failed; inspect exact artifacts before recovery'
        result=json.loads((OUT/'development.json').read_text())
        assert not result['fresh_C13_prediction_metrics_read'] and not result['automatic_submission']
        report('complete','development qualified' if result['new_candidate_selected'] else 'fixed recipe rejected',
            selected=result['new_candidate_selected'],development_sha256=sha(OUT/'development.json'))
        print('DONE C28 DEVELOPMENT WORKER',flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,text=True)
        report('error','inspect before recovery',error_type=type(exc).__name__)
        write(OUT/'worker-attention.json',dict(error_type=type(exc).__name__,reason=str(exc),automatic_retry=False))
        raise


if __name__=='__main__':main()
