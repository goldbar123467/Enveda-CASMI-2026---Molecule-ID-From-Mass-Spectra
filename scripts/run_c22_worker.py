"""Single local C22 train/development worker; no upload, fresh metrics or submit."""
import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write

OUT=ROOT/'outputs/local/c22-peak-dose-20261110'


def main():
    import msvcrt
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'worker-intent.json').exists(),'Inspect the previous worker; never start a duplicate or refit'
    paths=['scripts/run_c22_worker.py','scripts/c22_peak_train.py','scripts/peak_dropout.py','scripts/develop_c22_peak.py',
        'scripts/c22_models.py','configs/c22-peak-dose-20261110.json','configs/c22-evaluation-20261110.json']
    pins={p:sha(ROOT/p) for p in paths}
    write(OUT/'worker-intent.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        source_sha256=pins,automatic_submission=False,fresh_C13_prediction_metrics_read=False))
    stages=[('training',['scripts/c22_peak_train.py']),('prepare-development',['scripts/develop_c22_peak.py','--prepare-protocol']),
        ('development',['scripts/develop_c22_peak.py','--evaluate'])]
    for stage,args in stages:
        for p,pin in pins.items():assert sha(ROOT/p)==pin,p
        if stage=='training':assert not (OUT/'results.json').exists() and not (OUT/'launch.json').exists()
        if stage=='prepare-development':assert (OUT/'results.json').exists()
        write(OUT/'worker-status.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=stage,state='running',
            automatic_submission=False,fresh_C13_prediction_metrics_read=False))
        with (OUT/(stage+'.log')).open('w',encoding='utf-8') as stdout,(OUT/(stage+'.err')).open('w',encoding='utf-8') as stderr:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stdout,stderr=stderr)
        if result.returncode:
            write(OUT/'worker-status.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),stage=stage,state='error',
                exit_code=result.returncode,automatic_submission=False,fresh_C13_prediction_metrics_read=False))
            raise RuntimeError('C22 stage failed: '+stage)
    report=json.loads((ROOT/'outputs/local/c22-evaluation-20261110/development.json').read_text())
    write(OUT/'worker-status.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),state='complete',
        new_candidate_selected=report['new_candidate_selected'],automatic_submission=False,fresh_C13_prediction_metrics_read=False))


if __name__=='__main__':main()
