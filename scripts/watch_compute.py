"""Collect these fixed Kaggle training sessions once complete; no submission calls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('slots',nargs='+')
    ap.add_argument('--interval',type=int,default=180)
    args=ap.parse_args()
    assert args.interval>=60
    assert all(re.fullmatch(r'[a-z0-9-]+',s) for s in args.slots)
    target=ROOT/'outputs/local/professional/compute_watch.json'
    states={slot:dict(status='unknown',fetched=False) for slot in args.slots}
    while True:
        for slot,state in states.items():
            if state['fetched'] or state['status'] in ('error','cancelled'):
                continue
            result=subprocess.run([sys.executable,'-m','kaggle','kernels','status','clarkkitchen/casmi26-'+slot],capture_output=True,text=True,timeout=120)
            match=re.search(r'KernelWorkerStatus\.(\w+)',result.stdout+result.stderr)
            status=match.group(1).lower() if match else 'status_unavailable'
            if status!=state['status']:
                print(datetime.now(timezone.utc).isoformat(),slot,status,flush=True)
            state['status']=status
            if status=='complete':
                destination=ROOT/'outputs/forks'/slot
                pattern=r'\.pt$|_report\.json$|manifest|audit|\.log$|np_holdout_keys\.txt$|native_rows\.npy$'
                fetch=subprocess.run([sys.executable,str(ROOT/'scripts/fetch_output.py'),slot,str(destination),pattern],capture_output=True,text=True,timeout=600)
                print(fetch.stdout[-900:],fetch.stderr[-300:],flush=True)
                state['fetched']=fetch.returncode==0 and len(list(destination.rglob('*.pt')))>=2
                state['model_files']=[str(p.relative_to(destination)) for p in destination.rglob('*.pt')]
                state['fetch_returncode']=fetch.returncode
        report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),sessions=states)
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2))
        temporary.replace(target)
        if all(s['fetched'] or s['status'] in ('error','cancelled') for s in states.values()):
            assert all(s['fetched'] for s in states.values()),states
            print('DONE all requested training outputs collected',flush=True)
            return
        time.sleep(args.interval)


if __name__=='__main__':
    main()
