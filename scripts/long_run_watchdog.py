"""Bounded 30-minute supervisor for L06; never edits pinned model/training code.

Recovers dead/stalled processes from complete-epoch checkpoints, waits for RAM/VRAM
headroom, and stops on changed provenance, repeated deterministic errors or pause.
Submission consent is preserved for post-training validation; it is not a scientific
qualification receipt and does not bypass the real Kaggle smoke requirement.
"""
import argparse
import csv
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/local/long-20261106'
STATE=RUN/'watchdog'
SETTINGS=ROOT/'configs/long-watchdog-20261106.json'


def now():return datetime.now(timezone.utc)


def parse_time(value):
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def read_json(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError):return default


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def inventory():
    query="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine,@{n='created_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',query],capture_output=True,text=True,timeout=30)
    if result.returncode:raise RuntimeError('Process inventory failed: '+result.stderr[-500:])
    data=json.loads(result.stdout) if result.stdout.strip() else []
    return data if isinstance(data,list) else [data]


def owned_trainers(processes):
    interpreter=str(ROOT/'.venv-gpu/Scripts/python.exe').replace('/','\\').lower()
    def is_training(p):
        cmd=(p.get('CommandLine') or '').replace('/','\\').lower()
        return bool(re.search(r'\\long_training\.py(?:\s|"|$)',cmd)) and '--smoke' not in cmd
    owners={p['ProcessId'] for p in processes if is_training(p) and interpreter in (p.get('CommandLine') or '').replace('/','\\').lower()}
    matched=[p for p in processes if is_training(p) and (p['ProcessId'] in owners or p['ParentProcessId'] in owners)]
    return matched


def decide(status,processes,clock,stale_seconds,paused,deadline):
    if paused:return 'paused'
    if status and status.get('state')=='complete':return 'complete'
    if clock>=deadline:return 'budget_reached'
    if not processes:return 'restart'
    if not status:return 'attention_missing_status'
    age=(clock-parse_time(status['updated_utc'])).total_seconds()
    return 'restart_stalled' if age>stale_seconds else 'healthy'


def memory():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
                  ['totalPhysical','availablePhysical','totalPage','availablePage','totalVirtual','availableVirtual','availableExtended']]
    mem=Memory();mem.length=ctypes.sizeof(mem)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):raise RuntimeError('RAM query failed')
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
    if result.returncode:raise RuntimeError('GPU memory query failed')
    return mem.availablePhysical/2**30,float(result.stdout.splitlines()[0])/1024


def provenance_error():
    manifest=read_json(RUN/'manifest.json')
    if not manifest or manifest.get('smoke'):return 'Missing full-run manifest'
    spec=read_json(ROOT/'configs/long-training-20261106.json')
    if manifest['spec']!=spec:return 'Training plan changed'
    for relative,expected in manifest['source_sha256'].items():
        path=ROOT/relative
        if not path.is_file() or file_hash(path)!=expected:return 'Pinned source changed: '+relative
    for path,expected in manifest['fingerprint_sha256'].items():
        if not Path(path).is_file() or file_hash(path)!=expected:return 'Fingerprint cache changed'
    if file_hash(RUN/'spectrum_groups.npz')!=manifest['spectrum_group_sha256']:return 'Split groups changed'
    data=read_json(Path(manifest['cfg']['external_cache'])/'manifest.json')
    if not data:return 'External data manifest missing'
    return None


def eta_snapshot():
    epochs=0;seconds=[];complete=0
    for folder in (RUN/'models').glob('*/fold-*'):
        history=read_json(folder/'history.json',[])
        epochs+=len(history);seconds.extend(h['seconds'] for h in history[-20:])
        complete+=int((folder/'complete.json').exists())
    if not seconds:return dict(completed_epochs=epochs,completed_models=complete)
    pace=sum(seconds)/len(seconds)
    remaining=(1280-epochs)*pace+(10-complete)*45
    return dict(completed_epochs=epochs,total_epochs=1280,completed_models=complete,
                seconds_per_epoch=round(pace,2),estimated_training_complete_utc=(now()+timedelta(seconds=remaining)).isoformat())


def ledger_running(message):
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    matches=[r for r in rows if r['id']=='L06'];assert len(matches)==1
    matches[0].update(status='running',ended_utc='',result=message)
    temp=path.with_suffix('.watchdog.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def stop_owned(processes):
    # Only the exact project interpreter wrapper and its training child qualify.
    current={p['ProcessId']:p for p in owned_trainers(inventory())}
    for p in sorted(processes,key=lambda p:p['ParentProcessId'] not in current):
        pid=p['ProcessId']
        if pid not in current:continue
        if current[pid]['CommandLine']!=p['CommandLine'] or current[pid]['created_utc']!=p['created_utc']:
            raise RuntimeError('Process identity changed; refusing to terminate PID')
        try:os.kill(pid,signal.SIGTERM)
        except OSError:
            if any(p['ProcessId']==pid for p in owned_trainers(inventory())):raise
    time.sleep(2)
    if owned_trainers(inventory()):raise RuntimeError('Training process remains; refusing duplicate launch')


def checkpoint_ready():
    status=read_json(RUN/'status.json',{})
    if 'family' not in status or 'fold' not in status:return False
    folder=RUN/'models'/status['family']/('fold-'+str(status['fold']))
    # A newly started fold can be safely rebuilt; completed folds are receipt-checked.
    return (folder/'resume.pt').is_file() or (folder/'complete.json').is_file() or status.get('epoch',0)==0


def restart(processes,settings,state):
    problem=provenance_error()
    if problem:return dict(action='needs_attention',reason=problem)
    ram,gpu=memory()
    if ram<settings['restart_minimum_free_ram_gib'] or gpu<settings['restart_minimum_free_vram_gib']:
        return dict(action='waiting_for_memory',free_ram_gib=ram,free_vram_gib=gpu)
    err=(RUN/'train.err').read_text(encoding='utf-8',errors='replace') if (RUN/'train.err').exists() else ''
    signature=hashlib.sha256(err[-4000:].encode()).hexdigest() if err.strip() else None
    if signature and signature==state.get('last_error_signature'):
        return dict(action='needs_attention',reason='Same deterministic error recurred; scientific code repair/revalidation required')
    if state.get('restarts',0)>=settings['maximum_restarts']:
        return dict(action='needs_attention',reason='Restart limit reached')
    if not checkpoint_ready():return dict(action='needs_attention',reason='No verified completed-epoch recovery point')
    if processes:stop_owned(processes)
    # Copy failed logs before opening new logs; no model or data artifact is deleted.
    number=state.get('restarts',0)+1
    for name in ['train.log','train.err']:
        source=RUN/name
        if source.exists():shutil.copy2(source,STATE/f'restart-{number}-{name}')
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (RUN/'train.log').open('w',encoding='utf-8') as stdout,(RUN/'train.err').open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen([str(ROOT/'.venv-gpu/Scripts/python.exe'),'-u',str(ROOT/'scripts/long_training.py')],cwd=ROOT,
                               env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    state.update(restarts=number,last_error_signature=signature,last_restart_utc=now().isoformat(),launcher_pid=child.pid)
    write_json(STATE/'recovery.json',state)
    ledger_running(f'Watchdog restart{number}, PID{child.pid}: complete-epoch optimizer/scaler/RNG recovery; pinned code/data unchanged.')
    return dict(action='restarted',restart=number,launcher_pid=child.pid)


def check(settings,dry_run=False):
    status=read_json(RUN/'status.json')
    processes=owned_trainers(inventory())
    state=read_json(STATE/'recovery.json',dict(restarts=0))
    choice=decide(status,processes,now(),settings['stale_after_minutes']*60,
                  (ROOT/'.local/compute-paused.json').exists(),parse_time(settings['deadline_utc']))
    result=dict(action=choice,checked_utc=now().isoformat(),pid=os.getpid(),
                trainer_pids=[p['ProcessId'] for p in processes],progress=status,eta=eta_snapshot(),
                restarts=state.get('restarts',0),submission_authorized=settings['submission_approved'])
    if choice in {'restart','restart_stalled'} and not dry_run:
        result.update(restart(processes,settings,state))
    if choice=='budget_reached' and processes and not dry_run:
        stop_owned(processes);result['reason']='Global original-run 12h budget reached; last completed epoch retained'
    if choice=='complete':
        result['next_action']='Training complete. Submission approved, pending scientific/production packaging and real Kaggle smoke; no additional user approval needed.'
        write_json(STATE/'completion-review.json',result)
    result['next_check_utc']=(now()+timedelta(seconds=settings['interval_seconds'])).isoformat()
    write_json(STATE/'status.json',result)
    with (STATE/'checks.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();settings=read_json(SETTINGS);assert settings and settings['interval_seconds']==1800
    STATE.mkdir(parents=True,exist_ok=True)
    # OS lock releases on exit, allowing a failed watchdog to be restarted safely.
    import msvcrt
    lock=(STATE/'supervisor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        try:result=check(settings,args.dry_run)
        except Exception as exc:
            result=dict(action='needs_attention',error=repr(exc),checked_utc=now().isoformat())
            write_json(STATE/'status.json',result);print(json.dumps(result),flush=True)
        if args.once or result['action'] in {'complete','paused','budget_reached','needs_attention','attention_missing_status'}:break
        time.sleep(settings['interval_seconds'])


if __name__=='__main__':main()
