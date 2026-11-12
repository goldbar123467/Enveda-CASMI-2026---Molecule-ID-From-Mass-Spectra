"""One-shot local validation supervisor; wait for live asset jobs, then one run.

Never submits, opens quality labels, changes a cohort or restarts inference.
Enforce the host RAM reserve and terminate only the verified owned Linux PGID.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
from c75_port_utils import ROOT,read,write,sha,stamp
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'
WSL_PY='/home/imsa/.venvs/enveda-c75/bin/python'
LINUX_ROOT='/mnt/c/Users/imsa/Documents/Enveda CASMI'


def ram():
    r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command','(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],check=True,capture_output=True,text=True,timeout=30)
    return int(r.stdout.strip())/2**20


def authorization():
    a=read(ROOT/'.local/c38-active-goal.json');assert a['active'] and a['free_local_and_kaggle_compute_authorized']
    assert a['submissions_used']==0 and a['scientific_gate_hold']
    old=ROOT/'.local/compute-paused.json'
    if old.exists():assert datetime.fromisoformat(read(old)['paused_utc'])<datetime.fromisoformat(a['authorized_utc'])
    return a


def trim_own_cache():
    code="""import os
from pathlib import Path
roots=[Path('/home/imsa/.venvs/enveda-c75'),Path('/mnt/c/Users/imsa/Documents/Enveda CASMI/outputs/local/c75-local-core-port-20261112/input')]
n=0
for root in roots:
 for p in root.rglob('*'):
  if p.is_file() and not p.is_symlink():
   try:
    fd=os.open(p,os.O_RDONLY)
    try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)
    n+=1
   except OSError:pass
print(n)
"""
    return subprocess.run(['wsl.exe','-d','Ubuntu-24.04','--','python3','-c',code],check=True,capture_output=True,text=True,timeout=60).stdout.strip()


def stop_owned_group(target):
    intent=read(target/'run-intent.json');pid=int(intent['pid']);assert pid>1 and pid==int(intent['pgid'])
    code="""import os,signal,sys
from pathlib import Path
pid=int(sys.argv[1]);cmd=Path('/proc')/str(pid)/'cmdline'
if not cmd.exists():sys.exit(0)
text=cmd.read_bytes().replace(b'\\0',b' ')
assert b'run_c75_local_port.py' in text and b'enveda-c75/bin/python' in text
assert os.getpgid(pid)==pid
os.killpg(pid,signal.SIGTERM)
print('Stopped verified task process group',pid)
"""
    return subprocess.run(['wsl.exe','-d','Ubuntu-24.04','--','python3','-c',code,str(pid)],check=True,capture_output=True,text=True,timeout=30).stdout.strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fixture','full']);args=parser.parse_args();mode=args.mode
    target=OUT/(mode+'-port');assert not (target/'supervisor-intent.json').exists(),'Reconcile the existing supervisor; no duplicate run'
    authorization();assert read(target/'port-source-audit.json')['source_and_fixture_audit_passed']
    import msvcrt
    lock=(OUT/'supervisor.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    deadline=datetime.now(timezone.utc)+timedelta(hours=6 if mode=='fixture' else 14)
    write(target/'supervisor-intent.json',dict(started_utc=stamp(),pid=os.getpid(),source_sha256=sha(__file__),mode=mode,deadline_utc=deadline.isoformat(),
        wait_for_complete_assets=True,exactly_one_local_inference_launch=True,host_reserve_GiB=4,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    while not all((OUT/n).exists() for n in ['v4g-asset-freeze.json','aux-asset-freeze.json']):
        authorization();assert datetime.now(timezone.utc)<deadline,'Asset wait deadline ended; inspect live handles, do not restart blindly'
        write(target/'supervisor-status.json',dict(checked_utc=stamp(),pid=os.getpid(),phase='waiting_for_asset_freezes',host_free_GiB=ram(),mode=mode,
            inference_launches=0,new_source_quality_read=False,C13_read=False,scoring_entries=0))
        time.sleep(15)
    authorization();free=ram();assert free>=8,('Insufficient startup reserve',free)
    assert not (target/'run-intent.json').exists()
    log=target/'local-pipeline.log';error=target/'local-pipeline.err'
    command=['wsl.exe','-d','Ubuntu-24.04','--',WSL_PY,LINUX_ROOT+'/scripts/run_c75_local_port.py',mode]
    with log.open('w',encoding='utf8') as stdout,error.open('w',encoding='utf8') as stderr:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        try:
            while child.poll() is None:
                authorization();assert datetime.now(timezone.utc)<deadline,'Owned inference wall-clock limit ended'
                free=ram()
                if free<4:
                    advised=trim_own_cache();time.sleep(5);free=ram()
                    write(target/'cache-pressure-action.json',dict(checked_utc=stamp(),task_only_file_cache_advice=True,files=advised,host_free_after_GiB=free))
                    assert free>=4,('Host RAM reserve below4GiB after owned-cache advice',free)
                progress=read(target/'run-progress.json') if (target/'run-progress.json').exists() else None
                write(target/'supervisor-status.json',dict(checked_utc=stamp(),pid=os.getpid(),child_windows_pid=child.pid,phase='local_inference_running',mode=mode,
                    host_free_GiB=free,inference_launches=1,progress=progress,new_source_quality_read=False,C13_read=False,scoring_entries=0))
                time.sleep(10)
            assert child.returncode==0,('Local pipeline returned failure; inspect original log',child.returncode)
            assert (target/'run-complete.json').exists()
        except BaseException as exc:
            if child.poll() is None and (target/'run-intent.json').exists():
                stopped=stop_owned_group(target)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:raise RuntimeError('Owned Linux group termination needs inspection') from None
            write(target/'supervisor-failed.json',dict(checked_utc=stamp(),source_sha256=sha(__file__),mode=mode,error=type(exc).__name__+': '+str(exc)[:500],
                automatic_retry=False,new_source_quality_read=False,C13_read=False,scoring_entries=0))
            raise
    write(target/'supervisor-complete.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),mode=mode,inference_launches=1,
        run_complete_sha256=sha(target/'run-complete.json'),independent_parity_or_collection_audit_required=True,new_source_quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 ONE OWNED LOCAL PIPELINE FINISHED; independent audits required',mode,flush=True)


if __name__=='__main__':main()
