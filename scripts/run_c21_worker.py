"""Bounded C21 preparation/selection after existing C20/C22 handles finish."""
import ctypes,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from precision_cache import sha,write
OUT=ROOT/'outputs/local/c21-negative-tail-20261110'
C20=ROOT/'outputs/local/c20-negative-development-20261110'
C22=ROOT/'outputs/local/c22-peak-dose-20261110'


def alive(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    try:
        code=ctypes.c_ulong();assert kernel.GetExitCodeProcess(handle,ctypes.byref(code))
        return code.value==259
    finally:kernel.CloseHandle(handle)


def stamp():return datetime.now(timezone.utc).isoformat()


def main():
    import msvcrt
    lock=(OUT/'worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'worker-intent.json').exists(),'Inspect an existing worker before any recovery'
    protocol=OUT/'development-protocol.json';assert protocol.exists()
    pins={p:sha(ROOT/p) for p in ['scripts/run_c21_worker.py','scripts/develop_c21_negative.py','scripts/negative_tail.py']}
    protocol_pin=sha(protocol)
    push=json.loads((C20/'push-receipt.json').read_text());deadline=datetime.fromisoformat(push['checked_utc'])+timedelta(hours=10,minutes=30)
    write(OUT/'worker-intent.json',dict(started_utc=stamp(),pid=os.getpid(),source_sha256=pins,
        protocol_sha256=protocol_pin,deadline_utc=deadline.isoformat(),C13_confirmation_metrics_read=False,automatic_submission=False))
    def state(stage,**values):
        write(OUT/'worker-status.json',dict(checked_utc=stamp(),pid=os.getpid(),stage=stage,deadline_utc=deadline.isoformat(),
            C13_confirmation_metrics_read=False,automatic_submission=False,**values))
    def verify():
        assert sha(protocol)==protocol_pin
        for p,pin in pins.items():assert sha(ROOT/p)==pin,p
        assert datetime.now(timezone.utc)<deadline,'Bounded observation ended; inspect same jobs before any recovery'
        assert not (ROOT/'.local/compute-paused.json').exists(),'Project compute pause requested'
    def run(stage):
        verify();state(stage,state='running')
        with (OUT/(stage+'.log')).open('w',encoding='utf-8') as stdout,(OUT/(stage+'.err')).open('w',encoding='utf-8') as stderr:
            r=subprocess.run([sys.executable,'scripts/develop_c21_negative.py',stage],cwd=ROOT,stdout=stdout,stderr=stderr,timeout=3600)
        assert r.returncode==0,'C21 '+stage+' failed; inspect logs, never blindly repeat'
    while True:
        verify();collector=json.loads((C20/'collector-v1-status.json').read_text())
        if collector.get('state')=='collected':break
        process_alive=alive(collector['pid']);state('waiting-C20',state='waiting',collector_pid=collector['pid'],collector_process_live=process_alive,cloud_state=collector.get('state','unknown'))
        assert process_alive,'C20 collector handle missing; poll the same Kaggle kernel before any recovery. Cloud termination is not inferred.'
        time.sleep(20)
    run('prepare')
    while True:
        verify();worker=json.loads((C22/'worker-status.json').read_text());process_alive=alive(worker['pid'])
        if worker.get('state') in ['complete','error'] and not process_alive:break
        state('waiting-C22-GPU',state='waiting',C22_worker_pid=worker['pid'],C22_worker_process_live=process_alive,C22_state=worker.get('state','unknown'))
        assert process_alive,'C22 handle missing without a terminal worker receipt; inspect before sharing the GPU'
        time.sleep(20)
    run('score');run('evaluate')
    result=json.loads((OUT/'development.json').read_text());state('complete',state='complete',selected_group=result['selected_group'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'worker-attention.json',dict(checked_utc=stamp(),pid=os.getpid(),error=type(error).__name__+': '+str(error),
            job_termination_inferred=False,automatic_recovery=False,automatic_submission=False,C13_confirmation_metrics_read=False))
        raise
