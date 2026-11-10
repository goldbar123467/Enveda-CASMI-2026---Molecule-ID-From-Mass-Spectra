"""Bounded C24 scoring/global selection after the existing C21 worker exits."""
import ctypes,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from precision_cache import sha,write
OUT=ROOT/'outputs/local/c24-evaluation-20261110'
C21=ROOT/'outputs/local/c21-negative-tail-20261110'


def stamp():return datetime.now(timezone.utc).isoformat()


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


def main():
    import msvcrt
    lock=(OUT/'worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'worker-intent.json').exists(),'Inspect an existing worker before recovery'
    deadline=datetime(2026,11,10,21,0,tzinfo=timezone.utc)
    protocol=OUT/'development-protocol.json';pin=sha(protocol)
    sources={p:sha(ROOT/'scripts'/p) for p in ['run_c24_evaluation_worker.py','develop_c24_negative.py','c24_models.py']}
    write(OUT/'worker-intent.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),
        source_sha256=sources,protocol_sha256=pin,C13_confirmation_metrics_read=False,automatic_submission=False))
    def verify():
        assert datetime.now(timezone.utc)<deadline,'Bounded observation ended; inspect the same jobs before recovery'
        assert sha(protocol)==pin
        for p,digest in sources.items():assert sha(ROOT/'scripts'/p)==digest,p
        assert not (ROOT/'.local/compute-paused.json').exists()
    def state(stage,**values):
        write(OUT/'worker-status.json',dict(checked_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),
            stage=stage,C13_confirmation_metrics_read=False,automatic_submission=False,**values))
    while True:
        verify();upstream=json.loads((C21/'worker-status-v2.json').read_text());live=alive(upstream['pid'])
        if upstream.get('state')=='complete' and not live:break
        assert live,'C21 worker exited without complete receipt; inspect exact upstream handles'
        state('waiting-C21',state='waiting',C21_pid=upstream['pid'],C21_process_live=live,C21_stage=upstream.get('stage'))
        time.sleep(20)
    for stage in ['score','evaluate','select']:
        verify();state(stage,state='running')
        with (OUT/(stage+'.log')).open('w',encoding='utf-8') as stdout,(OUT/(stage+'.err')).open('w',encoding='utf-8') as stderr:
            result=subprocess.run([sys.executable,'scripts/develop_c24_negative.py',stage],cwd=ROOT,stdout=stdout,stderr=stderr,timeout=3600)
        assert result.returncode==0,'C24 '+stage+' failed; inspect before recovery'
    result=json.loads((OUT/'global-selection-freeze.json').read_text());state('complete',state='complete',selected_group=result['selected_group'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'worker-attention.json',dict(checked_utc=stamp(),pid=os.getpid(),error=type(error).__name__+': '+str(error),
            automatic_recovery=False,automatic_submission=False,C13_confirmation_metrics_read=False))
        raise
