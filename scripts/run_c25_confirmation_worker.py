"""Bounded once-only confirmation after the actual six-group selector exits."""
import ctypes,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from precision_cache import sha,write
OUT=ROOT/'outputs/local/c25-reserved-negative-20261110'
UPSTREAM=ROOT/'outputs/local/c24-evaluation-20261110'


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
    assert not (OUT/'worker-intent.json').exists(),'Inspect existing worker before any recovery'
    assert (OUT/'preparation.json').exists() and (OUT/'control-receipt.json').exists()
    deadline=datetime(2026,11,10,21,40,tzinfo=timezone.utc)
    pins={n:sha(ROOT/'scripts'/n) for n in ['run_c25_confirmation_worker.py','confirm_c25_negative.py','prepare_c25_reserved_inputs.py','c14_confirmation_gate.py']}
    preparation=sha(OUT/'preparation.json');control=sha(OUT/'control-receipt.json')
    write(OUT/'worker-intent.json',dict(started_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),
        source_sha256=pins,preparation_sha256=preparation,control_receipt_sha256=control,
        C13_confirmation_metrics_read=False,automatic_submission=False))
    def verify():
        assert datetime.now(timezone.utc)<deadline,'Bounded observation ended; inspect same handles before recovery'
        assert sha(OUT/'preparation.json')==preparation and sha(OUT/'control-receipt.json')==control
        for n,pin in pins.items():assert sha(ROOT/'scripts'/n)==pin,n
        assert not (ROOT/'.local/compute-paused.json').exists()
    def state(stage,**values):
        write(OUT/'worker-status.json',dict(checked_utc=stamp(),pid=os.getpid(),deadline_utc=deadline.isoformat(),stage=stage,automatic_submission=False,**values))
    while True:
        verify();upstream=json.loads((UPSTREAM/'worker-status.json').read_text());live=alive(upstream['pid'])
        if upstream.get('state')=='complete' and not live:break
        assert live,'Global selector exited without complete receipt; inspect exact upstream handle'
        state('waiting-global-selection',state='waiting',C24_worker_pid=upstream['pid'],C24_process_live=live,C24_stage=upstream.get('stage'),C13_confirmation_metrics_read=False)
        time.sleep(20)
    selection=json.loads((UPSTREAM/'global-selection-freeze.json').read_text())
    if not selection['new_candidate_selected']:
        assert selection['selected_group'] is None
        state('all-six-rejected',state='complete',selected_group=None,C13_confirmation_metrics_read=False,reserved120_preserved=True)
        return
    for stage in ['freeze','score','evaluate']:
        verify();state(stage,state='running',C13_confirmation_metrics_read=False)
        with (OUT/(stage+'.log')).open('w',encoding='utf-8') as stdout,(OUT/(stage+'.err')).open('w',encoding='utf-8') as stderr:
            result=subprocess.run([str(ROOT/'.venv-gpu/Scripts/python.exe'),'scripts/confirm_c25_negative.py',stage],cwd=ROOT,stdout=stdout,stderr=stderr,timeout=3600)
        assert result.returncode==0,'C25 '+stage+' failed; inspect consumption marker and logs, never reselect or blindly repeat'
    report=json.loads((OUT/'confirmation.json').read_text())
    state('complete',state='complete',selected_group=report['selected_group'],C13_confirmation_metrics_read=True,
        confirmation_passed=report['statistical_gate']['passed'],submission_qualified=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'worker-attention.json',dict(checked_utc=stamp(),pid=os.getpid(),error=type(error).__name__+': '+str(error),
            consumption_marker_exists=(ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists(),
            automatic_retry=False,automatic_submission=False))
        raise
