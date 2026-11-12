"""One resource-capped production preview; preserve the desktop RAM reserve."""
import os
import subprocess
import time
from c75_port_utils import ROOT, read, write, sha, stamp
from supervise_c75_local_port import authorization, ram, trim_own_cache


def main():
    out = ROOT / 'outputs/local/c76-numerical-portability-20261112/local-production-preview'
    authorization(); assert ram() >= 12 and not (out / 'run-intent.json').exists()
    assert not (out / 'supervisor-intent.json').exists()
    plan = read(out / 'launch-plan.json')
    for name, pin in plan['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    assert read(out / 'independent-source-audit.json')['passed']
    import msvcrt
    lock = (out / 'supervisor.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    unit = 'enveda-c76-production-preview-' + str(os.getpid()); wsl = ['wsl.exe', '-d', 'Ubuntu-24.04', '--']
    command = wsl + ['systemd-run', '--user', '--scope', '--unit=' + unit,
        '-p', 'MemoryHigh=8960M', '-p', 'MemoryMax=10G', '-p', 'MemorySwapMax=4G',
        '/usr/bin/env', 'PYTHONHASHSEED=0', 'CUDA_VISIBLE_DEVICES=0', 'PYTHONDONTWRITEBYTECODE=1',
        '/home/imsa/.venvs/enveda-c75-torch210/bin/python',
        '/mnt/c/Users/imsa/Documents/Enveda CASMI/scripts/run_c76_local_production.py']
    record = dict(started_utc=stamp(), pid=os.getpid(), supervisor_sha256=sha(__file__), unit=unit,
        launch_plan_sha256=sha(out / 'launch-plan.json'), launches=1, automatic_retry=False,
        host_reserve_GiB=4, memory_high_GiB=8.75, memory_max_GiB=10, swap_max_GiB=4,
        actual_single_2070=True, actual_T4_evidence=False, quality_read=False, C13_read=False, scoring_entries=0)
    write(out / 'supervisor-intent.json', record)
    with (out / 'pipeline.log').open('w', encoding='utf8') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 7200
        try:
            while child.poll() is None:
                authorization(); assert time.monotonic() < deadline
                free = ram()
                if free < 4:
                    trim_own_cache(); time.sleep(5); free = ram(); assert free >= 4, free
                write(out / 'supervisor-status.json', dict(record, checked_utc=stamp(), status='running', host_free_GiB=free))
                time.sleep(10)
            assert child.returncode == 0, child.returncode
            assert read(out / 'run-complete.json')['preview_queries'] == 12
        except BaseException as exc:
            if child.poll() is None:
                group = subprocess.check_output(wsl + ['systemctl', '--user', 'show', unit + '.scope', '-p', 'ControlGroup', '--value'], text=True).strip()
                assert group.endswith('/' + unit + '.scope'), group
                subprocess.run(wsl + ['systemctl', '--user', 'stop', unit + '.scope'], check=True, timeout=30)
                child.wait(timeout=30)
            write(out / 'supervisor-failed.json', dict(record, completed_utc=stamp(), status='failed', error=repr(exc))); raise
    write(out / 'supervisor-complete.json', dict(record, completed_utc=stamp(), status='complete', run_complete_sha256=sha(out / 'run-complete.json')))
    print('C76 OWNED LOCAL PRODUCTION PREVIEW SUPERVISOR COMPLETE', flush=True)


if __name__ == '__main__': main()
