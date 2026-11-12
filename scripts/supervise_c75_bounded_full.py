"""One full512 original inference in the proven task-only memory scope.

The exact USED fixture parity is a prerequisite. Never evaluates quality,
submits, restarts, changes scientific settings or stops unrelated processes.
"""
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from c75_port_utils import ROOT, read, write, sha, stamp
from retry_c75_memory_bound import OUT, PY, LINUX_ROOT, LIMITS, wsl, monitor_scope
from supervise_c75_local_port import authorization, ram, trim_own_cache, stop_owned_group


def advise_completed_cache():
    code = '''import os
from pathlib import Path
root=Path('/home/imsa/.local/share/enveda-c75/fixture').resolve()
assert str(root)=='/home/imsa/.local/share/enveda-c75/fixture'
import json
owner=json.loads((root/'owner.json').read_text())
assert owner['owner']=='enveda-c75-local-core-port-20261112'
count=0
for name in ['working','attempt-01-host-memory-stop-working']:
 target=root/name
 assert target.resolve().parent==root and not target.is_symlink()
 for p in target.rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  try:
   fd=os.open(p,os.O_RDONLY)
   try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
   finally:os.close(fd)
   count+=1
  except OSError:pass
print(count)
'''
    return wsl(PY, '-c', code, timeout=60).stdout.strip()


def main():
    authorization()
    target = OUT / 'full-port'
    parity = read(OUT / 'fixture-parity-audit.json')
    assert parity['passed'] and parity['source_and_full_flow_audited']
    assert read(OUT / 'fixture-port/supervisor-complete.json')['inference_launches'] == 1
    audit = read(target / 'port-source-audit.json')
    assert audit['source_and_fixture_audit_passed'] and audit['port_freeze_sha256'] == sha(target / 'port-freeze.json')
    assert read(target / 'port-freeze.json')['queries'] == 512
    assert not (target / 'supervisor-intent.json').exists() and not (target / 'run-intent.json').exists()
    assert read(OUT / 'memory-scope-probe.json')['passed']
    import msvcrt
    lock = (OUT / 'supervisor.lock').open('a+b')
    lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    advised = advise_completed_cache()
    trim_own_cache(); time.sleep(5)
    free = ram(); assert free >= 12, ('Startup host headroom insufficient for full bounded inference', free)
    deadline = datetime.now(timezone.utc) + timedelta(hours=14)
    unit = 'enveda-c75-memory-bound-full-' + str(os.getpid())
    command = ['wsl.exe', '-d', 'Ubuntu-24.04', '--', 'systemd-run', '--user',
               '--scope', '--unit='+unit, *LIMITS, PY,
               LINUX_ROOT+'/scripts/run_c75_local_port.py', 'full']
    write(target / 'supervisor-intent.json', dict(started_utc=stamp(), pid=os.getpid(),
        source_sha256=sha(__file__), resource_helper_sha256=sha(ROOT / 'scripts/retry_c75_memory_bound.py'),
        mode='full', unit=unit, deadline_utc=deadline.isoformat(), inference_launches=1,
        task_memory_high_GiB=8, task_memory_max_GiB=10, task_swap_max_GiB=4,
        host_reserve_GiB=4, startup_host_free_GiB=free, completed_fixture_files_cache_advised=advised,
        original_driver_sha256=sha(ROOT / 'scripts/run_c75_local_port.py'),
        fixture_parity_audit_sha256=sha(OUT / 'fixture-parity-audit.json'),
        port_freeze_sha256=sha(target / 'port-freeze.json'),
        new_source_quality_read=False, C13_read=False, scoring_entries=0))
    with (target / 'local-pipeline.log').open('w', encoding='utf8') as stdout, \
         (target / 'local-pipeline.err').open('w', encoding='utf8') as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        try:
            while child.poll() is None:
                authorization(); assert datetime.now(timezone.utc) < deadline
                free = ram()
                if free < 4:
                    cache_files = trim_own_cache(); time.sleep(5); free = ram()
                    write(target / 'cache-pressure-action.json', dict(checked_utc=stamp(),
                        task_only_file_cache_advice=True, files=cache_files, host_free_after_GiB=free))
                    assert free >= 4, ('Host RAM reserve below4GiB after owned-cache advice', free)
                progress = read(target / 'run-progress.json') if (target / 'run-progress.json').exists() else None
                intent = read(target / 'run-intent.json') if (target / 'run-intent.json').exists() else None
                scope = monitor_scope(intent['pid']) if intent else None
                write(target / 'supervisor-status.json', dict(checked_utc=stamp(), pid=os.getpid(),
                    child_windows_pid=child.pid, mode='full', phase='memory_bounded_local_inference_running',
                    host_free_GiB=free, inference_launches=1, progress=progress, scope=scope,
                    new_source_quality_read=False, C13_read=False, scoring_entries=0))
                time.sleep(10)
            assert child.returncode == 0, ('Full local pipeline failed; reconcile without automatic retry', child.returncode)
            assert (target / 'run-complete.json').exists()
        except BaseException as exc:
            if child.poll() is None and (target / 'run-intent.json').exists():
                stop_owned_group(target); child.wait(timeout=30)
            write(target / 'supervisor-failed.json', dict(checked_utc=stamp(), source_sha256=sha(__file__),
                mode='full', error=type(exc).__name__+': '+str(exc)[:600], automatic_retry=False,
                new_source_quality_read=False, C13_read=False, scoring_entries=0))
            raise
    write(target / 'supervisor-complete.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        mode='full', inference_launches=1, run_complete_sha256=sha(target / 'run-complete.json'),
        independent_collection_health_and_acquisition_audits_required=True,
        new_source_quality_read=False, C13_read=False, scoring_entries=0))
    print('C75 WHOLE512 ORIGINAL PIPELINE FINISHED; AUDITS REQUIRED', flush=True)


if __name__ == '__main__': main()
