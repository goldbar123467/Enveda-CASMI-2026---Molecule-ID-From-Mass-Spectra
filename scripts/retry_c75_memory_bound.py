"""One reconciled C75 engineering retry in an inherited Linux memory scope.

Preserve the failed attempt and the frozen scientific sources. Never evaluate
quality, submit, kill other applications, or change system-wide memory policy.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from c75_port_utils import ROOT, read, write, sha, stamp
from supervise_c75_local_port import authorization, ram, trim_own_cache, stop_owned_group

OUT = ROOT / 'outputs/local/c75-local-core-port-20261112'
PY = '/home/imsa/.venvs/enveda-c75/bin/python'
LINUX_ROOT = '/mnt/c/Users/imsa/Documents/Enveda CASMI'
LIMITS = ['-p', 'MemoryHigh=8G', '-p', 'MemoryMax=10G', '-p', 'MemorySwapMax=4G']


def wsl(*args, **kwargs):
    return subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '--', *args],
                          check=True, capture_output=True, text=True, **kwargs)


def probe():
    code = '''import json,subprocess,sys
from pathlib import Path
relative=Path('/proc/self/cgroup').read_text().strip().split('::')[1]
root=Path('/sys/fs/cgroup')/relative.lstrip('/')
limits={n:(root/n).read_text().strip() for n in ['memory.high','memory.max','memory.swap.max']}
data=bytearray(64*2**20)
child=subprocess.check_output([sys.executable,'-c',"from pathlib import Path;print(Path('/proc/self/cgroup').read_text().strip())"],text=True).strip()
assert child=='0::'+relative
print(json.dumps(dict(cgroup=relative,limits=limits,child_inherits_scope=True,allocation_bytes=len(data),memory_current=int((root/'memory.current').read_text()))))
'''
    r = wsl('systemd-run', '--user', '--scope', '--unit=enveda-c75-bounded-probe',
            *LIMITS, PY, '-c', code, timeout=45)
    result = json.loads(r.stdout)
    assert result['limits'] == {'memory.high': str(8*2**30),
                               'memory.max': str(10*2**30),
                               'memory.swap.max': str(4*2**30)}
    assert result['memory_current'] >= 64*2**20
    path = OUT / 'memory-scope-probe.json'
    assert not path.exists(), 'Preserve the original resource probe'
    write(path, dict(checked_utc=stamp(), source_sha256=sha(__file__), **result,
                     task_only=True, system_wide_policy_changed=False,
                     scientific_sources_changed=False, passed=True))
    print('C75 INHERITED TASK MEMORY LIMITS VERIFIED', flush=True)


def archive_failed():
    target = OUT / 'fixture-port'
    failed = read(target / 'supervisor-failed.json')
    assert 'Host RAM reserve below4GiB' in failed['error']
    assert not (target / 'run-complete.json').exists()
    assert not (target / 'attempt-01-host-memory-stop').exists()
    original = read(target / 'run-intent.json')
    assert original['mode'] == 'fixture' and original['pid'] == original['pgid']
    assert sha(ROOT / 'scripts/run_c75_local_port.py') == original['source_sha256']
    assert read(OUT / 'memory-scope-probe.json')['passed']
    code = '''import json,os,sys
from pathlib import Path
pid=int(sys.argv[1]);p=Path('/proc')/str(pid)
if p.exists():assert p.joinpath('stat').read_text().split()[2]=='Z', 'Original driver still live'
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:args=process.joinpath('cmdline').read_bytes().split(b'\\0')
 except OSError:continue
 assert not any(a.endswith(b'/scripts/run_c75_local_port.py') for a in args), ('Another C75 driver live',process.name)
base=Path('/home/imsa/.local/share/enveda-c75/fixture').resolve()
assert str(base)=='/home/imsa/.local/share/enveda-c75/fixture'
working=base/'working';archived=base/'attempt-01-host-memory-stop-working'
assert working.is_dir() and not working.is_symlink() and not archived.exists()
assert working.resolve().parent==base and archived.resolve().parent==base
assert Path('/kaggle/working').resolve()==working
working.rename(archived);working.mkdir()
print(json.dumps(dict(original_driver_stopped=True,working_preserved_at=str(archived),fresh_working=str(working),only_verified_task_directory_moved=True)))
'''
    result = json.loads(wsl(PY, '-c', code, str(original['pid']), timeout=30).stdout)
    names = ['cache-pressure-action.json', 'local-pipeline.err', 'local-pipeline.log',
             'run-intent.json', 'run-progress.json', 'supervisor-failed.json',
             'supervisor-intent.json', 'supervisor-status.json']
    existing = [target / name for name in names if (target / name).exists()]
    hashes = {p.name: sha(p) for p in existing}
    archive = target / 'attempt-01-host-memory-stop'
    archive.mkdir()
    for path in existing:
        assert path.resolve().parent == target.resolve()
        dest = archive / path.name
        assert dest.resolve().parent == archive.resolve() and not dest.exists()
        path.rename(dest)
        assert sha(dest) == hashes[path.name]
    write(archive / 'preservation-receipt.json', dict(archived_utc=stamp(),
        source_sha256=sha(__file__), files_sha256=hashes, **result,
        failed_attempt_not_qualified=True, quality_outcomes_read=False, C13_read=False,
        scoring_entries=0, notebook_and_models_unchanged=True))
    print('C75 FAILED ATTEMPT PRESERVED; FRESH WORKSPACE READY', flush=True)


def monitor_scope(pid):
    code = '''import json,sys
from pathlib import Path
proc=Path('/proc')/sys.argv[1]
if not proc.exists():print('{}');sys.exit(0)
relative=(proc/'cgroup').read_text().strip().split('::')[1]
assert 'enveda-c75-memory-bound-' in relative,relative
root=Path('/sys/fs/cgroup')/relative.lstrip('/')
limits={n:(root/n).read_text().strip() for n in ['memory.high','memory.max','memory.swap.max']}
assert limits=={'memory.high':str(8*2**30),'memory.max':str(10*2**30),'memory.swap.max':str(4*2**30)}
events={line.split()[0]:int(line.split()[1]) for line in (root/'memory.events').read_text().splitlines()}
assert events['oom']==0 and events['oom_kill']==0,events
print(json.dumps(dict(cgroup=relative,limits=limits,memory_current=int((root/'memory.current').read_text()),memory_swap_current=int((root/'memory.swap.current').read_text()),events=events)))
'''
    return json.loads(wsl(PY, '-c', code, str(pid), timeout=30).stdout)


def run():
    target = OUT / 'fixture-port'
    assert (target / 'attempt-01-host-memory-stop/preservation-receipt.json').exists()
    assert not (target / 'run-intent.json').exists()
    assert not (target / 'supervisor-intent.json').exists()
    authorization()
    assert ram() >= 12, 'Require headroom for a maximum10GiB task scope and host reserve'
    import msvcrt
    lock = (OUT / 'supervisor.lock').open('a+b')
    lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    unit = 'enveda-c75-memory-bound-' + str(os.getpid())
    command = ['wsl.exe', '-d', 'Ubuntu-24.04', '--', 'systemd-run', '--user',
               '--scope', '--unit=' + unit, *LIMITS, PY,
               LINUX_ROOT + '/scripts/run_c75_local_port.py', 'fixture']
    deadline = datetime.now(timezone.utc) + timedelta(hours=6)
    write(target / 'supervisor-intent.json', dict(started_utc=stamp(), pid=os.getpid(),
        source_sha256=sha(__file__), mode='fixture', unit=unit,
        deadline_utc=deadline.isoformat(), exactly_one_local_inference_launch=True,
        task_memory_high_GiB=8, task_memory_max_GiB=10, task_swap_max_GiB=4,
        host_reserve_GiB=4, new_source_quality_read=False, C13_read=False, scoring_entries=0,
        original_driver_sha256=sha(ROOT / 'scripts/run_c75_local_port.py'),
        notebook_sha256=sha(target / 'notebook.ipynb'),
        resource_probe_sha256=sha(OUT / 'memory-scope-probe.json'),
        preserved_attempt_sha256=sha(target / 'attempt-01-host-memory-stop/preservation-receipt.json')))
    with (target / 'local-pipeline.log').open('w', encoding='utf8') as stdout, \
         (target / 'local-pipeline.err').open('w', encoding='utf8') as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        try:
            while child.poll() is None:
                authorization()
                assert datetime.now(timezone.utc) < deadline
                free = ram()
                if free < 4:
                    advised = trim_own_cache(); time.sleep(5); free = ram()
                    write(target / 'cache-pressure-action.json', dict(checked_utc=stamp(),
                        task_only_file_cache_advice=True, files=advised, host_free_after_GiB=free))
                    assert free >= 4, ('Host RAM reserve below4GiB after owned-cache advice', free)
                progress = read(target / 'run-progress.json') if (target / 'run-progress.json').exists() else None
                intent = read(target / 'run-intent.json') if (target / 'run-intent.json').exists() else None
                scope = monitor_scope(intent['pid']) if intent else None
                write(target / 'supervisor-status.json', dict(checked_utc=stamp(), pid=os.getpid(),
                    child_windows_pid=child.pid, mode='fixture', phase='memory_bounded_local_inference_running',
                    host_free_GiB=free, inference_launches=1, progress=progress, scope=scope,
                    new_source_quality_read=False, C13_read=False, scoring_entries=0))
                time.sleep(10)
            assert child.returncode == 0, ('Local pipeline failed; no automatic retry', child.returncode)
            assert (target / 'run-complete.json').exists()
        except BaseException as exc:
            if child.poll() is None and (target / 'run-intent.json').exists():
                stop_owned_group(target)
                child.wait(timeout=30)
            write(target / 'supervisor-failed.json', dict(checked_utc=stamp(),
                source_sha256=sha(__file__), mode='fixture', error=type(exc).__name__+': '+str(exc)[:600],
                automatic_retry=False, new_source_quality_read=False, C13_read=False, scoring_entries=0))
            raise
    write(target / 'supervisor-complete.json', dict(completed_utc=stamp(),
        source_sha256=sha(__file__), mode='fixture', inference_launches=1,
        run_complete_sha256=sha(target / 'run-complete.json'),
        independent_parity_or_collection_audit_required=True,
        new_source_quality_read=False, C13_read=False, scoring_entries=0))
    print('C75 MEMORY-BOUNDED ORIGINAL PIPELINE FINISHED; INDEPENDENT AUDIT REQUIRED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['probe', 'archive', 'run'])
    action = parser.parse_args().action
    {'probe': probe, 'archive': archive_failed, 'run': run}[action]()
