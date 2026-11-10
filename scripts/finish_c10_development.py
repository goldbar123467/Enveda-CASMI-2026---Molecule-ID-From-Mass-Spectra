"""Bounded local continuation after the existing C10 collector; never submits."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local/c10-candidates-20261110'
LOCAL = ROOT / 'outputs/local/c10-casmi2022-20261110'


def main():
    import msvcrt
    lock = (OUT / 'development-worker.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    deadline = datetime.fromisoformat(json.loads((LOCAL / 'push-receipt.json').read_text())['checked_utc']) + timedelta(hours=10)
    while datetime.now(timezone.utc) < deadline:
        if (ROOT / '.local/compute-paused.json').exists(): raise RuntimeError('Local compute pause requested')
        if (LOCAL / 'collector-attention.json').exists(): raise RuntimeError('Inspect existing C10 collector attention; no duplicate launch')
        state = json.loads((LOCAL / 'collector-status.json').read_text())
        if state.get('state') == 'collected': break
        print('C10 DEVELOPMENT WAIT', datetime.now(timezone.utc).isoformat(), state.get('state'), flush=True)
        time.sleep(60)
    else: raise TimeoutError('Original C10 inference deadline reached; inspect same live run')
    owners = subprocess.run(['powershell', '-NoProfile', '-Command',
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'scripts[/\\\\](c09_(train|control_train|hard_train)|select_c09_followups|develop_c10_candidates)' } | Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True, check=True)
    assert not owners.stdout.strip(), 'Another local GPU workload is active'
    subprocess.run([str(ROOT / '.venv-gpu/Scripts/python.exe'), '-u', 'scripts/develop_c10_candidates.py'],
        cwd=ROOT, check=True, timeout=3600)
    print('[C10DEV] DONE development selection frozen; no reserved metrics/submission', flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        (OUT / 'development-worker-attention.json').write_text(json.dumps(dict(
            checked_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), error=repr(exc),
            automatic_submission=False), indent=2))
        raise
