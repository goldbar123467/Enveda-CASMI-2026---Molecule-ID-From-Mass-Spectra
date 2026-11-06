"""Read-only terminal/browser monitor for shared-fold training. Python stdlib only.

Run from any directory: python scripts/training_monitor.py --web --open
No Torch imports, model loads, training changes, or watchdog launches.
"""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name('training_monitor_assets')


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return fallback


def parse_time(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def age_seconds(value, now):
    stamp = parse_time(value)
    return max(0, (now - stamp).total_seconds()) if stamp else None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def process_alive(pid):
    """Probe without sending signals, including on Windows."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        try:
            code = wintypes.DWORD()
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return None


def system_memory():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    'totalPhysical', 'availablePhysical', 'totalPage', 'availablePage',
                    'totalVirtual', 'availableVirtual', 'availableExtended')]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return dict(total_gib=memory.totalPhysical / 2**30,
                        free_gib=memory.availablePhysical / 2**30)
    else:
        try:
            values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            return dict(total_gib=int(values['MemTotal'].split()[0]) / 2**20,
                        free_gib=int(values['MemAvailable'].split()[0]) / 2**20)
        except (OSError, KeyError, ValueError):
            pass
    return {}


def gpu_sample():
    executable = shutil.which('nvidia-smi')
    if not executable:
        return dict(available=False, reason='nvidia-smi unavailable')
    fields = 'name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'
    try:
        result = subprocess.run(
            [executable, '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            return dict(available=False, reason='GPU telemetry unavailable')
        values = [value.strip() for value in result.stdout.splitlines()[0].split(',')]
        sample = dict(available=True, name=values[0])
        for name, value in zip(('utilization', 'used_mib', 'total_mib', 'temperature', 'power_w'), values[1:]):
            try:
                number = float(value)
                sample[name] = number if math.isfinite(number) else None
            except ValueError:
                sample[name] = None
        return sample
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return dict(available=False, reason='GPU telemetry unavailable')


def tail(path, count=12):
    """Bound I/O even if a future training log grows very large."""
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 24000)
            stream.seek(start)
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
            return lines[int(start > 0):][-count:]
    except OSError:
        return []


def discover_run(root=ROOT):
    candidates = []
    for path in (root / 'outputs/local').glob('long-*'):
        manifest = read_json(path / 'manifest.json', {})
        if manifest and not manifest.get('smoke'):
            candidates.append(path)
    if candidates:
        return max(candidates, key=lambda path: (path / 'manifest.json').stat().st_mtime)
    return root / 'outputs/local/long-20261106'


def build_snapshot(run, *, now=None, alive=process_alive, root=ROOT):
    """Small JSON files only; never open a checkpoint, dataset or NumPy array."""
    now = now or datetime.now(timezone.utc)
    manifest = read_json(run / 'manifest.json', {})
    spec = manifest.get('spec', {})
    cfg = spec.get('training', manifest.get('cfg', {}))
    target = int(cfg.get('epochs', 128))
    fold_count = int(spec.get('folds', 5))
    status = read_json(run / 'status.json', {})
    active = (status.get('family'), status.get('fold'))
    models, times = [], []
    for fold in range(fold_count):
        for family in ('control', 'external'):
            folder = run / 'models' / family / f'fold-{fold}'
            history = read_json(folder / 'history.json', [])
            if not isinstance(history, list):
                history = []
            history = [row for row in history if isinstance(row, dict)]
            receipt = read_json(folder / 'complete.json', {})
            epochs = min(target, len(history))
            current = active == (family, fold) and status.get('state') != 'complete'
            # Status and history are separate atomic writes; take the newer epoch.
            if current:
                epochs = min(target, max(epochs, int(status.get('epoch', 0))))
            if receipt:
                epochs = min(target, max(epochs, int(receipt.get('epochs', 0))))
            times.extend(row['seconds'] for row in history[-20:]
                         if finite_number(row.get('seconds')) and row['seconds'] > 0)
            series = [dict(epoch=row.get('epoch', index + 1), loss=row['loss'])
                      for index, row in enumerate(history) if finite_number(row.get('loss'))]
            models.append(dict(id=f'{family}-{fold}', family=family, fold=fold,
                               epoch=epochs, epochs=target, complete=bool(receipt), active=current,
                               loss=series[-1]['loss'] if series else None,
                               metrics=receipt.get('metrics'), series=series))
    total_epochs = target * len(models)
    completed_epochs = sum(model['epoch'] for model in models)
    completed_models = sum(model['complete'] for model in models)
    pace = statistics.median(times) if times else None
    heartbeat_age = age_seconds(status.get('updated_utc'), now)
    trainer_alive = alive(status.get('pid'))
    state = status.get('state', 'waiting')
    if state != 'complete':
        if (root / '.local/compute-paused.json').exists():
            state = 'paused'
        elif trainer_alive is False:
            state = 'stopped'
        elif heartbeat_age is not None and heartbeat_age > 900:
            state = 'stale'
        elif completed_models == len(models):
            state = 'finalizing'
    failure = read_json(run / 'failure.json', {})
    failure_stamp = parse_time(failure.get('utc'))
    heartbeat_stamp = parse_time(status.get('updated_utc'))
    if state not in ('complete', 'paused') and failure_stamp is not None:
        if heartbeat_stamp is None or failure_stamp > heartbeat_stamp:
            state = 'error'
    remaining = max(0, total_epochs - completed_epochs)
    eta = None
    if pace and state in ('training', 'finalizing'):
        eta = (now + timedelta(seconds=remaining * pace + (len(models) - completed_models) * 45)).isoformat()
    watchdog = read_json(run / 'watchdog/status.json', {})
    watchdog['alive'] = alive(watchdog.get('pid'))
    watchdog['check_age_seconds'] = age_seconds(watchdog.get('checked_utc'), now)
    next_check = parse_time(watchdog.get('next_check_utc'))
    watchdog['overdue'] = bool(next_check and now > next_check + timedelta(minutes=2)
                               and watchdog.get('action') not in ('complete', 'paused', 'budget_reached', 'needs_attention'))
    try:
        started = datetime.fromtimestamp((run / 'manifest.json').stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        started = None
    # Do not send the watchdog's duplicated embedded training snapshot to the client.
    watchdog.pop('progress', None)
    watchdog.pop('eta', None)
    return dict(run_id=spec.get('run_id', run.name), run_path=str(run), state=state,
                sampled_utc=now.isoformat(), started_utc=started, updated_utc=status.get('updated_utc'),
                heartbeat_age_seconds=heartbeat_age, trainer_pid=status.get('pid'), trainer_alive=trainer_alive,
                progress=dict(completed_epochs=completed_epochs, total_epochs=total_epochs,
                              completed_models=completed_models, total_models=len(models),
                              percent=100 * completed_epochs / total_epochs if total_epochs else 0,
                              seconds_per_epoch=pace, eta_utc=eta),
                active=dict(family=status.get('family'), fold=status.get('fold'),
                            epoch=status.get('epoch'), epochs=target),
                resources=status.get('resources', {}), models=models, watchdog=watchdog,
                failure=failure if state == 'error' else None,
                log=tail(run / 'train.log'), errors=tail(run / 'train.err', 6),
                confirmation=read_json(run / 'stack/report.json', {}).get('cohorts', {}).get('confirmation'),
                limitation='Fold and OOF scores are retrieval diagnostics; they are not Kaggle scores.',
                monitor_mode='Read only. The existing watchdog handles recovery separately.')


class Monitor:
    def __init__(self, run):
        self.run = run
        self.lock = threading.Lock()
        self.cached, self.gpu = None, {}
        self.updated, self.gpu_updated = -math.inf, -math.inf

    def snapshot(self):
        with self.lock:
            clock = time.monotonic()
            if self.cached is None or clock - self.updated >= 3:
                snapshot = build_snapshot(self.run)
                if clock - self.gpu_updated >= 15:
                    self.gpu = gpu_sample()
                    self.gpu_updated = clock
                snapshot.update(gpu=self.gpu, memory=system_memory())
                self.cached, self.updated = snapshot, clock
            return self.cached


def handler_for(monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path == '/api/status':
                body = json.dumps(monitor.snapshot(), allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            elif path in assets:
                name, content_type = assets[path]
                body = (ASSETS / name).read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass
    return Handler


def local_time(value):
    stamp = parse_time(value)
    return stamp.astimezone().strftime('%I:%M %p %Z').lstrip('0') if stamp else '--'


def display_number(value, digits=1):
    return f'{value:.{digits}f}' if finite_number(value) else '--'


def terminal_view(data):
    progress, active, gpu = data['progress'], data['active'], data.get('gpu', {})
    filled = int(progress['percent'] / 100 * 36)
    bar = '#' * filled + '.' * (36 - filled)
    age = data['heartbeat_age_seconds']
    lines = [f"ENVEDA  /  TRAINING MONITOR                             {data['run_id']}",
             f"{data['state'].upper()} | heartbeat {int(age) if age is not None else '--'}s ago | PID {data['trainer_pid'] or '--'}",
             '', f"[{bar}] {progress['percent']:.1f}%",
             f"Epochs {progress['completed_epochs']:,}/{progress['total_epochs']:,}   Models {progress['completed_models']}/{progress['total_models']}",
             f"Training ETA {local_time(progress['eta_utc'])}   Pace {display_number(progress['seconds_per_epoch'])} s/epoch",
             f"Current {active['family'] or '--'} / fold {active['fold'] if active['fold'] is not None else '--'} / epoch {active['epoch'] or 0}/{active['epochs']}", '',
             f"GPU {gpu.get('name', 'unavailable')} | {display_number(gpu.get('utilization'), 0)}% | {display_number(gpu.get('temperature'), 0)} C",
             f"VRAM {display_number((gpu.get('used_mib') or 0) / 1024) if gpu.get('available') else '--'} / {display_number((gpu.get('total_mib') or 0) / 1024) if gpu.get('available') else '--'} GiB | Free RAM {display_number(data.get('memory', {}).get('free_gib'))} GiB",
             '', 'FOLD   CONTROL (epoch / state / loss)       EXTERNAL (epoch / state / loss)']
    def model_cell(model):
        state = 'done' if model['complete'] else ('active' if model['active'] and data['state'] == 'training' else 'pending' if model['epoch'] else 'queued')
        return f"{model['epoch']:>3}/{model['epochs']:<3} {state:<7} {display_number(model['loss'], 4):>6}"
    for index in range(0, len(data['models']), 2):
        control, external = data['models'][index:index + 2]
        lines.append(f"{control['fold']:<7}{model_cell(control):<36}{model_cell(external)}")
    watchdog = data['watchdog']
    action = watchdog.get('action', 'not configured')
    if watchdog.get('alive') is False and action not in ('complete', 'paused', 'budget_reached', 'needs_attention'):
        action += ' (process offline)'
    if watchdog.get('overdue'):
        action += ' (check overdue)'
    lines += ['', f"Watchdog: {action} | restarts {watchdog.get('restarts', '--')} | next {local_time(watchdog.get('next_check_utc'))}",
              data['limitation']]
    if data.get('failure'):
        lines += ['ERROR: ' + str(data['failure'].get('error', data['failure']))]
    if data['errors']:
        lines += ['STDERR'] + data['errors'][-2:]
    if shutil.get_terminal_size((110, 24)).lines >= 32:
        lines += ['', 'RECENT LOG'] + data['log'][-3:]
    lines += ['Read only | Browser: training-monitor.cmd --web --open | Ctrl+C exits monitor']
    width = shutil.get_terminal_size((110, 40)).columns
    return '\n'.join(line[:max(40, width - 1)] for line in lines)


def enable_ansi():
    if os.name == 'nt' and sys.stdout.isatty():
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32')
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        handle = kernel.GetStdHandle(-11 & 0xffffffff)
        mode = wintypes.DWORD()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel.SetConsoleMode(handle, mode.value | 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Run folder; defaults to newest non-smoke long run')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--web', action='store_true', help='Serve browser dashboard on localhost')
    modes.add_argument('--terminal', action='store_true', help='Live terminal view (Python default)')
    modes.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    modes.add_argument('--json', action='store_true', help='Print one JSON snapshot and exit')
    parser.add_argument('--open', action='store_true', help='Open browser when using --web')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--interval', type=float, default=5, help='Terminal refresh seconds (minimum 2)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if not math.isfinite(args.interval) or args.interval < 2:
        parser.error('--interval must be at least 2 seconds')
    run = (ROOT / args.run).resolve() if args.run else discover_run()
    monitor = Monitor(run)
    try:
        if args.web:
            url = f'http://127.0.0.1:{args.port}'
            try:
                server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(monitor))
            except OSError as error:
                parser.exit(1, f'Cannot open {url}: {error}\nChoose another --port or use the already-running dashboard.\n')
            print(f'Training dashboard: {url}\nRun: {run}\nRead only. Ctrl+C stops the dashboard, training continues.', flush=True)
            if args.open:
                webbrowser.open(url)
            try:
                server.serve_forever(poll_interval=0.5)
            finally:
                server.server_close()
        elif args.json:
            print(json.dumps(monitor.snapshot(), indent=2, allow_nan=False))
        else:
            enable_ansi()
            while True:
                if sys.stdout.isatty() and not args.once:
                    print('\033[2J\033[H', end='')
                print(terminal_view(monitor.snapshot()), flush=True)
                if args.once or not sys.stdout.isatty():
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nMonitor stopped. Training and watchdog continue.')


if __name__ == '__main__':
    main()
