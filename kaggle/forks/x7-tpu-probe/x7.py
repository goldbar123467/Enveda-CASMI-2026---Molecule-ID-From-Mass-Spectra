# ---- x7: can Saner Turhaner's scorer (kit sanpier/casmi-spectrum-model-v1) train on Kaggle's TPU v3-8 with PyTorch/XLA? ----
# The notebook process never initialises the TPU runtime (one process holds the device); every test runs tpu_train.py
# (kaggle/forks/tpu_train.py, embedded below) as a subprocess: 1) one core, 2 short epochs; 2) xmp.spawn, one seed per core.
import json, os, subprocess, sys, time
from pathlib import Path
W = Path('/kaggle/working'); T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
os.environ.setdefault('PJRT_DEVICE', 'TPU')
rep = {'cpu_count': os.cpu_count(), 'env': {k: v for k, v in os.environ.items() if any(s in k for s in ('TPU', 'XLA', 'PJRT'))}}
try: rep['mem_gb'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30, 1)
except Exception as e: rep['mem_error'] = repr(e)
pk = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True).stdout
rep['packages'] = [l for l in pk.splitlines() if any(s in l.lower() for s in ('torch', 'jax', 'tensorflow', 'libtpu', 'numpy', 'pandas', 'pyarrow'))]
q = subprocess.run([sys.executable, '-c', "import torch, torch_xla, torch_xla.runtime as xr; print(torch.__version__, torch_xla.__version__, xr.device_type(), xr.global_runtime_device_count())"],
                   capture_output=True, text=True, timeout=600)
rep['xla_query'] = dict(rc=q.returncode, out=q.stdout[-2000:], err=q.stderr[-3000:])
log(json.dumps(rep, indent=1))
(W / 'tpu_train.py').write_text(SRC)
tests = [('single', ['--mode', 'single', '--epochs', '2', '--steps', '100', '--k0', '301'], 1800),
         ('spawn', ['--mode', 'spawn', '--epochs', '2', '--steps', '100', '--k0', '311'], 2700)]
for name, cmd, timeout in tests:
    t = time.time()
    try:
        p = subprocess.run([sys.executable, str(W / 'tpu_train.py'), *cmd], capture_output=True, text=True, timeout=timeout)
        rep[name] = dict(rc=p.returncode, seconds=round(time.time() - t), stdout=p.stdout[-8000:], stderr=p.stderr[-8000:])
    except subprocess.TimeoutExpired as e:
        rep[name] = dict(timeout=timeout, stdout=str(e.stdout)[-8000:], stderr=str(e.stderr)[-8000:])
    log(name, 'rc', rep[name].get('rc'), 'seconds', rep[name].get('seconds'))
    print(rep[name]['stdout'][-4000:]); print(rep[name]['stderr'][-4000:])
    (W / 'x7_report.json').write_text(json.dumps(rep, indent=1, default=str))
reports = {p.name: json.loads(p.read_text()) for p in sorted((W / 'reports').glob('*.json'))} if (W / 'reports').is_dir() else {}
for n, r in reports.items():
    print('REPORT', n, r.get('device'), r.get('load_seconds'), r.get('train_seconds'), [h.get('seconds') for h in r.get('history', [])], (r.get('error') or '')[-600:])
log('X7_DONE')
