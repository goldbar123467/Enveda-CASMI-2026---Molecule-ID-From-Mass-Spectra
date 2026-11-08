# Local training monitor

Run these commands from the repository root in PowerShell or Command Prompt.
The monitor uses only Python's standard library (Python 3.10+). It prefers the
existing `.venv`, then `.venv-gpu`, then `python` on PATH. No installs are needed.

```powershell
# Browser dashboard (also works by double-clicking this file)
.\training-monitor.cmd

# Live terminal dashboard
.\training-monitor.cmd --terminal

# One terminal snapshot, or machine-readable JSON
.\training-monitor.cmd --once
.\training-monitor.cmd --json
```

The browser dashboard opens at **http://127.0.0.1:8765** and refreshes every five
seconds. Leave the server terminal open. Ctrl+C stops the monitor; training and
the independently running watchdog continue. If the dashboard is already running,
open that URL directly. Use `--web --open --port 8766` to choose a different port.

On macOS/Linux, or without the Windows launcher:

```bash
python scripts/training_monitor.py --web --open
python scripts/training_monitor.py --terminal
```

## What it shows

- Total progress across both model families and all folds, plus a training ETA
  derived from recent epoch timings. ETA excludes production qualification,
  notebook packaging, Kaggle runtime and scoring.
- A selectable loss curve for each model, completed model receipts, and local
  retrieval MRR. These diagnostics are not Kaggle scores.
- GPU utilization, whole-device VRAM, temperature, free system RAM, and the latest
  epoch's Torch peak reserved memory. GPU telemetry refreshes every 15 seconds;
  `nvidia-smi` is optional, and unavailable readings show a dash.
- Training heartbeat, watchdog process/check status, restart count and logs.
  Stale/dead training gets no moving ETA; an overdue/offline watchdog is visible.

## Other runs and future checkouts

C08 compatible-scorer runs are supported. The dashboard shows four recipes × two
folds, successful optimizer steps, data preparation, and `prepared` versus
`validation_pending` states. A prepared run has no moving ETA and has not
started full training. It reads the same small JSON receipts, never model/cache
files. Launch its independent recovery supervisor with
`.\start-compatible-training.cmd`; `--check` validates readiness without launch.
Use `--run outputs/local/long-compatible-20261108` to select it explicitly.
Its terminal/browser views also report the hourly tracker and completion runner.
Queued C08 recipes use their measured pilot pace for ETA; active/completed fits
use recent real training timings, so slower hard-negative recipes are included.

F07 frozen-feature runs are also supported. The monitor shows encoder stage
counts separately from the two seeds' optimizer steps. It reads
`models/seed-N/{history,complete}.json`, the frozen trainer status and latest
`train-N.log/.err`. Its ETA appears only after all spectrum features are cached
and full-training timing exists. The fixed24-hour deadline is a budget cap.

The monitor selects the newest non-smoke `outputs/local/long-*/manifest.json` at
startup. Select a particular run explicitly:

```powershell
.\training-monitor.cmd --web --open --run outputs/local/long-20261106
```

It reads `manifest.json` (`spec.training.epochs`, optional `spec.folds`, default
five folds), `status.json`, per-family `models/{control,external}/fold-N/`
`history.json` and `complete.json`, `watchdog/status.json`, and bounded log tails.
Training files are already written atomically by the shared-fold trainer.

Run outputs, datasets, weights, logs and virtual environments remain git-ignored.
A fresh checkout shows a waiting state until local run outputs exist. The
dashboard does not copy training results from GitHub or start a new training run.

## Resource use and recovery

The server binds only to loopback and serves a fixed set of dashboard assets and
one JSON endpoint (`/api/status`). It never loads Torch, checkpoints or datasets,
never writes training state, and never starts/stops training or a watchdog.
Snapshot work is shared between browser clients for three seconds. There is no
remote service, account, frontend build step, or JavaScript package dependency.

The separate recovery supervisor must already be running for automatic restart
to work. Opening this dashboard does not enable recovery or submission. Its
status panel reports the supervisor's actual process and last/next checks.

Validation:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_training_monitor.py -v
```
