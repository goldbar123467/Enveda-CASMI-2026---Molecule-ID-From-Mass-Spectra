"""Refresh runs.csv statuses and regenerate RUNS.md (run hourly; safe to run any time).

    .venv/Scripts/python.exe scripts/update_runs.py            # once
    .venv/Scripts/python.exe scripts/update_runs.py --loop     # every hour until killed

runs.csv is the record: one row per compute run, added by hand (or by Claude) when the run is launched, with the
result filled in when it finishes. This script only touches `status` / `ended_utc` of unfinished rows:
- kaggle-* rows: `kaggle kernels status clarkkitchen/casmi26-<slot>` (ref = "<slot> v<N>").
- local-* rows: the log at `ref`; finished when it has `expect` lines "] DONE" / "DIAG written", error on Traceback.
RUNS.md also shows the Kaggle quota and today's submission count.
"""
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "runs.csv"
MD = ROOT / "RUNS.md"
PY = str(ROOT / ".venv/Scripts/python.exe")
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
FINAL = {"complete", "error", "cancelled", "killed"}
GPU_HOURS, TPU_HOURS = 30, 20  # account page; the API's GPU total (6 h) is wrong


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def cli(*args):
    p = subprocess.run([PY, "-m", "kaggle", *args], capture_output=True, text=True, timeout=300)
    return p.stdout + p.stderr


def kaggle_status(ref):
    slot = ref.split()[0]
    m = re.search(r"KernelWorkerStatus\.(\w+)", cli("kernels", "status", f"clarkkitchen/casmi26-{slot}"))
    return m.group(1).lower() if m else None


def local_status(row):
    log = ROOT / row["ref"]
    manifest = log.parent / 'manifest.json'
    if manifest.is_file():
        try:
            kind = json.loads(manifest.read_text()).get('kind')
        except (OSError, ValueError):
            kind = None
        if kind == 'frozen_retrieval' or (kind == 'compatible_scorer' and row.get('kind') == 'train'):
            from training_monitor import build_snapshot
            snapshot = build_snapshot(log.parent)
            stage = snapshot['stage']
            progress = snapshot['progress']
            note = stage['label']
            if stage.get('total'):
                note += f" {stage['completed']:,}/{stage['total']:,}"
            note += f"; head steps {progress['completed_epochs']:,}/{progress['total_epochs']:,}"
            if snapshot['watchdog'].get('action') == 'needs_attention':
                return 'error', note + '; supervisor needs attention'
            return {'training': 'running', 'validation_pending': 'complete', 'pilot_checkpoint': 'complete', 'complete': 'complete', 'error': 'recovering',
                    'stopped': 'stalled', 'stale': 'stalled'}.get(snapshot['state'], snapshot['state']), note
    if not log.is_file():
        return row["status"], ""
    text = log.read_text(encoding="utf-8", errors="replace")
    done = len(re.findall(r"\] (?:DONE|DIAG written)|^DONE C09 (?:DEVELOPMENT FREEZE|FROZEN CONFIRMATION|EXTERNAL DEVELOPMENT|PILOT|DISJOINT VALIDATION PREPARATION)|^DONE FOLLOWUP DEVELOPMENT FREEZE", text, re.MULTILINE))
    expect = int(row["expect"] or 0)
    note = f"{done}/{expect} done" if expect else f"{done} done"
    if "Traceback" in text:
        return "error", note
    if expect and done >= expect:
        return "complete", note
    idle = time.time() - log.stat().st_mtime
    return ("running" if idle < 1800 else "stalled"), note + (f", log idle {idle / 60:.0f} min" if idle >= 1800 else "")


def quota():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        a = KaggleApi()
        a.authenticate()
        q = a.quota_view()
        if not isinstance(q, dict):
            q = json.loads(q.to_json() if hasattr(q, "to_json") else str(q))
        used =lambda d: float(re.match(r"[\d.]+?(?=\.\d+s$|s$)", str(d.get("timeUsed", "0s"))).group(0) or 0) / 3600
        return (f"GPU {used(q['gpuQuota']):.1f} h used of {GPU_HOURS} h · TPU {used(q['tpuQuota']):.1f} h of {TPU_HOURS} h"
                f" · resets {q['quotaRefreshTime'][:16].replace('T', ' ')} UTC")
    except Exception as e:  # noqa: BLE001
        return f"quota unavailable ({e.__class__.__name__})"


def submissions_today():
    out = cli("competitions", "submissions", COMP, "--csv")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for line in out.splitlines() if f",{today} " in line)


def refresh(rows):
    for r in rows:
        if r["status"] in FINAL:
            continue
        if r["platform"].startswith("kaggle"):
            s = kaggle_status(r["ref"])
            if s:
                r["status"] = {"cancel_acknowledged": "cancelled"}.get(s, s)
        elif r["platform"].startswith("local"):
            r["status"], note = local_status(r)
            if note and r["status"] not in FINAL:
                r["result"] = note
        if r["status"] in FINAL and not r["ended_utc"]:
            r["ended_utc"] = now()


def cell(s):
    return str(s).replace("|", "/").replace("\n", " ")


def render(rows):
    active = [r for r in rows if r["status"] not in FINAL]
    done = sorted((r for r in rows if r["status"] in FINAL), key=lambda r: r["ended_utc"] or r["started_utc"], reverse=True)
    out = [
        "# Runs",
        "",
        f"Updated {now()} UTC by `scripts/update_runs.py` (hourly). Generated from `runs.csv` — edit that file, not this one.",
        "",
        f"- Kaggle quota: {quota()}",
        f"- Submissions today (UTC): {submissions_today()} of 5 — see `SUBMISSIONS.md`",
        "",
        "## Active",
        "",
        "| id | platform | kind | ref | started (UTC) | status | purpose | progress |",
        "|---|---|---|---|---|---|---|---|",
    ]
    out += [f"| {r['id']} | {r['platform']} | {r['kind']} | {cell(r['ref'])} | {r['started_utc']} | **{r['status']}** | "
            f"{cell(r['purpose'])} | {cell(r['result'])} |" for r in active] or ["| — | | | | | | | |"]
    out += ["", "## Finished (newest first)", "",
            "| id | platform | kind | ref | started (UTC) | ended (UTC) | status | purpose | result |",
            "|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {r['id']} | {r['platform']} | {r['kind']} | {cell(r['ref'])} | {r['started_utc']} | {r['ended_utc']} | "
            f"{r['status']} | {cell(r['purpose'])} | {cell(r['result'])} |" for r in done]
    MD.write_text("\n".join(out) + "\n", encoding="utf-8")


def once():
    with CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    refresh(rows)
    with CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    render(rows)
    print(now(), "runs:", len(rows), "active:", sum(r["status"] not in FINAL for r in rows), flush=True)


def main(loop=False):
    while True:
        if loop and (ROOT / '.local/compute-paused.json').is_file():
            print('Project paused: hourly updater disabled until explicit user resumption.', flush=True)
            return
        once()
        if not loop:
            break
        time.sleep(3600)


if __name__ == "__main__":
    main(loop="--loop" in sys.argv)
