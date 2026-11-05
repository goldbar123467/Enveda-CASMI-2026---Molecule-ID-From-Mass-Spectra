"""Queue Kaggle fork runs and submit completed versions. Local orchestration only: it never runs a pipeline.

Usage: python scripts/autopilot.py kaggle/forks/queue.json

queue.json: {"date": "YYYY-MM-DD", "daily_cap": 5, "slots": [
    {"slot": "s1-p477", "message": "...", "pushed_version": 1}, {"slot": "s3-relaxmerge", "message": "..."}]}
A slot without `pushed_version` is pushed when fewer than 2 GPU sessions run. Completed versions are submitted in
queue order while today's (UTC) submission count is below the cap; `"not_before": "<UTC ISO time>"` delays a slot's
submission and `"hold": true` keeps it from being pushed. State is written back to queue.json.
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
OWNER = "clarkkitchen"
MAX_GPU = 2
POLL = 90


def cli(*args):
    run = subprocess.run([PY, "-m", "kaggle", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    out = "\n".join(l for l in (run.stdout + run.stderr).splitlines() if "outdated `kaggle` version" not in l)
    return run.returncode, out


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}"
    print(line, flush=True)
    with open(ROOT / "outputs/autopilot.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(slot):
    _, out = cli("kernels", "status", f"{OWNER}/casmi26-{slot}")
    m = re.search(r'KernelWorkerStatus\.(\w+)', out)
    return m.group(1) if m else "UNKNOWN"


def submitted_today():
    _, out = cli("competitions", "submissions", COMP)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for l in out.splitlines() if re.match(r"\s*\d+\s+\S+\s+" + today, l))


def main(queue_path):
    queue_path = Path(queue_path)
    (ROOT / "outputs").mkdir(exist_ok=True)
    while True:
        q = json.loads(queue_path.read_text(encoding="utf-8"))
        slots = q["slots"]
        running = 0
        for s in slots:
            if s.get("pushed_version") and not s.get("final_status"):
                st = status(s["slot"])
                s["last_status"] = st
                if st in ("RUNNING", "QUEUED"):
                    running += 1
                elif st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
                    s["final_status"] = st
                    log(f"{s['slot']} v{s['pushed_version']} finished: {st}")
        for s in slots:
            if running >= MAX_GPU:
                break
            if not s.get("pushed_version") and not s.get("hold"):
                code, out = cli("kernels", "push", "-p", f"kaggle/forks/{s['slot']}",
                                "--accelerator", "NvidiaTeslaT4", "--timeout", "32400")
                m = re.search(r"Kernel version (\d+) successfully pushed", out)
                if m:
                    s["pushed_version"] = int(m.group(1))
                    s.pop("final_status", None)
                    running += 1
                    log(f"pushed {s['slot']} v{m.group(1)}")
                else:
                    log(f"push {s['slot']} failed: {out.strip()[:200]}")
                    break
        for s in slots:
            if s.get("not_before") and datetime.now(timezone.utc).isoformat() < s["not_before"]:
                continue
            if s.get("final_status") == "COMPLETE" and not s.get("submitted") and not s.get("no_submit"):
                used = submitted_today()
                if used >= q.get("daily_cap", 5):
                    if not s.get("cap_logged"):
                        log(f"daily cap reached ({used}); {s['slot']} waits")
                        s["cap_logged"] = True
                    break
                msg = s.get("message", s["slot"])[:240]
                code, out = cli("competitions", "submit", COMP, "-k", f"{OWNER}/casmi26-{s['slot']}",
                                "-v", str(s["pushed_version"]), "-f", "submission.csv", "-m", msg)
                log(f"submit {s['slot']} v{s['pushed_version']}: rc={code} {out.strip()[:200]}")
                if code == 0 and "error" not in out.lower():
                    s["submitted"] = datetime.now(timezone.utc).isoformat()
                else:
                    s["submit_error"] = out.strip()[:300]
                    s["no_submit"] = True
        queue_path.write_text(json.dumps(q, indent=2) + "\n", encoding="utf-8")
        pending = [s for s in slots if not s.get("hold") and not s.get("no_submit")
                   and not (s.get("submitted") or s.get("final_status") in ("ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"))]
        if not pending:
            log("queue drained")
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
