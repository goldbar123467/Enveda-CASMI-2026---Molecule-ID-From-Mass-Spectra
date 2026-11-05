"""Push Kaggle forks in order as GPU sessions free up, registering each in runs.csv.

    python scripts/push_queue.py <run_id>:<slot>:<accelerator>:<kind>[:<needs>] ...

Every 90 s it tries to push the head of the queue; Kaggle refuses while 2 GPU sessions run ("Maximum batch GPU
session count"), and the item is retried. `needs` is an optional comma list of kernel slugs that must be COMPLETE
first (e.g. the refit jobs a variant attaches). Purpose text comes from the fork's title.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")


def cli(*args):
    p = subprocess.run([PY, "-m", "kaggle", *args], capture_output=True, text=True, timeout=600)
    return "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "outdated" not in l)


def status(slot):
    m = re.search(r"KernelWorkerStatus\.(\w+)", cli("kernels", "status", f"clarkkitchen/casmi26-{slot}"))
    return m.group(1) if m else "UNKNOWN"


def main():
    queue = [item.split(":") for item in sys.argv[1:]]
    while queue:
        rid, slot, acc, kind, *rest = queue[0]
        needs = [n for n in (rest[0].split(",") if rest else []) if n]
        waiting = [n for n in needs if status(n) != "COMPLETE"]
        if not waiting:
            out = cli("kernels", "push", "-p", str(ROOT / "kaggle/forks" / slot), "--accelerator", acc, "--timeout", "32400")
            print(time.strftime("%H:%M"), "push", slot, "|", out.strip()[-160:], flush=True)
            m = re.search(r"Kernel version (\d+) successfully pushed", out)
            if m:
                title = json.loads((ROOT / "kaggle/forks" / slot / "kernel-metadata.json").read_text())["title"]
                subprocess.run([PY, str(ROOT / "scripts/add_run.py"), rid, "kaggle-tpu" if "Tpu" in acc else "kaggle-gpu",
                                kind, f"{slot} v{m.group(1)}", title], capture_output=True)
                queue.pop(0)
                continue
        else:
            print(time.strftime("%H:%M"), slot, "waits for", waiting, flush=True)
        time.sleep(90)
    print("queue empty", flush=True)


if __name__ == "__main__":
    main()
