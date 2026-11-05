"""Append a run to runs.csv (status 'running', or 'queued' for Kaggle) and regenerate RUNS.md.

    python scripts/add_run.py <id> <platform> <kind> "<ref>" "<purpose>" [expect]
    e.g. python scripts/add_run.py n7a kaggle-gpu train "n7a-e128-refits v1" "Refits 221-222 at 128 epochs"
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rid, platform, kind, ref, purpose = sys.argv[1:6]
    expect = sys.argv[6] if len(sys.argv) > 6 else ""
    path = ROOT / "runs.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    assert rid not in {r["id"] for r in rows}, f"run id {rid} exists"
    rows.append(dict(id=rid, platform=platform, kind=kind, ref=ref,
                     started_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), ended_utc="",
                     status="queued" if platform.startswith("kaggle") else "running", expect=expect,
                     purpose=purpose, result=""))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # A new version is registered immediately before push. Refreshing here can
    # read the preceding version's terminal status and permanently mislabel it.
    # Render the just-recorded state; the next ordinary refresh observes the push.
    from update_runs import render
    render(rows)


if __name__ == "__main__":
    main()
