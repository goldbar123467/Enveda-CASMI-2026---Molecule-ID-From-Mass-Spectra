"""Monitor the live Kaggle submissions until a completed public score beats a fixed target.

Read-only; never submits or changes selected submissions. Polls at most every 180 s.
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'
ROOT = Path(__file__).resolve().parents[1]


def record(item):
    get = item.get if isinstance(item, dict) else lambda k, default=None: getattr(item, k, default)
    raw_score = get('public_score', get('publicScore'))
    try:
        score = float(raw_score)
        if not math.isfinite(score):
            score = None
    except (ValueError, TypeError):
        score = None
    status = str(get('status', '')).split('.')[-1].upper()
    return dict(ref=str(get('ref', '')), status=status, public_score=score,
                description=str(get('description', '')), date=str(get('date', '')))


def evaluate(items, threshold):
    rows = [record(i) for i in items]
    completed = [r for r in rows if r['status'] == 'COMPLETE' and r['public_score'] is not None]
    best = max(completed, key=lambda r: r['public_score']) if completed else None
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(), target=threshold,
                beaten=bool(best is not None and best['public_score'] > threshold), best=best,
                pending=[r for r in rows if r['status'] not in ['COMPLETE', 'ERROR', 'CANCELLED']],
                recent=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=.477)
    ap.add_argument('--interval', type=int, default=180)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    assert args.interval >= 180
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    out = ROOT / 'outputs/local/professional'
    out.mkdir(parents=True, exist_ok=True)
    previous = None
    while True:
        try:
            state = evaluate(api.competition_submissions(COMPETITION), args.threshold)
            temp = out / 'submission_watch.tmp'
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(out / 'submission_watch.json')
            change = json.dumps({k: state[k] for k in ['beaten', 'best', 'pending']}, sort_keys=True)
            if change != previous:
                print(state['checked_utc'], change, flush=True)
                previous = change
            if state['beaten'] or args.once:
                return
        except Exception as exc:
            print(datetime.now(timezone.utc).isoformat(), 'POLL_ERROR', type(exc).__name__, flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
