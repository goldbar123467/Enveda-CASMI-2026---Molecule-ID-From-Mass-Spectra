"""Bounded collection of the existing C10 run; no label metrics or submission."""
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local/c10-casmi2022-20261110'
DEST = ROOT / 'outputs/forks/c10-casmi2022-validation-v1'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    path = OUT / name; temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8'); temporary.replace(path)


def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest
    from finish_compatible_campaign import status_name
    from submit_c09_external_once import scientific_sha
    lock = (OUT / 'collector.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    receipt = json.loads((OUT / 'push-receipt.json').read_text())
    kernel = receipt['kernel']; slug = kernel.split('/')[1]
    deadline = datetime.fromisoformat(receipt['checked_utc']) + timedelta(hours=10)
    api = KaggleApi(); api.authenticate()
    while datetime.now(timezone.utc) < deadline:
        try:
            state = status_name(api.kernels_status(kernel))
        except Exception as exc:
            write('collector-status.json', dict(checked_utc=stamp(), pid=os.getpid(), poll_error=type(exc).__name__,
                terminal_state_inferred=False, deadline_utc=deadline.isoformat()))
            time.sleep(60); continue
        write('collector-status.json', dict(checked_utc=stamp(), pid=os.getpid(), kernel=kernel, state=state,
            deadline_utc=deadline.isoformat(), label_metrics_computed=False, automatic_submission=False))
        print('C10 INFERENCE WAIT', stamp(), state, flush=True)
        if state == 'COMPLETE': break
        if state in ['ERROR', 'CANCELLED', 'CANCEL_ACKNOWLEDGED']:
            raise RuntimeError('Existing C10 run ended ' + state)
        time.sleep(60)
    else:
        raise TimeoutError('Bounded wait expired; recheck the same live kernel before any recovery')
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest(); request.user_name = 'clarkkitchen'; request.kernel_slug = slug
        response = json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    assert response['metadata']['currentVersionNumber'] == 1
    assert response['metadata']['isPrivate'] and not response['metadata']['enableInternet']
    assert scientific_sha(json.loads(response['blob']['source'])) == receipt['remote_scientific_sha256']
    pattern = r'(c10_.*\.(json|csv|parquet|log)|neural_(public|ours)\.parquet|outsider_neural_features\.parquet|frozen_candidate_evidence\.parquet)$'
    subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), 'scripts/fetch_output.py', slug.removeprefix('casmi26-'), str(DEST), pattern], cwd=ROOT, check=True, timeout=1800)
    required = ['c10_report.json', 'c10_p01.csv', 'c10_p14.csv', 'c10_neural_p01.parquet', 'c10_queries.parquet',
        'c10_gpu1_preflight.json', 'c10_t4_allocation.json', 'neural_public.parquet', 'neural_ours.parquet', 'outsider_neural_features.parquet']
    assert all((DEST / name).is_file() for name in required), 'Incomplete C10 artifact collection'
    report = json.loads((DEST / 'c10_report.json').read_text())
    assert report['queries'] == 160 and report['baseline_replay_exact']
    assert not report['query_labels_mounted'] and not report['metrics_computed'] and not report['recipe_selection']
    assert hashlib.sha256((DEST / 'c10_p01.csv').read_bytes()).hexdigest() == report['baseline_sha256']
    for name in ['c10_p01.csv', 'c10_p14.csv']:
        with (DEST / name).open(newline='', encoding='utf-8') as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == 160 and len({row['molecule_id'] for row in rows}) == 160
        assert all(1 <= len(row['smiles'].split(';')) <= 25 for row in rows)
    write('collector-status.json', dict(completed_utc=stamp(), pid=os.getpid(), kernel=kernel, state='collected',
        artifacts=str(DEST), label_metrics_computed=False, automatic_submission=False,
        next_action='Use only40 development labels for selection; freeze recipe before120 reserved metrics'))
    print('[C10COL] DONE collected unlabelled inference; development-only selection next', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write('collector-attention.json', dict(checked_utc=stamp(), error=repr(exc), automatic_submission=False))
        raise
