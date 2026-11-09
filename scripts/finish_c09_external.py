"""Bounded smoke/confirmation collection for the existing P16 run; never submits."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/local/c09-external-reference-20261109'
SMOKE = ROOT / 'outputs/forks/p16-external-reference-c09-v1'
CLI = ROOT / '.venv/Scripts/python.exe'
PYTHON = ROOT / '.venv-gpu/Scripts/python.exe'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    path = OUT / name; temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def command(label, args, timeout=3600):
    with (OUT / (label + '.log')).open('w') as stdout, (OUT / (label + '.err')).open('w') as stderr:
        subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr, check=True, timeout=timeout)


def main():
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi
    from finish_compatible_campaign import status_name
    lock = (OUT / 'completion-worker.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    assert (OUT / 'implementation-freeze.json').exists()
    api = KaggleApi(); api.authenticate()
    deadline = datetime.now(timezone.utc) + timedelta(hours=9)
    downloaded = (SMOKE / 'p16_execution_receipt.json').exists()
    while datetime.now(timezone.utc) < deadline:
        assert not (ROOT / '.local/compute-paused.json').exists(), 'Local compute pause requested'
        state = status_name(api.kernels_status('clarkkitchen/casmi26-p16-external-reference-c09'))
        confirmation = ROOT / 'outputs/local/c09-followup-evaluation-20261109/confirmation.json'
        attention = confirmation.parent / 'confirmation-attention.json'
        failure = json.loads(attention.read_text()) if attention.exists() else {}
        unresolved = bool(failure and not failure.get('resolved_utc'))
        write('completion-status.json', dict(checked_utc=stamp(), pid=os.getpid(), smoke=state,
            original_confirmation_ready=confirmation.exists(), original_confirmation_attention=unresolved,
            deadline_utc=deadline.isoformat(), automatic_submission=False))
        print('C09E WAIT', stamp(), state, 'baseline_ready', confirmation.exists(), flush=True)
        if state in ['ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED']:
            command('failed-smoke-output', [str(CLI), 'scripts/fetch_output.py', 'p16-external-reference-c09', str(SMOKE), '--log-only'], timeout=1800)
            raise RuntimeError('P16 smoke ended ' + state)
        if state == 'COMPLETE' and not downloaded:
            pattern = r'(p16_execution_receipt\.json|c09e_.*\.(json|csv|parquet|log)|submission\.csv|effective_query_spectra\.parquet|frozen_candidate_evidence\.parquet|neural_(public|ours|candidate_features)\.parquet|outsider_neural_features\.parquet|execution_receipt\.json|c09e_code/.*)$'
            command('download-smoke', [str(CLI), 'scripts/fetch_output.py', 'p16-external-reference-c09', str(SMOKE), pattern], timeout=1800)
            downloaded = True
        # Serialize CPU-heavy work after the existing local model evaluation.
        if downloaded and (confirmation.exists() or unresolved):
            if not (SMOKE / 'independent_external_smoke_audit.json').exists():
                command('independent-smoke', [str(PYTHON), '-u', 'scripts/audit_external_reference_smoke.py', str(SMOKE)])
            if unresolved:
                raise RuntimeError('Smoke audited but original full-pipeline confirmation needs recovery')
            if not (OUT / 'confirmation.json').exists():
                command('confirmation', [str(PYTHON), '-u', 'scripts/evaluate_c09_external_confirmation.py'])
            write('completion-status.json', dict(completed_utc=stamp(), pid=os.getpid(), state='complete',
                smoke='COMPLETE', independent_audit=str(SMOKE / 'independent_external_smoke_audit.json'),
                confirmation=str(OUT / 'confirmation.json'), automatic_submission=False))
            print('[C09EF] DONE smoke audit and frozen confirmation; review qualification before submission', flush=True)
            return
        time.sleep(60)
    raise TimeoutError('Existing P16 smoke/confirmation bounded wait expired')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write('completion-attention.json', dict(checked_utc=stamp(), error=repr(exc), automatic_submission=False))
        raise
