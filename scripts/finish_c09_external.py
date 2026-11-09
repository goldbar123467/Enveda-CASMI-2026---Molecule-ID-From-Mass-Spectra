"""Bounded smoke/confirmation collection for the existing P16 run; never submits."""
import argparse
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


def local_model_evaluation_active():
    result = subprocess.run(['powershell', '-NoProfile', '-Command',
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match 'finish_c09_confirmation\\.py.*--evaluate' } | Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def main():
    global SMOKE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, default=1)
    args = parser.parse_args()
    SMOKE = ROOT / f'outputs/forks/p16-external-reference-c09-v{args.version}'
    import msvcrt
    from kaggle.api.kaggle_api_extended import KaggleApi
    from finish_compatible_campaign import status_name
    lock = (OUT / 'completion-worker.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    assert (OUT / 'implementation-freeze.json').exists()
    api = KaggleApi(); api.authenticate()
    previous_status = OUT / 'completion-status.json'
    previous = json.loads(previous_status.read_text()) if previous_status.exists() else {}
    deadline = datetime.fromisoformat(previous['deadline_utc']) if previous.get('deadline_utc') else datetime.now(timezone.utc) + timedelta(hours=9)
    pin = json.loads((ROOT / 'kaggle/forks/p16-external-reference-c09/external-pin.json').read_text())
    required = ['p16_execution_receipt.json', 'c09e_external_receipt.json', 'c09e_p01_submission.csv',
        'submission.csv', 'c09e_matches.parquet', 'c09e_actions.csv', 'effective_query_spectra.parquet',
        'frozen_candidate_evidence.parquet', 'neural_public.parquet', 'neural_ours.parquet',
        'neural_candidate_features.parquet', 'outsider_neural_features.parquet', 'c09e_code/c09e_policy.json']
    required += ['c09e_code/' + name for name in pin['source_files']]
    if pin.get('fixture_contract_sha256'):
        required += ['p16_fixture_receipt.json', 'c09e_fixture_queries.parquet', 'c09e_fixture_baseline.csv',
            'c09e_fixture_library.csv', 'c09e_fixture_identities.parquet', 'c09e_fixture_matches.parquet', 'c09e_fixture_final.csv']
    downloaded = all((SMOKE / name).exists() for name in required)
    while datetime.now(timezone.utc) < deadline:
        assert not (ROOT / '.local/compute-paused.json').exists(), 'Local compute pause requested'
        state = status_name(api.kernels_status('clarkkitchen/casmi26-p16-external-reference-c09'))
        confirmation = ROOT / 'outputs/local/c09-followup-evaluation-20261109/confirmation.json'
        attention = confirmation.parent / 'confirmation-attention.json'
        failure = json.loads(attention.read_text()) if attention.exists() else {}
        unresolved = bool(failure and not failure.get('resolved_utc'))
        write('completion-status.json', dict(checked_utc=stamp(), pid=os.getpid(), smoke=state,
            original_confirmation_ready=confirmation.exists(), original_confirmation_attention=unresolved,
            independent_smoke_audit_ready=(SMOKE / 'independent_external_smoke_audit.json').exists(),
            deadline_utc=deadline.isoformat(), automatic_submission=False))
        print('C09E WAIT', stamp(), state, 'baseline_ready', confirmation.exists(), flush=True)
        if state in ['ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED']:
            command('failed-smoke-output', [str(CLI), 'scripts/fetch_output.py', 'p16-external-reference-c09', str(SMOKE), '--log-only'], timeout=1800)
            raise RuntimeError('P16 smoke ended ' + state)
        if state == 'COMPLETE' and not downloaded:
            pattern = r'(p16_(execution|fixture)_receipt\.json|c09e_.*\.(json|csv|parquet|log)|submission\.csv|effective_query_spectra\.parquet|frozen_candidate_evidence\.parquet|neural_(public|ours|candidate_features)\.parquet|outsider_neural_features\.parquet|execution_receipt\.json|c09e_code/.*)$'
            command('download-smoke', [str(CLI), 'scripts/fetch_output.py', 'p16-external-reference-c09', str(SMOKE), pattern], timeout=1800)
            missing = [name for name in required if not (SMOKE / name).exists()]
            assert not missing, 'Completed smoke is missing required artifacts: ' + repr(missing)
            downloaded = True
        # Audit as soon as the smoke is complete while the original pipeline is
        # still in the cloud; only defer if its local model evaluator is active.
        if downloaded and not (SMOKE / 'independent_external_smoke_audit.json').exists() and not local_model_evaluation_active():
            command('independent-smoke', [str(PYTHON), '-u', 'scripts/audit_external_reference_smoke.py', str(SMOKE)])
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
