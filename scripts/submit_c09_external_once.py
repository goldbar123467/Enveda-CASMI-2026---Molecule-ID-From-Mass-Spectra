"""Submit a reviewed, hash-bound C09E candidate once; reconcile ambiguous intents."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from c09_submission_guard import check

ROOT = Path(__file__).resolve().parents[1]
COMP = 'enveda-CASMI26-molecule-id-mass-spectra'
KERNEL = 'clarkkitchen/casmi26-p16-external-reference-c09'
MARKER = 'C09-P16-EXTREF-20261109'
OUT = ROOT / 'outputs/local/c09-external-reference-20261109'
SMOKE = ROOT / 'outputs/forks/p16-external-reference-c09-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def scientific_sha(notebook):
    cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    return hashlib.sha256(json.dumps(cells, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def verify_review(qualification, audit, confirmation, pin):
    assert qualification['decision'] == 'qualified_sequential_trial'
    assert qualification['marker'] == MARKER and qualification['maximum_pending'] == 1
    assert qualification['champion_retained'] == .480
    assert audit['execution_qualified'] and audit['changed_path_exercised']
    assert audit['fixture_real_queries'] == 30 and audit['fixture_changed_rows'] == 7 and audit['fixture_inserted_identities'] == 2
    assert audit['fixture_local_cloud_peak_score_max_abs_difference'] < 1e-12 and audit['invalid_structure_fallback_passed']
    assert audit['valid_rows'] == 400 and audit['real_queries'] == 12 and audit['official_identity_duplicates'] == 0
    assert audit['real_queries_supported_by_both_P01_models'] == 12
    assert audit['exact_independent_merge'] and audit['independent_one_to_one_peak_assignment']
    assert audit['local_cloud_peak_score_max_abs_difference'] < 1e-12 and audit['unchanged_P01_primary_error'] < 1e-6
    assert audit['source_files'] == pin['source_files'] and audit['reference_files'] == pin['reference_files']
    assert confirmation['policy_sha256'] == pin['policy_sha256']
    assert confirmation['truth_absent_mrr'] == 0 and not confirmation['confirmation_recipe_selection']
    assert confirmation['cohorts']['full80']['queries'] == 80
    full = confirmation['cohorts']['full80']['comparisons']
    clean = confirmation['cohorts']['no_raw_competition_signature_overlap']
    assert clean['queries'] >= 40
    for summary in [full['all_reference'], clean['comparisons']['all_reference']]:
        assert summary['delta'] > 0 and summary['ci95'][0] > 0, 'Confirmation does not establish a supported gain'
    assert full['input_alias_excluded']['delta'] >= 0 and full['input_alias_excluded']['ci95'][0] >= 0
    interpretation = qualification['negative_control_interpretation']
    assert interpretation['detects_known_P14_regression'] == confirmation['detects_known_P14_regression']
    assert isinstance(interpretation['assessment'], str) and len(interpretation['assessment']) > 30
    assert qualification['limitations'] and qualification['rationale']


def main():
    global SMOKE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    SMOKE = ROOT / f'outputs/forks/p16-external-reference-c09-v{args.version}'
    assert not (ROOT / '.local/compute-paused.json').exists(), 'Project paused'
    import msvcrt
    lock = (ROOT / '.local/c09-submission.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    qpath = OUT / 'submission-qualification.json'
    qualification = json.loads(qpath.read_text())
    for name, expected in qualification['artifact_sha256'].items():
        assert sha(ROOT / name) == expected, name
    required = [f'outputs/forks/p16-external-reference-c09-v{args.version}/independent_external_smoke_audit.json',
                'outputs/local/c09-external-reference-20261109/confirmation.json',
                'outputs/local/c09-external-reference-20261109/development.json',
                'outputs/local/c09-external-reference-20261109/implementation-freeze.json',
                'outputs/local/c09-followup-evaluation-20261109/confirmation.json',
                'kaggle/forks/p16-external-reference-c09/notebook.ipynb',
                'scripts/audit_external_reference_smoke.py', 'scripts/submit_c09_external_once.py']
    assert set(required) <= set(qualification['artifact_sha256']), 'Review omitted required evidence'
    pin = json.loads((ROOT / 'kaggle/forks/p16-external-reference-c09/external-pin.json').read_text())
    audit = json.loads((SMOKE / 'independent_external_smoke_audit.json').read_text())
    confirmation = json.loads((OUT / 'confirmation.json').read_text())
    verify_review(qualification, audit, confirmation, pin)
    assert audit['independent_audit_source_sha256'] == sha(ROOT / 'scripts/audit_external_reference_smoke.py')
    assert sha(SMOKE / 'submission.csv') == audit['csv_sha256']
    notebook_path = ROOT / 'kaggle/forks/p16-external-reference-c09/notebook.ipynb'
    assert sha(notebook_path) == pin['notebook_sha256']
    assert sha(ROOT / 'configs/c09-external-reference-20261109.json') == pin['policy_sha256']
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest
    from finish_compatible_campaign import status_name
    api = KaggleApi(); api.authenticate()
    assert status_name(api.kernels_status(KERNEL)) == 'COMPLETE'
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest(); request.user_name = 'clarkkitchen'; request.kernel_slug = KERNEL.split('/')[1]
        response = json.loads(client.kernels.kernels_api_client.get_kernel(request).to_json())
    assert response['metadata']['currentVersionNumber'] == args.version
    assert response['metadata']['isPrivate'] and not response['metadata']['enableInternet']
    remote = scientific_sha(json.loads(response['blob']['source']))
    assert remote == scientific_sha(json.loads(notebook_path.read_text(encoding='utf-8')))
    ledger_path = ROOT / '.local/c09-submission-ledger.json'; ledger = json.loads(ledger_path.read_text())
    submissions = api.competition_submissions(COMP); action = check(submissions, ledger, MARKER)
    if action['action'] == 'already_recorded':
        reference = int(action['submission']['reference'])
        if args.dry_run:
            print(json.dumps(dict(already_recorded=True, reference=reference, submitted=False)), flush=True)
            return
    else:
        today = datetime.now(timezone.utc).date()
        assert sum(row.date.date() == today for row in submissions) < 5, 'Daily competition allowance exhausted'
        if args.dry_run:
            print(json.dumps(dict(ready=True, marker=MARKER, version=args.version, remote_scientific_sha256=remote,
                                  qualification_sha256=sha(qpath), pending=0, submitted=False)), flush=True)
            return
        intent = dict(marker=MARKER, kernel=KERNEL, version=args.version, utc=datetime.now(timezone.utc).isoformat(),
                      csv_sha256=audit['csv_sha256'], qualification_sha256=sha(qpath), remote_scientific_sha256=remote, resolved=False)
        ledger.setdefault('intents', []).append(intent); write(ledger_path, ledger)
        # Preserve the unresolved intent if this call raises or its acceptance
        # is ambiguous. A retry must reconcile account history before sending.
        response = api.competition_submit_code('submission.csv',
            MARKER + ': Complete pinned P01 plus fixed measured-reference spectral promotions; official identity dedup. '
            'Real12/400 Kaggle smoke, independent peak/merge/fallback/local-cloud parity, frozen development and complete80 confirmation reviewed. '
            'External-source/provenance limitations retained; hidden gain unproven.',
            competition=COMP, kernel=KERNEL, kernel_version=args.version)
        reference = int(response.ref)
    assert reference > 0
    record = dict(marker=MARKER, reference=reference, kernel=KERNEL, version=args.version,
                  submitted_utc=datetime.now(timezone.utc).isoformat(), status='accepted; actual score pending',
                  csv_sha256=audit['csv_sha256'], qualification_sha256=sha(qpath))
    for intent in ledger.get('intents', []):
        if intent['marker'] == MARKER:
            intent.update(resolved=True, reference=reference)
    existing = [s for s in ledger.get('submissions', []) if s['marker'] == MARKER]
    assert not existing or int(existing[0]['reference']) == reference
    if existing:
        record = existing[0]
    if not existing:
        ledger.setdefault('submissions', []).append(record)
    write(ledger_path, ledger); write(ROOT / 'docs/evidence/c09-2026-11-09/p16-submission.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
