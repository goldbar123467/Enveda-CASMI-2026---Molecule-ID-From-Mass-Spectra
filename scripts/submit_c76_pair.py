"""One-shot current-family submission or score observation; never auto-retry.

Uses the SAME C42 allowance and lock as retired submission tooling. An accepted
error consumes its entry. An ambiguous API result requires exact reconciliation.
"""
import argparse
import csv
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from c75_port_utils import ROOT, read, sha, stamp
from c09_submission_guard import normalize, TERMINAL
from c76_account_history import complete_history
from submit_c09_external_once import scientific_sha
from c76_submission_contract import (ACTIVE, BASE, COMP, FAMILY, LEDGER, OUT,
    QUALIFICATION, TRIALS, authorization, marker, new_release_allowed, qualified,
    rooted, validate_ledger)


def durable_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def shared_lock():
    import msvcrt
    path = ROOT / '.local/c42-submission.lock'
    with path.open('a+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try: yield
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def empty_ledger():
    return dict(created_utc=stamp(), family=FAMILY, maximum_new_accepted_submissions=2,
        maximum_pending=1, submissions=[], intents=[])


def register(record):
    path = ROOT / 'runs.csv'
    with path.open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    run_id = 'C76SCPU' if record['trial'] == 'cpu' else 'C76SGPU'
    matches = [r for r in rows if r['id'] == run_id]
    assert len(matches) <= 1
    if matches:
        row = matches[0]; assert row['ref'] == str(record['reference'])
    else:
        row = {name: '' for name in fields}; rows.append(row)
        row.update(id=run_id, platform='kaggle-submission', kind='eval', ref=str(record['reference']),
            started_utc=record.get('submitted_utc', record.get('reconciled_utc', stamp())),
            purpose='Approved new C76 R1 ' + record['trial'] + '-fitted direct isomer ranker over the complete formate-compatible parent')
    row['status'] = record['status'].lower()
    if record['status'] in TERMINAL:
        row['ended_utc'] = record.get('actual_checked_utc', stamp())
        row['result'] = 'Actual ' + record['status'] + '; public score=' + str(record.get('public_score')) + '; error=' + str(record.get('error')) + '. Preserved champion0.480; no extra retry.'
    else:
        row['result'] = 'Accepted authorized C76 R1 trial; actual score pending. Shared allowance at most2, max1pending.'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def persist(ledger):
    validate_ledger(ledger, sha(QUALIFICATION)); durable_write(LEDGER, ledger)
    active = authorization(); active['submissions_used'] = len(ledger['submissions'])
    active['phase'] = 'C76_R1_qualified_actual_scoring'
    active['updated_utc'] = stamp(); durable_write(ACTIVE, active)
    for record in ledger['submissions']: register(record)


def client():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate(); return api


def remote_verified(api, entry):
    from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
    assert json.loads(str(api.kernels_status(entry['kernel'])))['status'] == 'COMPLETE'
    with api.build_kaggle_client() as kaggle_client:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = entry['kernel'].split('/')
        remote = json.loads(kaggle_client.kernels.kernels_api_client.get_kernel(request).to_json())
    metadata = remote['metadata']; expected = read(rooted(entry['metadata_path']))
    assert metadata['currentVersionNumber'] == entry['version'] == 1
    assert metadata['isPrivate'] and metadata['enableGpu'] and not metadata['enableInternet']
    assert metadata['dockerImage'] == expected['docker_image']
    actual_sha = scientific_sha(json.loads(remote['blob']['source']))
    assert actual_sha == scientific_sha(read(rooted(entry['notebook_path'])))
    assert sha(rooted(entry['smoke_output'])) == entry['csv_sha256']
    return actual_sha


def reconcile(trial, rows, ledger, entry, remote_sha, dry):
    matches = [r for r in rows if marker(trial) in r['description']]
    assert len(matches) <= 1, 'Duplicate account marker; inspect without another submission'
    records = [r for r in ledger['submissions'] if r['trial'] == trial]
    assert len(records) <= 1
    if not matches:
        assert not records, 'Accepted reference missing from account history; never resubmit'
        return False
    intents = [i for i in ledger['intents'] if i['trial'] == trial]
    assert len(intents) == 1, 'Account marker without its durable exact intent'
    intent = intents[0]
    for key, expected in [('kernel', entry['kernel']), ('version', entry['version']),
        ('csv_sha256', entry['csv_sha256']), ('remote_scientific_sha256', remote_sha)]:
        assert intent[key] == expected, key
    observed = matches[0]
    assert str(observed['reference']).isdigit() and int(observed['reference']) > 0
    if records:
        record = records[0]; assert str(record['reference']) == observed['reference']
    else:
        record = dict(trial=trial, marker=marker(trial), kernel=entry['kernel'], version=entry['version'],
            reconciled_utc=stamp(), qualification_sha256=sha(QUALIFICATION),
            csv_sha256=entry['csv_sha256'], remote_scientific_sha256=remote_sha)
        ledger['submissions'].append(record)
    record.update(observed, actual_checked_utc=stamp())
    intent.update(resolved=True, reference=record['reference'])
    validate_ledger(ledger, sha(QUALIFICATION))
    if not dry: persist(ledger)
    print('C76 EXISTING ACCEPTED ENTRY RECONCILED', json.dumps(record), flush=True)
    return True


def submit(trial, dry=False):
    with shared_lock():
        q = qualified(); qsha = sha(QUALIFICATION)
        ledger = read(LEDGER) if LEDGER.exists() else empty_ledger()
        validate_ledger(ledger, qsha)
        active = authorization()
        assert active['submissions_used'] <= len(ledger['submissions']), 'Shared allowance mismatch'
        api = client(); raw = complete_history(api); rows = [normalize(r) for r in raw]
        entry = q['trials'][trial]; remote_sha = remote_verified(api, entry)
        if reconcile(trial, rows, ledger, entry, remote_sha, dry): return
        new_release_allowed(trial, rows, ledger)
        dates = []
        for item in raw:
            date = item.date
            assert isinstance(date, datetime), 'Unknown account timestamp; inspect before release'
            dates.append(date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date.astimezone(timezone.utc))
        assert sum(d.date() == datetime.now(timezone.utc).date() for d in dates) < 5, 'Daily account limit reached'
        if dry:
            print('C76 READY; NOT SUBMITTED', json.dumps(dict(trial=trial, kernel=entry['kernel'], version=entry['version'])), flush=True); return
        intent = dict(created_utc=stamp(), trial=trial, marker=marker(trial), resolved=False,
            kernel=entry['kernel'], version=entry['version'], qualification_sha256=qsha,
            csv_sha256=entry['csv_sha256'], remote_scientific_sha256=remote_sha)
        ledger['intents'].append(intent); durable_write(LEDGER, ledger)
        try:
            result = api.competition_submit_code('submission.csv',
                marker(trial) + ': Approved new ' + trial + '-fitted direct isomer ranker; frozen complete R3 formate-compatible parent and C76 numerical adapter. Audited complete source, once-only paired stress confirmation and actual T4 production. Exploratory transfer trial.',
                competition=COMP, kernel=entry['kernel'], kernel_version=entry['version'])
            reference = str(result.ref); assert reference.isdigit() and int(reference) > 0
        except Exception as error:
            response = getattr(error, 'response', None)
            durable_write(OUT / (trial + '-submission-attention.json'), dict(checked_utc=stamp(),
                error_type=type(error).__name__, http_status=None if response is None else response.status_code,
                response_body=None if response is None else response.text, intent_unresolved=True,
                automatic_retry=False, terminal_result_inferred=False))
            raise
        intent.update(resolved=True, reference=reference)
        record = dict(trial=trial, marker=marker(trial), reference=reference,
            kernel=entry['kernel'], version=entry['version'], qualification_sha256=qsha,
            csv_sha256=entry['csv_sha256'], remote_scientific_sha256=remote_sha,
            submitted_utc=stamp(), status='PENDING', success=False, public_score=None, error=None)
        ledger['submissions'].append(record); persist(ledger)
        print('C76 ACCEPTED; ACTUAL SCORE PENDING', json.dumps(record), flush=True)


def observe():
    with shared_lock():
        q = qualified(); ledger = read(LEDGER); validate_ledger(ledger, sha(QUALIFICATION))
        api = client(); rows = [normalize(r) for r in complete_history(api)]
        for record in ledger['submissions']:
            matches = [r for r in rows if r['reference'] == str(record['reference'])]
            assert len(matches) == 1, 'Accepted reference not visible; never infer completion'
            assert marker(record['trial']) in matches[0]['description']
            record.update(matches[0], actual_checked_utc=stamp())
        persist(ledger)
        records = {r['trial']: r for r in ledger['submissions']}
        assert records, 'No accepted entries to observe'
        failures = {t: r for t, r in records.items() if r['status'] in TERMINAL and not r['success']}
        if failures:
            durable_write(OUT / 'actual-scoring-attention.json', dict(checked_utc=stamp(), entries=failures,
                automatic_retry=False, accepted_entries=len(records), numeric_score_inferred=False))
        complete = set(records) == set(TRIALS) and all(r['success'] and r['status'] == 'COMPLETE'
            and r['public_score'] is not None and math.isfinite(r['public_score']) for r in records.values())
        if complete:
            result = dict(completed_utc=stamp(), family=FAMILY, qualification_sha256=sha(QUALIFICATION),
                actual_scored_entries=2, entries=records, no_third_entry=True,
                champion_0_point_480_artifacts_preserved=True,
                best_public_score_including_preserved_champion=max([.480] + [r['public_score'] for r in records.values()]))
            target = OUT / 'actual-final-results.json'
            if target.exists():
                old = read(target)
                assert old['qualification_sha256'] == sha(QUALIFICATION)
                assert {t: (r['reference'], r['public_score']) for t, r in old['entries'].items()} == {t: (r['reference'], r['public_score']) for t, r in records.items()}
            else: durable_write(target, result)
        snapshot = dict(checked_utc=stamp(), actual_scored_entries=sum(bool(r['success']) for r in records.values()),
            accepted_entries=len(records), entries=records, both_actual_numeric_scores_complete=complete,
            automatic_release=False, pending_account=[r['reference'] for r in rows if r['status'] not in TERMINAL])
        durable_write(OUT / 'actual-score-status.json', snapshot); print(json.dumps(snapshot), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['submit', 'observe'])
    parser.add_argument('trial', nargs='?', choices=TRIALS); parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.phase == 'submit': assert args.trial; submit(args.trial, args.dry_run)
    else: assert args.trial is None and not args.dry_run; observe()
