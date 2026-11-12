"""Adversarial offline tests for the shared two-entry submission state machine.

No Kaggle client, live ledger, account mutation, labels or GPU is used.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import c76_submission_contract as contract
import submit_c76_pair as submission
from c76_account_history import complete_history
from c09_submission_guard import normalize


def intent(trial='cpu', resolved=False):
    row = dict(trial=trial, marker=contract.marker(trial), qualification_sha256='q',
        resolved=resolved, kernel='owner/' + trial, version=1, csv_sha256='csv', remote_scientific_sha256='source')
    if resolved: row['reference'] = '101' if trial == 'cpu' else '102'
    return row


def accepted(trial='cpu'):
    row = intent(trial, True); row.pop('resolved')
    row.update(status='PENDING', public_score=None, success=False, error=None)
    return row


def actual(status='COMPLETE', score=.481, error=None, trial='cpu'):
    return normalize(dict(ref='101' if trial == 'cpu' else '102', status=status,
        publicScore=score, errorDescription=error, description=contract.marker(trial) + ': approved'))


def ledger(cpu=False):
    value = submission.empty_ledger()
    if cpu: value.update(submissions=[accepted()], intents=[intent(resolved=True)])
    return value


def paged_history(pages):
    from contextlib import nullcontext
    from types import SimpleNamespace
    def get(request):
        assert request.page_size == 100
        rows, token = pages[request.page_token or '']
        return SimpleNamespace(submissions=rows, next_page_token=token)
    client = SimpleNamespace(competitions=SimpleNamespace(competition_api_client=SimpleNamespace(list_submissions=get)))
    return complete_history(SimpleNamespace(build_kaggle_client=lambda: nullcontext(client)))


class GuardTests(unittest.TestCase):
    def test_history_all_tokens_even_for_short_pages(self):
        raw = paged_history({'': ([dict(ref='100', status='COMPLETE', publicScore=.48)], 'tail'),
            'tail': ([dict(ref='99', status='RUNNING')], '')})
        self.assertEqual([r['ref'] for r in raw], ['100', '99'])
        with self.assertRaises(AssertionError): contract.new_release_allowed('cpu', [normalize(r) for r in raw], ledger())

    def test_history_tail_marker_cannot_be_overlooked(self):
        raw = paged_history({'': ([dict(ref='100', status='COMPLETE', publicScore=.48)], 'tail'),
            'tail': ([dict(ref='99', status='COMPLETE', publicScore=.47, description=contract.marker('cpu'))], '')})
        with self.assertRaises(AssertionError): contract.new_release_allowed('cpu', [normalize(r) for r in raw], ledger())

    def test_history_repeated_tokens_references_or_invalid_rows_rejected(self):
        for pages in [
            {'': ([dict(ref='100')], 'tail'), 'tail': ([dict(ref='99')], 'tail')},
            {'': ([dict(ref='100')], 'tail'), 'tail': ([dict(ref='100')], '')},
            {'': ([dict(ref='invalid')], '')}, {'': ([None], '')}]:
            with self.assertRaises(AssertionError): paged_history(pages)

    def test_first_cpu_only(self):
        contract.validate_ledger(ledger(), 'q')
        contract.new_release_allowed('cpu', [], ledger())
        with self.assertRaises(AssertionError): contract.new_release_allowed('gpu', [], ledger())

    def test_pending_anywhere_blocks(self):
        for status in ['PENDING', 'RUNNING', 'CANCEL_REQUESTED', '']:
            with self.subTest(status=status), self.assertRaises(AssertionError):
                contract.new_release_allowed('cpu', [actual(status=status)], ledger())

    def test_gpu_requires_actual_terminal_numeric_score(self):
        for status, score, error in [('PENDING', .49, None), ('COMPLETE', None, None),
            ('COMPLETE', float('nan'), None), ('COMPLETE', float('inf'), None),
            ('ERROR', .49, None), ('CANCELLED', .49, None), ('COMPLETE', .49, 'failure')]:
            with self.subTest(status=status, score=score), self.assertRaises(AssertionError):
                contract.new_release_allowed('gpu', [actual(status, score, error)], ledger(True))
        with self.assertRaises(AssertionError): contract.new_release_allowed('gpu', [], ledger(True))
        value = ledger(True)
        contract.new_release_allowed('gpu', [actual(score=.470)], value)
        self.assertEqual(value['submissions'][0]['public_score'], .470)

    def test_unresolved_intent_blocks_retry(self):
        value = ledger(); value['intents'].append(intent())
        contract.validate_ledger(value, 'q')
        with self.assertRaises(AssertionError): contract.new_release_allowed('cpu', [], value)

    def test_accepted_error_is_consumed(self):
        value = ledger(True); value['submissions'][0].update(actual('ERROR', None, 'runtime'))
        with self.assertRaises(AssertionError): contract.new_release_allowed('cpu', [], value)
        with self.assertRaises(AssertionError): contract.new_release_allowed('gpu', [actual('ERROR', None, 'runtime')], value)

    def test_no_third_or_duplicate_entry(self):
        value = ledger(True); value['submissions'].append(accepted('gpu')); value['intents'].append(intent('gpu', True))
        contract.validate_ledger(value, 'q')
        for trial in contract.TRIALS:
            with self.assertRaises(AssertionError): contract.new_release_allowed(trial, [actual(), actual(trial='gpu')], value)
        value['submissions'].append(accepted('gpu'))
        with self.assertRaises(AssertionError): contract.validate_ledger(value, 'q')

    def test_foreign_allowance_or_changed_qualification_rejected(self):
        value = ledger(True); value['family'] = 'retired_C42'
        with self.assertRaises(AssertionError): contract.validate_ledger(value, 'q')
        with self.assertRaises(AssertionError): contract.validate_ledger(ledger(True), 'other_qualification')

    def test_account_marker_without_intent_never_adopted(self):
        with patch.object(submission, 'sha', return_value='q'):
            with self.assertRaises(AssertionError):
                submission.reconcile('cpu', [actual()], ledger(), dict(kernel='owner/cpu', version=1, csv_sha256='csv'), 'source', True)
        with self.assertRaises(AssertionError): contract.new_release_allowed('cpu', [actual()], ledger())

    def test_ambiguous_acceptance_reconciles_once(self):
        value = ledger(); value['intents'].append(intent())
        entry = dict(kernel='owner/cpu', version=1, csv_sha256='csv')
        with patch.object(submission, 'sha', return_value='q'):
            self.assertTrue(submission.reconcile('cpu', [actual()], value, entry, 'source', True))
            self.assertEqual(len(value['submissions']), 1)
            self.assertTrue(submission.reconcile('cpu', [actual()], value, entry, 'source', True))
            self.assertEqual(len(value['submissions']), 1)
        contract.validate_ledger(value, 'q')

    def test_reconciliation_rejects_wrong_source_reference_and_missing_record(self):
        entry = dict(kernel='owner/cpu', version=1, csv_sha256='csv')
        value = ledger(); value['intents'].append(intent())
        with patch.object(submission, 'sha', return_value='q'):
            with self.assertRaises(AssertionError): submission.reconcile('cpu', [actual()], value, entry, 'changed', True)
            with self.assertRaises(AssertionError): submission.reconcile('cpu', [actual(), actual()], value, entry, 'source', True)
            with self.assertRaises(AssertionError): submission.reconcile('cpu', [], ledger(True), entry, 'source', True)
            wrong = actual(); wrong['reference'] = '999'
            with self.assertRaises(AssertionError): submission.reconcile('cpu', [wrong], ledger(True), entry, 'source', True)

    def test_resolved_intent_must_match_record(self):
        for field, value in [('reference', '999'), ('kernel', 'other/kernel'), ('csv_sha256', 'other')]:
            state = ledger(True); state['intents'][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError): contract.validate_ledger(state, 'q')
        state = ledger(); state['intents'] = [intent(resolved=True)]
        with self.assertRaises(AssertionError): contract.validate_ledger(state, 'q')

    def test_success_is_accepted_once_even_before_a_score_exists(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); ledger_path = folder / 'ledger.json'
            active_path = folder / 'active.json'
            class API:
                calls = 0
                def competition_submissions(self, comp):
                    return [dict(ref='101', status='PENDING', publicScore=None,
                        description=contract.marker('cpu') + ': approved')] if API.calls else []
                def competition_submit_code(self, *args, **kw):
                    API.calls += 1
                    assert not submission.read(ledger_path)['intents'][0]['resolved']
                    from types import SimpleNamespace
                    return SimpleNamespace(ref=101)
            entry = dict(kernel='owner/cpu', version=1, csv_sha256='csv')
            from contextlib import nullcontext
            with patch.object(submission, 'LEDGER', ledger_path), patch.object(submission, 'ACTIVE', active_path), \
                patch.object(submission, 'OUT', folder), patch.object(submission, 'shared_lock', nullcontext), \
                patch.object(submission, 'qualified', return_value={'trials': {'cpu': entry}}), \
                patch.object(submission, 'authorization', return_value={'submissions_used': 0}), \
                patch.object(submission, 'client', return_value=API()), patch.object(submission, 'remote_verified', return_value='source'), \
                patch.object(submission, 'complete_history', side_effect=lambda api: api.competition_submissions(contract.COMP)), \
                patch.object(submission, 'sha', return_value='q'), patch.object(submission, 'register'):
                submission.submit('cpu')
                saved = submission.read(ledger_path)
                self.assertEqual(len(saved['submissions']), 1)
                self.assertIsNone(saved['submissions'][0]['public_score'])
                self.assertTrue(saved['intents'][0]['resolved'])
                submission.submit('cpu')
                self.assertEqual(API.calls, 1)
                self.assertEqual(submission.read(active_path)['submissions_used'], 1)

    def test_durable_intent_written_before_submit_and_exception_no_retry(self):
        # Exercise the real submission function through its irreversible boundary
        # with a fake API. The fake observes the on-disk unresolved intent before
        # simulating a lost response. A second invocation must not call submit.
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); ledger_path = folder / 'ledger.json'
            class API:
                calls = 0
                def competition_submissions(self, comp): return []
                def competition_submit_code(self, *args, **kw):
                    API.calls += 1
                    saved = submission.read(ledger_path)
                    assert len(saved['intents']) == 1 and not saved['intents'][0]['resolved']
                    raise ConnectionError('Simulated lost response after request')
            entry = dict(kernel='owner/cpu', version=1, csv_sha256='csv')
            from contextlib import nullcontext
            with patch.object(submission, 'LEDGER', ledger_path), patch.object(submission, 'OUT', folder), \
                patch.object(submission, 'shared_lock', nullcontext), patch.object(submission, 'qualified', return_value={'trials': {'cpu': entry}}), \
                patch.object(submission, 'authorization', return_value={'submissions_used': 0}), \
                patch.object(submission, 'client', return_value=API()), patch.object(submission, 'remote_verified', return_value='source'), \
                patch.object(submission, 'complete_history', side_effect=lambda api: api.competition_submissions(contract.COMP)), \
                patch.object(submission, 'sha', return_value='q'):
                with self.assertRaises(ConnectionError): submission.submit('cpu')
                with self.assertRaises(AssertionError): submission.submit('cpu')
                self.assertEqual(API.calls, 1)
                self.assertEqual(len(submission.read(ledger_path)['submissions']), 0)


if __name__ == '__main__': unittest.main(verbosity=2)
