import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from update_runs import refresh, submission_run_status


class ScoringRunTests(unittest.TestCase):
    def test_omitted_status_remains_pending_without_kernel_lookup(self):
        row = dict(platform='kaggle-submission', status='pending', ref='58181280', result='', ended_utc='')
        with patch('update_runs.kaggle_status', side_effect=AssertionError('Must not inspect the completed commit kernel')):
            refresh([row], submission_rows=[dict(ref=58181280)])
        self.assertEqual(row['status'], 'pending')
        self.assertEqual(row['ended_utc'], '')

    def test_empty_or_error_complete_is_not_success_but_zero_score_is_valid(self):
        self.assertEqual(submission_run_status(dict(ref=1, status='COMPLETE', publicScore=''))[0], 'error')
        self.assertEqual(submission_run_status(dict(ref=1, status='COMPLETE', publicScore='.5', errorDescription='failed'))[0], 'error')
        self.assertEqual(submission_run_status(dict(ref=1, status='COMPLETE', publicScore='0'))[0], 'complete')

    def test_missing_account_record_does_not_infer_completion(self):
        row = dict(platform='kaggle-submission', status='running', ref='58181280', result='', ended_utc='')
        refresh([row], submission_rows=[])
        self.assertEqual(row['status'], 'running')
        self.assertEqual(row['ended_utc'], '')
        self.assertIn('no terminal state inferred', row['result'])


if __name__ == '__main__':
    unittest.main()
