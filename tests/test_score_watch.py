import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('watch_score', Path(__file__).resolve().parents[1] / 'scripts/watch_score.py')
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


class ScoreWatchTests(unittest.TestCase):
    def test_tie_is_not_an_improvement(self):
        r = W.evaluate([dict(ref=1, status='SubmissionStatus.COMPLETE', public_score='.477')], .477)
        self.assertFalse(r['beaten'])

    def test_unfinished_or_invalid_score_cannot_win(self):
        rows = [dict(ref=1, status='PENDING', public_score='.9'),
                dict(ref=2, status='COMPLETE', public_score='NaN'),
                dict(ref=3, status='ERROR', public_score='.8')]
        r = W.evaluate(rows, .477)
        self.assertFalse(r['beaten'])
        self.assertIsNone(r['best'])

    def test_only_completed_strict_improvement_ends_the_watch(self):
        r = W.evaluate([dict(ref=4, status='SubmissionStatus.COMPLETE', public_score='.480')], .477)
        self.assertTrue(r['beaten'])
        self.assertEqual(r['best']['ref'], '4')


if __name__ == '__main__':
    unittest.main()
