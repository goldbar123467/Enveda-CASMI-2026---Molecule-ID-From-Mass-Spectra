import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c09_submission_guard import check,normalize


class SubmissionGuardTests(unittest.TestCase):
    def setUp(self):self.ledger=dict(campaign='C09',maximum_pending=1,intents=[])
    def test_missing_status_blocks_as_pending(self):
        with self.assertRaises(AssertionError):check([dict(ref=58163163)],self.ledger,'C09-P16-testmarker')
    def test_complete_empty_score_is_not_success(self):
        self.assertFalse(normalize(dict(ref=1,status='COMPLETE',publicScore=''))['success'])
        self.assertFalse(normalize(dict(ref=1,status='COMPLETE',publicScore='.5',errorDescription='failed'))['success'])
        self.assertTrue(normalize(dict(ref=1,status='COMPLETE',publicScore='.480'))['success'])
    def test_ambiguous_intent_prevents_retry(self):
        self.ledger['intents']=[dict(marker='C09-P16-testmarker')]
        with self.assertRaises(AssertionError):check([],self.ledger,'C09-P16-testmarker')
    def test_reconciled_marker_does_not_resubmit(self):
        result=check([dict(ref=1,status='COMPLETE',description='C09-P16-testmarker',publicScore='.48')],self.ledger,'C09-P16-testmarker')
        self.assertEqual(result['action'],'already_recorded')


if __name__=='__main__':unittest.main()
