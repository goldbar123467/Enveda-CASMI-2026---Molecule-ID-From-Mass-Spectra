"""An unknown channel input is retained globally; missing supported inputs fail."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_c59_core_health import check_channel_keys


class SupportTests(unittest.TestCase):
    def test_documented_unsupported_channel_keeps_full_global_denominator(self):
        result=check_channel_keys(['a','b','unknown'],['a','b'],dict(a={},b={},unknown={}),dict(a={},b={}),['a','b','unknown'])
        self.assertTrue(result['all_original_queries_retained'])
    def test_missing_supported_query_fails(self):
        with self.assertRaisesRegex(AssertionError,'supported query is missing'):
            check_channel_keys(['a','b','unknown'],['a','b'],dict(a={},b={},unknown={}),dict(a={}),['a','b','unknown'])
    def test_dropping_unknown_from_global_evaluation_fails(self):
        with self.assertRaisesRegex(AssertionError,'Every original query'):
            check_channel_keys(['a','b','unknown'],['a','b'],dict(a={},b={},unknown={}),dict(a={},b={}),['a','b'])
    def test_silent_unknown_token_reinterpretation_fails(self):
        with self.assertRaisesRegex(AssertionError,'silently reinterpreted'):
            check_channel_keys(['a','b','unknown'],['a','b'],dict(a={},b={},unknown={}),dict(a={},b={},unknown={}),['a','b','unknown'])


if __name__=='__main__':unittest.main()
