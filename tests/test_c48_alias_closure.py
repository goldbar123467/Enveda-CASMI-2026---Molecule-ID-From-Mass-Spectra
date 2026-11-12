"""Identity quarantine follows every raw/official alias, regardless of row order."""
import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_c48_fit_boundary import alias_closure


class ClosureTests(unittest.TestCase):
    def test_transitive_raw_official_chain_and_unrelated_component(self):
        edges=pd.DataFrame(dict(raw_key=['a','a','b','b','c','z'],official_key=['Q','B','B','C','C','Z']))
        for frame in [edges,edges.iloc[::-1]]:
            official,raw,_=alias_closure(frame,{'Q'},set())
            self.assertEqual(official,{'Q','B','C'});self.assertEqual(raw,{'a','b','c'})
    def test_invalid_structure_quarantines_other_labels_of_the_same_raw_source(self):
        edges=pd.DataFrame(dict(raw_key=['a','a','b','z'],official_key=[None,'B','B','Z']))
        official,raw,_=alias_closure(edges,set(),set())
        self.assertEqual(official,{'B'});self.assertEqual(raw,{'a','b'})
    def test_explicit_acquisition_alias_propagates_in_both_directions(self):
        edges=pd.DataFrame(dict(raw_key=['a','b','z'],official_key=['B','B','Z']))
        official,raw,_=alias_closure(edges,set(),{'b'})
        self.assertEqual(official,{'B'});self.assertEqual(raw,{'a','b'})


if __name__=='__main__':unittest.main()
