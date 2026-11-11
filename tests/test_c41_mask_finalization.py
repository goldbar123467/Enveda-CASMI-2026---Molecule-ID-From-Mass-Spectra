import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finalize_c41_reference_masks import finalize


class GlobalAliasMaskTests(unittest.TestCase):
    def test_late_discovery_excludes_earlier_aliases(self):
        frame=pd.DataFrame(dict(raw_key=['early','late','other'],official_key=['same','same','different'],exclude=[False,True,False]))
        result,blocked,raw=finalize(frame)
        self.assertEqual(blocked,{'same'})
        self.assertEqual(result.exclude.tolist(),[True,True,False])
        self.assertTrue(raw['early']);self.assertTrue(raw['late']);self.assertFalse(raw['other'])

    def test_row_order_does_not_change_final_raw_key_mask(self):
        frame=pd.DataFrame(dict(raw_key=['early','late','other'],official_key=['same','same','different'],exclude=[False,True,False]))
        a=finalize(frame)[2].sort_index();b=finalize(frame.iloc[::-1])[2].sort_index()
        pd.testing.assert_series_equal(a,b)

    def test_unparseable_structure_remains_quarantined(self):
        result,_,raw=finalize(pd.DataFrame(dict(raw_key=['unknown'],official_key=[None],exclude=[False])))
        self.assertTrue(result.exclude.iloc[0]);self.assertTrue(raw['unknown'])


if __name__=='__main__':unittest.main()
