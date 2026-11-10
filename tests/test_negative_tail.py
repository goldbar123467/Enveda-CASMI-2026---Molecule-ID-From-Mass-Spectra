import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from negative_tail import merge


class NegativeTailTests(unittest.TestCase):
    def setUp(self):
        self.values=['s'+str(i) for i in range(25)]
        self.parent=pd.DataFrame(dict(molecule_id=['positive','negative'],smiles=['a;b',';'.join(self.values)]))
        self.scores=pd.DataFrame(dict(molecule_id='negative',smiles=self.values,baseline_rank=range(1,26),c08_ok=True,c08_score=range(25)))
        self.modes={'positive':'positive','negative':'negative'}

    def test_positive_exact_and_negative_rank1_and_set_preserved(self):
        result,report=merge(self.parent,self.scores,self.modes,.2)
        self.assertEqual(result.smiles.iloc[0],'a;b')
        after=result.smiles.iloc[1].split(';')
        self.assertEqual(after[0],self.values[0]);self.assertEqual(set(after),set(self.values))
        self.assertNotEqual(after,self.values);self.assertEqual(report['positive_exact_passthrough'],1)

    def test_missing_or_unknown_modes_and_extra_scores_rejected(self):
        for modes in [{'positive':'positive'},{'positive':'positive','negative':'unknown'}]:
            with self.assertRaises(AssertionError):merge(self.parent,self.scores,modes,.2)
        with self.assertRaises(AssertionError):merge(self.parent,self.scores.assign(molecule_id='positive'),self.modes,.2)

    def test_unsupported_negative_is_explicit_exact_fallback(self):
        result,report=merge(self.parent,self.scores.assign(c08_ok=False),self.modes,.2)
        self.assertTrue(result.equals(self.parent));self.assertEqual(report['passthrough_molecules'],1)


if __name__=='__main__':unittest.main()
