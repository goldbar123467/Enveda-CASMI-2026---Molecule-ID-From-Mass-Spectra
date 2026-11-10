import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fold_agreement_expansion import expand


class FoldAgreementTests(unittest.TestCase):
    def setUp(self):
        self.before=['s'+str(i) for i in range(25)]
        self.parent=pd.DataFrame(dict(molecule_id=['a'],smiles=[';'.join(self.before)]))
        self.cfg=dict(protect_library_at=.9,minimum_raw_top_margin=1e-4)
        values=self.before+['extra']
        self.pool=pd.DataFrame(dict(molecule_id='a',smiles=values,metric_key=values,prior_rank=range(1,27),
            is_parent=[True]*25+[False],library_max=.1,c08_ok=True,c08_raw_fold0=list(range(25))+[100.],c08_raw_fold1=list(range(25))+[100.]))

    def test_agreement_inserts_rank2_and_drops_only_last_parent(self):
        result,actions=expand(self.parent,self.pool,self.cfg)
        self.assertEqual(result.smiles.iloc[0].split(';'),self.before[:1]+['extra']+self.before[1:24])
        self.assertEqual(actions.inserted.iloc[0],1)

    def test_disagreement_near_ties_protection_and_invalid_scores_pass_through(self):
        cases=[self.pool.assign(c08_raw_fold1=list(range(25))+[-1.]),self.pool.assign(c08_raw_fold0=list(range(25))+[24.00001]),
            self.pool.assign(library_max=.9),self.pool.assign(c08_ok=False),self.pool.assign(c08_raw_fold0=float('nan')),
            self.pool.assign(library_max=float('nan'))]
        for case in cases:self.assertTrue(expand(self.parent,case,self.cfg)[0].equals(self.parent))

    def test_metric_aliases_fail_closed(self):
        pool=self.pool.copy();pool.loc[25,'metric_key']='s0'
        with self.assertRaises(AssertionError):expand(self.parent,pool,self.cfg)

    def test_short_parent_and_missing_query_keep_expected_membership(self):
        parent=self.parent.assign(smiles='s0;s1');pool=self.pool.iloc[[0,1,25]].copy();pool.prior_rank=[1,2,3]
        self.assertEqual(expand(parent,pool,self.cfg)[0].smiles.iloc[0],'s0;extra;s1')
        absent=self.parent.assign(molecule_id='absent');self.assertTrue(expand(absent,self.pool,self.cfg)[0].equals(absent))


if __name__=='__main__':unittest.main()
