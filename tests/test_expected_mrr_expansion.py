import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from expected_mrr_expansion import expand


class ExpectedMRRTests(unittest.TestCase):
    def setUp(self):
        self.before=['s'+str(i) for i in range(25)];self.parent=pd.DataFrame(dict(molecule_id=['a'],smiles=[';'.join(self.before)]))
        values=self.before+['extra'];self.pool=pd.DataFrame(dict(molecule_id='a',smiles=values,metric_key=values,prior_rank=range(1,27),
            is_parent=[True]*25+[False],library_max=.1,probability=[.9]+[0.]*24+[.05]))
        self.cfg=dict(protect_library_at=.9,minimum_expected_mrr_gain=1e-6)

    def test_insert_and_expected_gain_are_exact(self):
        result,actions=expand(self.parent,self.pool,self.cfg)
        self.assertEqual(result.smiles.iloc[0].split(';'),self.before[:1]+['extra']+self.before[1:24])
        self.assertAlmostEqual(actions.estimated_mrr_gain.iloc[0],.025)

    def test_cost_of_shifting_parent_rank2_blocks_insert(self):
        pool=self.pool.copy();pool.probability=[0.,.8]+[0.]*23+[.1]
        result,actions=expand(self.parent,pool,self.cfg)
        self.assertTrue(result.equals(self.parent));self.assertAlmostEqual(actions.estimated_mrr_gain.iloc[0],.05-.8/6)

    def test_short_parent_and_missing_or_invalid_posterior(self):
        parent=self.parent.assign(smiles='s0;s1');pool=self.pool.iloc[[0,1,25]].copy();pool.prior_rank=[1,2,3]
        self.assertEqual(expand(parent,pool,self.cfg)[0].smiles.iloc[0],'s0;extra;s1')
        for case in [self.pool.assign(library_max=.9),self.pool.assign(probability=float('nan')),self.pool.assign(probability=0.)]:
            self.assertTrue(expand(self.parent,case,self.cfg)[0].equals(self.parent))

    def test_invalid_alias_and_probability_mass_fail_closed(self):
        pool=self.pool.copy();pool.loc[25,'metric_key']='s0'
        with self.assertRaises(AssertionError):expand(self.parent,pool,self.cfg)
        with self.assertRaises(AssertionError):expand(self.parent,self.pool.assign(probability=.5),self.cfg)


if __name__=='__main__':unittest.main()
