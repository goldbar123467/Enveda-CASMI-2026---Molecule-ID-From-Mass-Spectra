import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from agreement_expansion import expand


class AgreementExpansionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(protect_library_at=.9,minimum_raw_top_margin=1e-4)
        self.before = ['s'+str(i) for i in range(25)]
        self.parent = pd.DataFrame(dict(molecule_id=['a'],smiles=[';'.join(self.before)]))
        smiles = self.before+['outsider']
        self.pool = pd.DataFrame(dict(molecule_id='a',smiles=smiles,metric_key=smiles,prior_rank=range(1,27),
            is_parent=[True]*25+[False],is_lotus=[False]*25+[True],library_max=.1,neural_ok=True,
            public_score=list(range(25))+[100.],v80_score=list(range(25))+[100.]))

    def test_joint_top_inserts_rank2_preserves_parent_prefix(self):
        frame,actions = expand(self.parent,self.pool,self.cfg)
        self.assertEqual(frame.smiles.iloc[0].split(';'),self.before[:1]+['outsider']+self.before[1:24])
        self.assertEqual(actions.inserted.iloc[0],1)

    def test_disagreement_parent_top_and_wrong_source_are_exact(self):
        cases = [self.pool.assign(v80_score=list(range(25))+[-1.]),
            self.pool.assign(public_score=list(range(25))+[-1.],v80_score=list(range(25))+[-1.]),
            self.pool.assign(is_lotus=False)]
        for pool in cases:self.assertTrue(expand(self.parent,pool,self.cfg)[0].equals(self.parent))

    def test_protected_missing_unsupported_and_near_tie_are_exact(self):
        cases = [self.pool.assign(library_max=.9),self.pool.assign(library_max=float('nan')),
            self.pool.assign(neural_ok=False),self.pool.assign(public_score=float('nan')),
            self.pool.assign(public_score=list(range(25))+[24.00001])]
        for pool in cases:self.assertTrue(expand(self.parent,pool,self.cfg)[0].equals(self.parent))

    def test_identity_alias_is_rejected(self):
        pool = self.pool.copy();pool.loc[25,'metric_key'] = 's0'
        with self.assertRaises(AssertionError):expand(self.parent,pool,self.cfg)

    def test_short_parent_and_unprocessed_fallback(self):
        parent = self.parent.assign(smiles='s0;s1');pool=self.pool.iloc[[0,1,25]].copy();pool.prior_rank=[1,2,3]
        self.assertEqual(expand(parent,pool,self.cfg)[0].smiles.iloc[0],'s0;outsider;s1')
        self.assertTrue(expand(self.parent.assign(molecule_id='missing'),self.pool,self.cfg)[0].equals(self.parent.assign(molecule_id='missing')))


if __name__ == '__main__':unittest.main()
