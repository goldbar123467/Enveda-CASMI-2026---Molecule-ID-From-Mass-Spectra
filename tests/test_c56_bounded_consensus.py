import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c56_bounded_consensus_tail import order,utilities


class BoundedConsensusTests(unittest.TestCase):
    def test_endpoints_formula_membership_and_pairwise_votes(self):
        rng=np.random.default_rng(56)
        for n in range(1,26):
            for _ in range(20):
                formula=rng.integers(0,3,n).tolist();a=rng.normal(size=n);b=rng.normal(size=n);valid=np.ones(n,bool)
                actual=order(formula,a,valid,reference_scores=b)
                self.assertEqual(actual[:2],list(range(min(2,n))));self.assertEqual(actual[10:],list(range(10,n)))
                self.assertEqual(sorted(actual),list(range(n)));self.assertEqual([formula[i] for i in actual],formula)
                for f in set(formula[2:10]):
                    positions=[i for i in range(2,min(10,n)) if formula[i]==f];own=utilities(positions,a);reference=utilities(positions,b);loc={v:i for i,v in enumerate(actual)}
                    for i in positions:
                        for j in positions:
                            if i<j and loc[j]<loc[i]:self.assertTrue(own[j]>own[i] and reference[j]>reference[i])
    def test_agreeing_heads_equal_cpu_rule(self):
        rng=np.random.default_rng(156)
        for _ in range(100):
            n=int(rng.integers(1,26));f=rng.integers(0,3,n).tolist();a=rng.normal(size=n);valid=rng.random(n)>.15
            self.assertEqual(order(f,a,valid),order(f,a,valid,reference_scores=a))
    def test_unsupported_and_missing_group_keep_parent(self):
        f=['f']*25;a=list(range(25));expected=list(range(25))
        self.assertEqual(order(f,a,[True]*25,False),expected)
        eligible=[True]*25;eligible[5]=False
        self.assertEqual(order(f,a,eligible),expected)


if __name__=='__main__':unittest.main()
