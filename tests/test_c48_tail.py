"""CPU remains exactly compatible; GPU cannot demote either protected candidate."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c48_tail import order
from crossmodal_tail import order as original


class TailTests(unittest.TestCase):
    def test_cpu_rule_matches_original_for_random_formula_groups_and_fallbacks(self):
        rng=np.random.default_rng(411)
        for _ in range(300):
            n=int(rng.integers(1,26));f=rng.integers(0,4,n).tolist();scores=rng.normal(size=n);eligible=rng.random(n)>.15;supported=bool(rng.integers(0,2))
            self.assertEqual(order(f,scores,eligible,supported,protected_prefix=1),original(f,scores,eligible,supported))
    def test_gpu_top2_stays_fixed_while_later_isomers_can_change(self):
        self.assertEqual(order(['f']*7,[0,0,0,0,0,0,99],[True]*7,protected_prefix=2),[0,1,2,3,4,6,5])
    def test_one_ineligible_candidate_keeps_the_whole_formula_tail(self):
        self.assertEqual(order(['f']*7,[0,0,0,0,0,0,99],[True]*6+[False],protected_prefix=2),list(range(7)))
    def test_short_parent_lists_and_unsupported_queries_are_exact(self):
        for n in [1,2]:self.assertEqual(order(['f']*n,[99]*n,[True]*n,protected_prefix=2),list(range(n)))
        self.assertEqual(order(['f']*7,list(range(7)),[True]*7,False,protected_prefix=2),list(range(7)))


if __name__=='__main__':unittest.main()
