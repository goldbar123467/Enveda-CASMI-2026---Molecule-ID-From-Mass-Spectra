import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from crossmodal_tail import order


class FrozenTailTests(unittest.TestCase):
    def test_first_choice_membership_and_formula_slots_are_invariants(self):
        formulas=['A' if i%2 else 'B' for i in range(25)]
        result=order(formulas,np.arange(25.),[True]*25)
        self.assertEqual(result[0],0)
        self.assertEqual(set(result),set(range(25)))
        self.assertEqual([formulas[i] for i in result],formulas)
        self.assertNotEqual(result,list(range(25)))

    def test_unsupported_query_and_incomplete_groups_keep_parent_order(self):
        formulas=['A']*25;scores=list(range(25))
        self.assertEqual(order(formulas,scores,[True]*25,supported=False),list(range(25)))
        eligible=[True]*25;eligible[2]=False
        self.assertEqual(order(formulas,scores,eligible),list(range(25)))

    def test_constant_scores_do_not_create_an_arbitrary_change(self):
        self.assertEqual(order(['A']*25,[1.]*25,[True]*25),list(range(25)))


if __name__=='__main__':unittest.main()
