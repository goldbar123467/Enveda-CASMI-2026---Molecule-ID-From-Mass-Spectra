import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from confirm_c10_candidate import statistical_gate


class C10ConfirmationGateTests(unittest.TestCase):
    def setUp(self):
        self.comparisons = {name: dict(n=120, delta=.04, ci95=[.01, .07]) for name in [
            'winner_vs_P01', 'winner_vs_P16', 'alias_excluded_winner_vs_P01', 'alias_excluded_winner_vs_P16']}

    def test_gain_over_p01_without_increment_over_p16_is_rejected(self):
        self.comparisons['winner_vs_P16'] = dict(n=120, delta=0., ci95=[0., 0.])
        result = statistical_gate(self.comparisons)
        self.assertFalse(result['passed']); self.assertEqual(result['failed_comparisons'], ['winner_vs_P16'])

    def test_alias_sensitivity_reversal_blocks_an_apparent_gain(self):
        self.comparisons['alias_excluded_winner_vs_P16'] = dict(n=120, delta=-.01, ci95=[-.02, 0.])
        self.assertFalse(statistical_gate(self.comparisons)['passed'])

    def test_zero_boundary_is_not_positive_support(self):
        self.comparisons['winner_vs_P16']['ci95'][0] = 0.
        self.assertFalse(statistical_gate(self.comparisons)['passed'])

    def test_both_controls_and_nonnegative_sensitivity_pass(self):
        self.comparisons['alias_excluded_winner_vs_P16'] = dict(n=120, delta=0., ci95=[0., 0.])
        self.assertTrue(statistical_gate(self.comparisons)['passed'])

    def test_partial_or_nonfinite_panels_are_rejected(self):
        for change in [dict(n=119), dict(delta=float('nan'))]:
            comparisons = copy.deepcopy(self.comparisons)
            comparisons['winner_vs_P16'].update(change)
            with self.assertRaises(AssertionError): statistical_gate(comparisons)


if __name__ == '__main__': unittest.main()
