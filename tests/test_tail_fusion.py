import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from tail_fusion import baseline_order, paired_summary, partition, passes, reciprocal_rank, tail_order


@unittest.skipIf(np is None, "Numerical tests require .venv-gpu")
class TailFusionTests(unittest.TestCase):
    def test_new_model_cannot_change_the_public_leader(self):
        public = np.arange(25, dtype=float)
        ours = -public
        for alpha in [0, .1, .5, 1]:
            order = tail_order(public, ours, alpha)
            self.assertEqual(order[0], baseline_order(public)[0])
            self.assertEqual(sorted(order.tolist()), list(range(25)))

    def test_zero_alpha_exactly_reproduces_baseline_even_with_constant_scores(self):
        for public in [np.zeros(25), np.arange(25.)]:
            np.testing.assert_array_equal(tail_order(public, np.ones(25), 0), baseline_order(public))

    def test_identity_partition_and_rank_25_cutoff(self):
        self.assertEqual(partition("same-key"), partition("same-key"))
        self.assertEqual(reciprocal_rank([0] * 25 + [1], np.arange(26)), 0)
        self.assertEqual(reciprocal_rank([0] * 24 + [1], np.arange(25)), 1/25)
        self.assertEqual(reciprocal_rank([0] * 25, np.arange(25)), 0)

    def test_gate_rejects_tie_small_panel_and_uncertain_gain(self):
        self.assertFalse(passes(paired_summary(np.ones(60), np.ones(60))))
        self.assertFalse(passes(paired_summary(np.zeros(10), np.ones(10))))
        self.assertFalse(passes(paired_summary(np.r_[1., np.zeros(59)], np.r_[0., .5, np.zeros(58)])))
        self.assertTrue(passes(paired_summary(np.zeros(60), np.ones(60))))

    def test_nonfinite_scores_cannot_silently_fall_back(self):
        with self.assertRaises(ValueError):
            tail_order(np.arange(25.), np.full(25, np.nan), .5)


if __name__ == "__main__":
    unittest.main()
