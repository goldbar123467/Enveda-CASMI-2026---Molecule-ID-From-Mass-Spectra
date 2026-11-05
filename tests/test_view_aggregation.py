import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from view_aggregation import aggregate_views


@unittest.skipIf(np is None, 'Numerical tests require .venv-gpu')
class ViewAggregationTests(unittest.TestCase):
    def test_one_view_is_the_original_average(self):
        s = np.array([[1., 4.], [3., 2.]])
        np.testing.assert_array_equal(aggregate_views(s, [1, 1]), aggregate_views(s))

    def test_replicating_an_entire_view_does_not_change_its_weight(self):
        s = np.array([[1., 4.], [3., 2.], [8., 0.]])
        expected = aggregate_views(s, [1, 1, 2])
        repeated = np.vstack([s, s[:2], s[:2]])
        np.testing.assert_array_equal(aggregate_views(repeated, [1, 1, 2, 1, 1, 1, 1]), expected)
        self.assertFalse(np.array_equal(aggregate_views(repeated), expected))

    def test_invalid_input_does_not_silently_fall_back(self):
        with self.assertRaises(ValueError):
            aggregate_views([[1., float('nan')]])
        with self.assertRaises(ValueError):
            aggregate_views([[1.], [2.]], [1])


if __name__ == '__main__':
    unittest.main()
