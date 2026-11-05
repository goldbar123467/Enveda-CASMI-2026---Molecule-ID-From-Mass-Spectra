import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from metric_identity import unique_metric_order


class MetricIdentityTests(unittest.TestCase):
    def test_deduplication_cannot_worsen_first_correct_identity_rank(self):
        keys = ['A'*14, 'A'*14, 'B'*14, 'C'*14, 'B'*14, 'D'*14]
        order = list(range(len(keys)))
        result = unique_metric_order(order, keys)
        self.assertEqual(result, [0, 2, 3, 5])
        for truth in set(keys):
            before = next(i for i, j in enumerate(order) if keys[j] == truth)
            after = next(i for i, j in enumerate(result) if keys[j] == truth)
            self.assertLessEqual(after, before)
        self.assertEqual(result[0], order[0])

    def test_unknown_or_invalid_identities_are_not_collapsed(self):
        keys = ['', '', None, 'nan', 'A'*14, 'A'*14]
        self.assertEqual(unique_metric_order(range(6), keys), [0, 1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
