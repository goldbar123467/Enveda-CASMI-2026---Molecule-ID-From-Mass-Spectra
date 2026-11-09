import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finish_c09_confirmation import metric_vector


class ConfirmationMetricTests(unittest.TestCase):
    def test_missing_truth_stays_in_denominator(self):
        frame = pd.DataFrame(dict(molecule_id=['a', 'b'], smiles=['x;y', 'x;y']))
        labels = pd.DataFrame(dict(molecule_id=['a', 'b'], key=['Y', 'ABSENT'])).set_index('molecule_id')
        result = metric_vector(frame, labels, dict(x='X', y='Y'))
        self.assertEqual(result.tolist(), [.5, 0.])
        self.assertEqual(result.mean(), .25)

    def test_official_identity_aliases_cannot_duplicate_ranks(self):
        frame = pd.DataFrame(dict(molecule_id=['a'], smiles=['tautomer1;tautomer2']))
        labels = pd.DataFrame(dict(molecule_id=['a'], key=['X'])).set_index('molecule_id')
        with self.assertRaises(AssertionError):
            metric_vector(frame, labels, dict(tautomer1='X', tautomer2='X'))

    def test_missing_molecule_is_not_silently_dropped(self):
        frame = pd.DataFrame(dict(molecule_id=['a'], smiles=['x']))
        labels = pd.DataFrame(dict(molecule_id=['a', 'b'], key=['X', 'Y'])).set_index('molecule_id')
        with self.assertRaises(AssertionError):
            metric_vector(frame, labels, dict(x='X'))


if __name__ == '__main__':
    unittest.main()
