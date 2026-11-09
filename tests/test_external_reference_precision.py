import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_external_reference_smoke import independent_best
from external_reference_overlay import cosine, prepare


class ServedQueryPrecisionTests(unittest.TestCase):
    def test_audit_uses_float32_serving_representation_of_float64_storage(self):
        masses = np.array([100.0000003, 200.0000004, 300.0000007, 400.0000008])
        intensity = np.array([.123456789, .987654321, .456789123, .234567891])
        query = pd.DataFrame(dict(molecule_id=['a'], precursor_mz=[500.], ionization_mode=['positive'],
                                  ms2_mzs=[masses], ms2_normalized_intensities=[intensity]))
        reference = pd.DataFrame(dict(precursor_mz=[500.], ionization_mode=['positive'], normalized_smiles=['CC']))
        rm = np.array([[100., 200., 300., 400.]], dtype=np.float32)
        ri = np.array([[.1, .9, .4, .2]], dtype=np.float32)
        actual = independent_best(query, reference, rm, ri)['a']
        served = cosine(prepare(masses.astype(np.float32), intensity.astype(np.float32), 500.), prepare(rm[0], ri[0]))
        self.assertAlmostEqual(actual[0], served[0], places=13)
        self.assertEqual(actual[1], 4)


if __name__ == '__main__':
    unittest.main()
