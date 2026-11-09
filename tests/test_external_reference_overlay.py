import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from external_reference_overlay import prepare, cosine, apply_overlay


class ExternalReferenceTests(unittest.TestCase):
    def test_peak_cannot_be_reused_to_inflate_cosine(self):
        score, count = cosine(prepare([100., 100.01], [1., 1.]), prepare([100.005], [1.]))
        self.assertEqual(count, 1)
        self.assertAlmostEqual(score, 2 ** -.5)

    def test_exact_scaled_spectrum_is_invariant(self):
        score, count = cosine(prepare([100., 200., 300.], [1., 4., 9.]), prepare([300., 100., 200.], [90., 10., 40.]))
        self.assertEqual(count, 3); self.assertAlmostEqual(score, 1.)

    def test_metric_alias_promotion_preserves_existing_representation(self):
        baseline = pd.DataFrame(dict(molecule_id=['a'], smiles=['wrong;tautomer1;other']))
        match = pd.DataFrame(dict(molecule_id=['a'], cosine=[.9], matched_peaks=[4], smiles=['tautomer2']))
        policy = dict(thresholds=dict(protect_pipeline_library_at=.9, sqrt_cosine=.8, minimum_matched_peaks=4))
        keys = dict(wrong='W', tautomer1='T', tautomer2='T', other='O')
        final, _ = apply_overlay(baseline, match, dict(a=.2), policy, keys.__getitem__)
        self.assertEqual(final.smiles.iloc[0], 'tautomer1;wrong;other')
        protected, _ = apply_overlay(baseline, match, dict(a=.95), policy, keys.__getitem__)
        missing, _ = apply_overlay(baseline, match, {}, policy, keys.__getitem__)
        self.assertTrue(protected.equals(baseline)); self.assertTrue(missing.equals(baseline))

    def test_invalid_external_structure_preserves_original_list(self):
        baseline = pd.DataFrame(dict(molecule_id=['a'], smiles=['x;y']))
        match = pd.DataFrame(dict(molecule_id=['a'], cosine=[.9], matched_peaks=[4], smiles=['invalid']))
        policy = dict(thresholds=dict(protect_pipeline_library_at=.9, sqrt_cosine=.8, minimum_matched_peaks=4))
        def identity(value):
            if value == 'invalid':
                raise ValueError('Invalid SMILES')
            return value
        final, audit = apply_overlay(baseline, match, dict(a=.2), policy, identity)
        self.assertTrue(final.equals(baseline))
        self.assertEqual(audit.reason.iloc[0], 'invalid_external_structure')


if __name__ == '__main__':
    unittest.main()
