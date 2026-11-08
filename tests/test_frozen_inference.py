"""Training/serving range and score-scale contracts for the frozen channel."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch  # Encoder environment requires Torch DLLs before NumPy.
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from score_frozen_channel import preprocess,fuse
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Torch/NumPy inference environment required')
class FrozenInferenceTests(unittest.TestCase):
    def query(self,**overrides):
        fields=dict(ionization_mode='positive',adduct='[M+H]+',precursor_mz=500.,
                    ms2_mzs=[100.,200.,300.,1500.],ms2_normalized_intensities=[1.,2.,3.,999.])
        fields.update(overrides);return SimpleNamespace(**fields)

    def test_out_of_range_base_peak_is_removed_before_normalizing(self):
        a=preprocess(self.query())
        self.assertEqual(a.shape,(101,2));self.assertEqual(a.dtype,np.float32)
        self.assertEqual(float(a[:,0].max()),500.)
        np.testing.assert_allclose(a[1:4,1],[1/3,2/3,1],atol=1e-7)
        self.assertAlmostEqual(float(a[0,1]),1.1,places=6)

    def test_unsupported_queries_skip_before_decoding_peak_arrays(self):
        for fields in [dict(ionization_mode='negative'),dict(adduct='[M+2H]2+'),dict(precursor_mz=1000.01)]:
            self.assertIsNone(preprocess(self.query(ms2_mzs=None,**fields)))

    def test_range_filter_cannot_create_a_usable_spectrum_with_two_peaks(self):
        self.assertIsNone(preprocess(self.query(ms2_mzs=[100.,200.,1500.],ms2_normalized_intensities=[1.,2.,3.])))

    def test_equal_intensity_cutoff_has_an_explicit_order(self):
        value=preprocess(self.query(ms2_mzs=np.arange(10.,111.),ms2_normalized_intensities=np.ones(101)))
        np.testing.assert_array_equal(value[1:,0],np.arange(11.,111.))

    def test_new_score_temperature_and_offset_do_not_change_fusion(self):
        a=np.array([3.,1.,-2.,7.]);b=np.array([.4,2.,-1.,.7])
        np.testing.assert_allclose(fuse(a,b,.5),fuse(a,13*b+21,.5),atol=1e-12)

    def test_identical_ranker_keeps_the_original_score_scale(self):
        a=np.array([3.,1.,-2.,7.])
        np.testing.assert_allclose(fuse(a,13*a+21,.5),a,atol=1e-12)
        np.testing.assert_array_equal(fuse(a,np.ones(4),0),a)


if __name__=='__main__':unittest.main()
