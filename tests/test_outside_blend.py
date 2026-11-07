import importlib.util
from pathlib import Path
import unittest
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd=np=None

@unittest.skipIf(pd is None,'Numerical environment required')
class OutsideBlendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('outside_blend',Path(__file__).resolve().parents[1]/'scripts/outside_blend.py')
        cls.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.mod)
    def frames(self):
        a=pd.DataFrame(dict(molecule_id=['a','a','a'],smiles=['C','CC','C'],neural_score=[1.,3.,2.],
            neural_cos=[1.,3.,2.],neural_fp=[1.,3.,2.],neural_std=[0.,0.,0.],neural_ok=[True]*3))
        b=a.copy();b['neural_score']=[3.,2.,1.]
        return a,b
    def test_duplicate_candidates_keep_occurrence_alignment(self):
        a,b=self.frames();result=self.mod.combine(a,b)
        np.testing.assert_allclose(result.neural_score,[-.0,.5/np.std([1,3,2]),-.5/np.std([1,3,2])],atol=1e-8)
        self.assertEqual(result.smiles.tolist(),a.smiles.tolist())
    def test_one_unusable_channel_falls_back_per_row(self):
        a,b=self.frames();b['neural_ok']=False
        result=self.mod.combine(a,b)
        np.testing.assert_allclose(result.neural_score,[-1/np.std([1,3,2]),1/np.std([1,3,2]),0])
        self.assertTrue(result.neural_ok.all())
    def test_both_unusable_channels_keep_invalid_flag_without_crashing(self):
        a,b=self.frames();a['neural_ok']=False;b['neural_ok']=False
        result=self.mod.combine(a,b)
        self.assertFalse(result.neural_ok.any());self.assertTrue((result.neural_score==0).all())
    def test_different_candidate_sets_fail(self):
        a,b=self.frames()
        with self.assertRaises(AssertionError):self.mod.combine(a,b.iloc[:2])

if __name__=='__main__':unittest.main()
