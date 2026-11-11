"""Independent numerical checks and training-only batch composition."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from crossmodal_heads import fit_ridge,ridge_predict,pair_batch,normalize_rows


class AlignmentTests(unittest.TestCase):
    def test_block_ridge_matches_independent_augmented_least_squares(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(51,5));y=rng.normal(size=(51,3));lam=.1
        got=fit_ridge(x,y,lam,block_size=7)
        a=np.column_stack([x,np.ones(len(x))]);penalty=np.eye(6)*np.sqrt(len(x)*lam);penalty[-1,-1]=0
        expected=np.linalg.lstsq(np.vstack([a,penalty]),np.vstack([y,np.zeros((6,3))]),rcond=None)[0]
        np.testing.assert_allclose(got,expected,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(got,fit_ridge(x,y,lam,block_size=100),rtol=1e-11,atol=1e-12)

    def test_ridge_bias_is_not_penalized(self):
        x=np.zeros((20,3));y=np.full((20,2),[2.,-3.])
        w=fit_ridge(x,y,100.)
        np.testing.assert_allclose(ridge_predict(x,w),y)

    def test_formula_partners_stay_unique_inside_the_fitting_population(self):
        formulas=np.asarray(['A']*20+['B']*20+['C']*20)
        groups={f:np.flatnonzero(formulas==f) for f in set(formulas)}
        rng=np.random.default_rng(9)
        for _ in range(30):
            ids=pair_batch(rng,len(formulas),formulas,groups,size=32,fraction=1.)
            self.assertEqual(len(set(ids)),32)
            self.assertTrue(np.all((ids>=0)&(ids<60)))

    def test_zero_or_nonfinite_vectors_are_not_invented_as_valid(self):
        with self.assertRaises(AssertionError):normalize_rows(np.zeros((2,3)))
        with self.assertRaises(AssertionError):normalize_rows(np.asarray([[np.nan,0.]]))
        np.testing.assert_allclose(np.linalg.norm(normalize_rows(np.asarray([[2.,3.],[4.,-1.]])),axis=1),1.)


if __name__=='__main__':unittest.main()
