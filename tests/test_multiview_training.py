import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from multiview_training import select_mates,average_encoded


class MultiviewTests(unittest.TestCase):
    def setUp(self):
        meta=np.zeros((6,8));meta[:,3]=[1,-1,1,-1,1,-1]
        self.data=SimpleNamespace(order=np.arange(6),offset=np.array([0,4,5,6]),meta=meta)

    def test_mate_is_distinct_same_identity_and_polarity(self):
        second,use=select_mates(self.data,np.array([0,0]),np.array([0,1]),np.random.default_rng(2),1.)
        np.testing.assert_array_equal(second,[2,3]);self.assertTrue(use.all())

    def test_single_acquisition_and_zero_probability_are_exact(self):
        for probability,ids,rows in [(1.,[1,2],[4,5]),(0.,[0,0],[0,1])]:
            result,use=select_mates(self.data,np.array(ids),np.array(rows),np.random.default_rng(2),probability)
            np.testing.assert_array_equal(result,rows);self.assertFalse(use.any())

    def test_average_matches_equal_mean_served_score_and_gradient(self):
        z=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True);p=torch.tensor([[2.,3.],[4.,5.]],requires_grad=True)
        zz=torch.tensor([[0.,1.]],requires_grad=True);pp=torch.tensor([[6.,7.]],requires_grad=True)
        a,b=average_encoded(z,p,zz,pp,torch.tensor([0]));m=torch.tensor([.6,.8]);fp=torch.tensor([1.,0.])
        score=a@m+b@fp
        expected=((z[0]@m+p[0]@fp)+(zz[0]@m+pp[0]@fp))/2
        self.assertEqual(float(score[0].detach()),float(expected.detach()))
        self.assertTrue(torch.equal(a[1],z[1]));self.assertTrue(torch.equal(b[1],p[1]))
        score[0].backward();self.assertTrue(torch.equal(z.grad[0],m/2));self.assertTrue(torch.equal(zz.grad[0],m/2))
        self.assertLess(float(a[0].detach().norm()),1.)


if __name__=='__main__':unittest.main()
