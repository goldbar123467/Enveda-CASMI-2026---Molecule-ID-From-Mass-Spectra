import sys,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from polarity_loss import query_weights,weighted_loss


class PolarityLossTests(unittest.TestCase):
    def test_half_exposure_is_exact_original_loss_and_gradient(self):
        torch.manual_seed(19)
        logits=torch.randn(4,7,requires_grad=True);fp=torch.randn(4,9,requires_grad=True)
        target=torch.tensor([0,1,2,3]);truth=torch.randint(0,2,(4,9)).float();pw=torch.arange(1,10).float();pol=torch.tensor([1.,-1.,1.,-1.])
        original=(F.cross_entropy(logits,target)+.4*F.binary_cross_entropy_with_logits(fp,truth,pos_weight=pw))*.5
        changed=weighted_loss(logits,target,fp,truth,pw,pol,.5,.4,8)
        self.assertTrue(torch.allclose(original,changed,atol=1e-7))
        a=torch.autograd.grad(original,(logits,fp),retain_graph=True);b=torch.autograd.grad(changed,(logits,fp))
        self.assertTrue(all(torch.allclose(x,y,atol=1e-7) for x,y in zip(a,b)))

    def test_effective_query_exposure_balances_without_resampling(self):
        p=torch.tensor([-1.,1.,1.,1.,1.]);w=query_weights(p,.2)
        self.assertAlmostEqual(float(w[p<0].sum()/w.sum()),.5);self.assertAlmostEqual(float(w.mean()),1.)

    def test_unknown_polarity_rejected(self):
        with self.assertRaises(AssertionError):query_weights(torch.tensor([0.]),.2)


if __name__=='__main__':unittest.main()
