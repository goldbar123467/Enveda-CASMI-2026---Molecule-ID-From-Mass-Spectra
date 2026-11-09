"""Independent served-score arithmetic and derivative checks."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c09_train import served_logits


class ServedScoreTests(unittest.TestCase):
    def test_matches_independent_candidate_arithmetic_and_gradients(self):
        torch.manual_seed(9)
        z=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
        p=torch.randn(3,7,dtype=torch.float64,requires_grad=True)
        pos=torch.randint(0,2,(3,7)).double();m=torch.randn(3,5,dtype=torch.float64)
        neg=torch.randint(0,2,(3,4,7)).double();nm=torch.randn(3,4,5,dtype=torch.float64)
        cfg=dict(temperature=.1,fp_score_weight=2.,fp_bits=7)
        primary,hard=served_logits(z,p,pos,m,neg,nm,cfg)
        expected=torch.stack([torch.stack([(z[i]*m[j]).sum()/.1+2*(p[i]*pos[j]).sum()/7 for j in range(3)]) for i in range(3)])
        expected_hard=torch.stack([torch.stack([(z[i]*nm[i,j]).sum()/.1+2*(p[i]*neg[i,j]).sum()/7 for j in range(4)]) for i in range(3)])
        torch.testing.assert_close(primary,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(hard,expected_hard,rtol=1e-12,atol=1e-12)
        loss=torch.nn.functional.cross_entropy(torch.cat([primary,hard],1),torch.arange(3))
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad.abs().sum().item(),0.)


if __name__=='__main__':unittest.main()
