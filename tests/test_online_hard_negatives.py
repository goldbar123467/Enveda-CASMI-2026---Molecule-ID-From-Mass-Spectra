import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from online_hard_negatives import hardest_indices,mine


class HardMiningTests(unittest.TestCase):
    def test_stable_ties_and_positive_exclusion(self):
        values=torch.tensor([[9.,2.,2.,1.]])
        mask=torch.tensor([[True,False,False,False]])
        self.assertEqual(hardest_indices(values,mask,2).tolist(),[[1,2]])

    def test_mining_restores_mode_masks_equal_fingerprints_and_has_no_gradients(self):
        class Fake(torch.nn.Module):
            def encode_spectra(self,x):return x,torch.tensor([[1.,.5,0.]])
            def encode_molecules(self,x):return x[:,:2]
        model=Fake().train();batch=(torch.tensor([[1.,0.]],requires_grad=True),)
        positive=torch.tensor([[1.,1.,0.]])
        candidates=torch.tensor([[[1.,1.,0.],[.8,.1,0.],[.2,.9,0.],[.1,.1,0.]]],requires_grad=True)
        selected,mask,stats=mine(model,batch,positive,candidates,torch.zeros(1,4,dtype=torch.bool),
            dict(temperature=.1,fp_score_weight=2.,fp_bits=3),2)
        torch.testing.assert_close(selected,candidates[:,[1,2]])
        self.assertFalse(mask.any());self.assertTrue(model.training);self.assertFalse(selected.requires_grad)
        self.assertEqual(stats['equivalent_fingerprints_masked'],1);self.assertGreaterEqual(stats['mining_gain'],0.)
        self.assertIsNone(candidates.grad)


if __name__=='__main__':unittest.main()
