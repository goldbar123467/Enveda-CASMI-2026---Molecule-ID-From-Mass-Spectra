import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from peak_dropout import drop_peaks


class PeakDropTests(unittest.TestCase):
    def batch(self,device='cpu'):
        x=torch.zeros((3,10,4),device=device);lengths=torch.tensor([3,6,10],device=device)
        for i,n in enumerate(lengths.tolist()):
            x[i,:n,0]=torch.arange(1,n+1,device=device)
            x[i,:n,1]=torch.arange(1,n+1,device=device)/n
            x[i,:n,2]=x[i,:n,1]*2;x[i,:n,3]=100-x[i,:n,0]
        return x,lengths,torch.ones((3,4),device=device),torch.arange(3,device=device)

    def test_full_drop_keeps_three_strongest_in_original_order(self):
        batch=self.batch();before=batch[0].clone();after,counts=drop_peaks(batch,1,torch.Generator().manual_seed(9))
        self.assertEqual(after[1].tolist(),[3,3,3]);self.assertEqual(counts['peaks_before'],19);self.assertEqual(counts['peaks_after'],9)
        for i,n in enumerate(batch[1].tolist()):self.assertTrue(torch.equal(after[0][i,:3],before[i,n-3:n]))
        self.assertFalse(after[0][:,3:].any());self.assertTrue(torch.equal(batch[0],before))
        self.assertIs(after[2],batch[2]);self.assertIs(after[3],batch[3])

    def test_zero_drop_is_exact(self):
        batch=self.batch();after,_=drop_peaks(batch,0,torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(after[0],batch[0]));self.assertTrue(torch.equal(after[1],batch[1]))

    def test_checkpoint_rng_restore_and_global_rng_independence(self):
        batch=self.batch();rng=torch.Generator().manual_seed(12);global_before=torch.get_rng_state();saved=rng.get_state()
        a,_=drop_peaks(batch,.4,rng);rng.set_state(saved);b,_=drop_peaks(batch,.4,rng)
        self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(global_before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
