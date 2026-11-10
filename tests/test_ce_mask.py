import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ce_mask import mask_known_ce


class CollisionEnergyMaskTests(unittest.TestCase):
    def batch(self):
        meta=torch.tensor([[.2,.3,1.,1.],[.4,.5,1.,-1.],[.6,0.,0.,1.]])
        return (torch.zeros(3,4,4),torch.tensor([3,4,3]),meta,torch.tensor([1,2,1]))

    def test_full_mask_changes_only_known_energy_fields(self):
        before=self.batch();saved=before[2].clone()
        after,stats=mask_known_ce(before,1.,torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(before[2],saved));self.assertTrue(torch.equal(after[2][:,[0,3]],saved[:,[0,3]]))
        self.assertTrue(torch.equal(after[2][:,1:3],torch.zeros(3,2)))
        self.assertEqual(stats,dict(eligible=2,masked=2,unknown_before=1))
        for i in [0,1,3]:self.assertIs(before[i],after[i])

    def test_zero_probability_is_exact(self):
        before=self.batch();after,stats=mask_known_ce(before,0.,torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(before[2],after[2]));self.assertEqual(stats['masked'],0)

    def test_rng_resume_matches_without_changing_global_rng(self):
        generator=torch.Generator().manual_seed(51);batch=self.batch();global_state=torch.get_rng_state().clone()
        mask_known_ce(batch,.5,generator);state=generator.get_state().clone()
        expected=mask_known_ce(batch,.5,generator)
        resumed=torch.Generator();resumed.set_state(state);actual=mask_known_ce(batch,.5,resumed)
        self.assertTrue(torch.equal(expected[0][2],actual[0][2]));self.assertEqual(expected[1],actual[1])
        self.assertTrue(torch.equal(global_state,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
