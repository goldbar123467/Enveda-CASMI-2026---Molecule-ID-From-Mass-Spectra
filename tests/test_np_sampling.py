from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np=None

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipIf(np is None,'numeric environment required')
class MixtureSamplingTests(unittest.TestCase):
    def test_mixture_normalizes_over_training_only(self):
        from np_sampling import mixture_probabilities
        p,audit=mixture_probabilities([1,3,4,5],['heldout_np','np','heldout','other','other','other'],{'heldout_np','np'})
        np.testing.assert_allclose(p,[.625,.125,.125,.125])
        self.assertEqual(audit['annotated_np_molecules'],1)
        self.assertAlmostEqual(audit['sampled_np_fraction'],.625)

    def test_missing_np_or_duplicate_training_fails(self):
        from np_sampling import mixture_probabilities
        for training,keys in [([0,0],{'a'}),([0,1],set())]:
            with self.assertRaises(AssertionError):
                mixture_probabilities(training,['a','b'],keys)

    def test_loop_patch_preserves_negative_sampler_and_loss(self):
        from np_sampling import patch_training_loop
        source="def train_stage():\n    model=SpectrumCandidateModel(cfg).to(device)\n    batch_ids=rng.choice(ids,min(cfg['batch_size'],len(ids)),replace=False)\n    neg=rng.choice(pool,16)\n    loss=retrieval+cfg['aux_weight']*auxiliary\n"
        patched=patch_training_loop(source)
        self.assertIn('neg=rng.choice(pool,16)',patched)
        self.assertIn("loss=retrieval+cfg['aux_weight']*auxiliary",patched)
        self.assertIn('model.load_state_dict(_INITIAL_STATE,strict=True)',patched)
        with self.assertRaises(AssertionError):
            patch_training_loop(source.replace('replace=False','replace=True'))


if __name__=='__main__':
    unittest.main()
