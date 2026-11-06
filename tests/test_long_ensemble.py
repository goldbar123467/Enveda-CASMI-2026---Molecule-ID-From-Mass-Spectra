import importlib.util
from pathlib import Path
import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None


@unittest.skipIf(pd is None, 'Numerical checks use .venv-gpu')
class LongEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('score_long', Path(__file__).resolve().parents[1] / 'scripts/score_long_ensemble.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def frames(self):
        result = []
        for fold in range(5):
            result.append(pd.DataFrame(dict(molecule_id=['q', 'q'], smiles=['C', 'CC'],
                processed=[True, True], neural_ok=[True, True], n_spectra=[2, 2], n_adducts=[1, 1],
                neural_cos=[fold, fold + 1], neural_fp=[fold, fold + 1],
                neural_std=[fold, fold + 1], neural_score=[fold, fold + 1])))
        return result

    def test_mean_uses_all_folds_and_aligns_candidate_identity(self):
        frames = self.frames()
        frames[2] = frames[2].iloc[::-1]
        result = self.module.combine(frames)
        np.testing.assert_allclose(result.neural_score, [2, 3])
        self.assertEqual(result.smiles.tolist(), ['C', 'CC'])

    def test_missing_fold_or_candidate_fails(self):
        with self.assertRaises(AssertionError):
            self.module.combine(self.frames()[:-1])
        frames = self.frames();frames[3] = frames[3].iloc[:1]
        with self.assertRaises(AssertionError):
            self.module.combine(frames)

    def test_failed_or_nonfinite_scores_cannot_silently_be_averaged(self):
        frames = self.frames();frames[4].loc[0, 'neural_ok'] = False
        with self.assertRaises(AssertionError):
            self.module.combine(frames)
        frames = self.frames();frames[4]['neural_score'] = [float('nan'), 1]
        with self.assertRaises(AssertionError):
            self.module.combine(frames)


if __name__ == '__main__':
    unittest.main()
