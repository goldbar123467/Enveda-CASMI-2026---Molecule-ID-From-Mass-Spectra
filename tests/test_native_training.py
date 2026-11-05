import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None


@unittest.skipIf(np is None or pd is None, 'numeric environment required')
class NativeTrainingBoundaryTests(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            count = 1004
            data = types.SimpleNamespace(
                train=np.arange(1003), val=np.array([1003]),
                mol=pd.DataFrame({'key':[f'KEY{i:011d}' for i in range(count)]}),
                spec=pd.DataFrame({'mol_idx':np.tile(np.arange(count),2)}),
                by_mol={i:np.array([i,i+count]) for i in range(count)})
            before = {i:v.copy() for i,v in data.by_mol.items()}
            native = folder/'native.npy'
            np.save(native,np.r_[np.arange(1002),1003])
            cfg = dict(width=256,heads=8,layers=4)
            fake = types.SimpleNamespace(OVERRIDES={},BASE_SEED=20261023,OUT=folder,
                                         setup=lambda split:(cfg,data,'cuda'))
            calls = []
            def train(stage,seeds,epochs,split):
                calls.append((stage,seeds,epochs,split))
                fake.setup(split)
            fake.train = train
            path = Path(__file__).resolve().parents[1]/'scripts/native_training.py'
            spec = importlib.util.spec_from_file_location('tested_native_training',path)
            module = importlib.util.module_from_spec(spec)
            args = ['native_training.py','--mode',mode,'--seed','21','--epochs','128','--native-index',str(native)]
            with patch.dict(sys.modules,{'local_seeds':fake}), patch.object(sys,'argv',args):
                spec.loader.exec_module(module)
                module.main()
            self.assertEqual(calls,[('holdout',[21],128,'np')])
            np.testing.assert_array_equal(data.train,np.arange(1003))
            np.testing.assert_array_equal(data.val,[1003])
            np.testing.assert_array_equal(data.by_mol[1003],before[1003])
            np.testing.assert_array_equal(data.by_mol[1002],before[1002])
            manifest = json.loads((folder/'training_manifest.json').read_text())
            if mode == 'native':
                self.assertEqual(manifest['native_training_molecules'],1002)
                for i in range(1002):
                    np.testing.assert_array_equal(data.by_mol[i],[i])
            else:
                for i in range(count):
                    np.testing.assert_array_equal(data.by_mol[i],before[i])

    def test_native_changes_training_views_only(self):
        self.exercise('native')

    def test_general_keeps_original_views_and_objective(self):
        self.exercise('general')


if __name__ == '__main__':
    unittest.main()
