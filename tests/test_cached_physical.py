import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipIf(np is None,'numeric environment required')
class CachedPhysicalTests(unittest.TestCase):
    def test_subset_reordering_baseline_and_input_mismatch(self):
        from cached_physical import CachedFragment,spectrum_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spectra = [dict(adduct='[M+H]+',prec=100.,mode=1,mz=[40.,60.],it=[1.,.5])]
            digest = spectrum_digest(spectra)
            (root/'fragment_lookup.json').write_text(json.dumps({'m':dict(candidates=['A','B'],lookups={digest+'_opt':'opt',digest+'_old':'old'})}))
            np.savez(root/'fragment_scores.npz',opt=[.2,.4],old=[.1,np.nan])
            frag = CachedFragment(root)
            frag.mid = 'm'
            self.assertEqual(frag.score_molecule(spectra,['B','A']),[.4,.2])
            self.assertEqual(frag.score_molecule(spectra,['B','A'],cfg=frag.BASELINE_CFG),[None,.1])
            with self.assertRaises(KeyError):
                frag.score_molecule([dict(spectra[0],prec=101.)],['A'])
            with self.assertRaises(KeyError):
                frag.score_molecule(spectra,['OUTSIDE'])

    def test_only_hash_pinned_literal_module_executes(self):
        from cached_physical import physical_module
        source = 'VALUE = 42\n'
        digest = hashlib.sha256(source.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'notebook.ipynb'
            path.write_text(json.dumps({'cells':[dict(cell_type='code',source=[f'_post = {source!r}\nraise RuntimeError("notebook must not execute")'])]}))
            self.assertEqual(physical_module(path,digest).VALUE,42)
            with self.assertRaises(AssertionError):
                physical_module(path,'0'*64)


if __name__ == '__main__':
    unittest.main()
