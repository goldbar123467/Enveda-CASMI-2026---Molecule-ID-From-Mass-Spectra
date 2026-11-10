import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from empty_pool_recovery import augment


class EmptyPoolRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.pool=dict(mass=np.array([100.,200.]),keys=np.array(['old1','old2'],object),smiles=np.array(['old-a','old-b'],object),fp=np.array([[1,2],[3,4]],np.uint8))
        self.queries=pd.DataFrame(dict(molecule_id=['q'],precursor_mz=[500.],adduct=['H']))
        self.engine=SimpleNamespace(CFG=SimpleNamespace(PPM_FALLBACK=30.),pv=SimpleNamespace(neutral_mass=lambda mz,ad:mz-1),
            fp_and_mass=lambda s:(np.array([5,6],np.uint8),{'a':499.,'b':499.001,'far':700.}[s]),canon_key=lambda s:'key-'+s)

    def test_real_proposals_added_with_original_pool_unchanged(self):
        before=self.pool['fp'].copy()
        with tempfile.TemporaryDirectory() as d:r=augment(self.pool,self.queries,{'q':['a','b','far']},self.engine,Path(d)/'audit.json')
        self.assertEqual(r['mass'].tolist(),[100.,200.,499.,499.001]);self.assertTrue(np.array_equal(self.pool['fp'],before))
        self.assertEqual(r['keys'].tolist(),['old1','old2','key-a','key-b'])

    def test_covered_query_cannot_change_its_candidate_pool(self):
        q=self.queries.assign(precursor_mz=101.)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):augment(self.pool,q,{'q':['a']},self.engine,Path(d)/'audit.json')

    def test_no_valid_mass_match_fails_without_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):augment(self.pool,self.queries,{'q':['far']},self.engine,Path(d)/'audit.json')

    def test_wrong_fingerprint_shape_fails(self):
        self.engine.fp_and_mass=lambda s:(np.array([1],np.uint8),499.)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):augment(self.pool,self.queries,{'q':['a']},self.engine,Path(d)/'audit.json')


if __name__=='__main__':unittest.main()
