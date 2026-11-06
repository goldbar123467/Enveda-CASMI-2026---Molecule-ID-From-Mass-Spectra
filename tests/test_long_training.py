"""Leakage boundaries, row indexes and intake filtering for the new long run."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


try:
    import numpy as np
    import torch
    L=load('long_training','scripts/long_training.py')
    E=load('build_external_cache','scripts/build_external_cache.py')
except ImportError:
    L=E=None


@unittest.skipIf(L is None,'Numerical GPU environment required')
class LongTrainingTests(unittest.TestCase):
    def test_duplicate_identity_always_uses_same_fold(self):
        keys=['AAAAAAAAAAAAAA','BBBBBBBBBBBBBB','AAAAAAAAAAAAAA']
        fold=[L.fold_for(k) for k in keys]
        self.assertEqual(fold[0],fold[2])
        for k in range(5):
            for external in (False,True):
                ids=L.training_ids(fold,k,2,external)
                self.assertTrue(all(fold[i]!=k for i in ids))
                if not external:self.assertTrue(all(i<2 for i in ids))

    def test_csr_rows_preserve_source_row_numbers(self):
        ids=np.array([2,0,2,1,0,2])
        index=L.RowsByMolecule(ids,4)
        for i in range(4):np.testing.assert_array_equal(index[i],np.flatnonzero(ids==i))
        subset=np.array([0,3,5])
        filtered=L.RowsByMolecule(ids[subset],4,subset)
        np.testing.assert_array_equal(filtered[2],[0,5])

    def test_conflicting_spectrum_labels_form_transitive_group(self):
        parent=np.arange(5)
        for a,b in [(3,4),(1,3),(2,1)]:L.merge_groups(parent,a,b)
        def root(i):
            while parent[i]!=i:i=parent[i]
            return i
        self.assertEqual({root(i) for i in [1,2,3,4]},{1})
        self.assertEqual(root(0),0)

    def test_nist_and_propagated_sources_rejected(self):
        for name in ['GNPS-NIST14-MATCHES','METLIN','GNPS-PROPOGATED','GNPS-SUSPECTLIST','DEREPLICATOR_IDENTIFIED_LIBRARY']:
            self.assertTrue(E.forbidden_library(name),name)
        self.assertFalse(E.forbidden_library('GNPS-NIH-NATURALPRODUCTSLIBRARY'))

    def test_nce_and_percent_not_fabricated_ev(self):
        for value in ['10(NCE)','35%','20-40','unknown',None]:self.assertEqual(E.collision_energy(value),[])
        self.assertEqual(E.collision_energy('20 eV'),[20.])

    def test_gnps_adduct_aliases_are_explicit(self):
        for source in ['M+H','[M+H]+','[M+H]1+','[M + H]+']:
            self.assertEqual(E.normalize_adduct(source),'[M+H]+')
        self.assertNotIn(E.normalize_adduct('[2M+H]+'),E.OFFSETS)

    @unittest.skipUnless(L is not None and torch.cuda.is_available(),'Real CUDA required')
    def test_interrupted_epoch_boundary_resumes_same_model(self):
        """Real optimizer/scaler/RNG replay, on synthetic data; no score claim."""
        import pandas as pd
        class TinyData:
            def __init__(self,cfg):
                self.base_count=20
                self.folds=np.arange(20)%5
                self.row_molecule=np.arange(20)
                self.mol=pd.DataFrame(dict(key=[f'key{i}' for i in range(20)],formula=['CH4']*20,mass=np.arange(20)))
                rng=np.random.default_rng(18)
                self.fp=rng.integers(0,2,(20,2048),dtype=np.uint8)
                self.by_mol=self.base_rows={i:np.array([i]) for i in range(20)}
                self.x=rng.random((20,8,4),dtype=np.float32)
            def query_rows(self,i):return np.array([i])
            def batch(self,rows,device,augment=False):
                return (torch.tensor(self.x[rows],device=device),torch.full((len(rows),),8,device=device,dtype=torch.long),
                        torch.zeros((len(rows),4),device=device),torch.ones(len(rows),device=device,dtype=torch.long))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            ext=root/'external';ext.mkdir();(ext/'manifest.json').write_text('{}')
            cfg=dict(seed=123,epochs=2,steps_per_epoch=2,batch_size=4,negative_pool=8,hard_negatives=2,
                     width=16,heads=2,layers=1,embedding=16,dropout=.1,fp_bits=2048,temperature=.1,
                     fp_score_weight=2.,lr=.0002,weight_decay=.01,aux_weight=.5,
                     minimum_free_ram_gib=0,external_cache=str(ext))
            data=TinyData(cfg)
            plan=dict(ids=np.array([0]),keys=np.array(['key0']),folds=np.array([0]),
                      candidates=np.array([[0,1,2]]),natural_product=np.array([False]))
            for name in ['continuous','resumed']:
                out=root/name;out.mkdir();np.savez_compressed(out/'evaluation.npz',**plan)
            # The timer permits one full epoch, then expires before epoch two.
            with patch.object(L.time,'monotonic',side_effect=[0.,0.,.1,2.]):
                with self.assertRaisesRegex(RuntimeError,'epoch boundary'):
                    L.fit(data,plan,cfg,root/'resumed',0,False,deadline=1.)
            self.assertTrue((root/'resumed/models/control/fold-0/resume.pt').exists())
            L.fit(data,plan,cfg,root/'resumed',0,False)
            L.fit(data,plan,cfg,root/'continuous',0,False)
            def weights(name):
                return torch.load(root/name/'models/control/fold-0/model.pt',map_location='cpu',weights_only=True)['state_dict']
            a,b=weights('resumed'),weights('continuous')
            for k in a:
                torch.testing.assert_close(a[k],b[k],rtol=0,atol=0,msg=k)


if __name__=='__main__':unittest.main()
