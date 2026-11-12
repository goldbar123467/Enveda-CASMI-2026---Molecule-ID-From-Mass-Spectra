"""Check the independent chemistry guard and measured-hit reconstruction."""
import sys,unittest
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import RDLogger
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import c45_complete_parent as c45


class CompleteParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RDLogger.DisableLog('rdApp.*');c45.frozen_contract()
        cls.features=c45.module('c45_test_features','c09e_features.py')
        cls.queries=[c45.descriptor(s,cls.features) for s in ['CCO','Oc1ccccc1','CCCCCCCC']]

    def flags(self,smiles,known=()):
        q=self.queries
        return c45.chemistry_flags(c45.descriptor(smiles,self.features),{d['key'] for d in q},
            {d['formula'] for d in q},{d['scaffold'] for d in q if d['scaffold']},[d['fp'] for d in q],set(known))

    def test_official_alias_is_blocked(self):self.assertTrue(self.flags('OCC')['query_identity'])
    def test_different_isomer_with_same_formula_is_blocked(self):
        f=self.flags('COC');self.assertFalse(f['query_identity']);self.assertTrue(f['same_formula'])
    def test_nonempty_scaffold_is_blocked(self):self.assertTrue(self.flags('Cc1ccccc1')['same_nonempty_scaffold'])
    def test_acyclic_neighbor_is_blocked_by_similarity(self):
        f=self.flags('CCCCCCCCC');self.assertFalse(f['same_formula']);self.assertFalse(f['same_nonempty_scaffold']);self.assertTrue(f['near_query_tanimoto'])
    def test_unrelated_reference_stays_eligible(self):
        f=self.flags('C[Si](C)(C)C')
        self.assertFalse(any(value for key,value in f.items() if key!='maximum_query_tanimoto'))
    def test_prior_official_alias_quarantine_is_preserved(self):
        s='C[Si](C)(C)C';key=c45.descriptor(s,self.features)['key'];self.assertTrue(self.flags(s,[key])['known_official_alias'])
    def test_unparseable_structure_cannot_acquire_a_valid_identity(self):
        with self.assertRaises(ValueError):c45.descriptor('not_a_smiles',self.features)
    def test_overlay_can_change_rank_one_and_insert_without_head_changes(self):
        baseline=pd.DataFrame(dict(molecule_id=['a','b','c'],smiles=['x;y;z','x;y;z','x;y;z']))
        matches=pd.DataFrame(dict(molecule_id=['a','b','c'],cosine=[.9,.9,.9],matched_peaks=[5,5,5],smiles=['new','z','new']))
        result=c45.independent_overlay(baseline,matches,dict(a=.2,b=.2,c=.95),{'thresholds':{'protect_pipeline_library_at':.9,'sqrt_cosine':.8,'minimum_matched_peaks':4}},lambda s:s)
        self.assertEqual(result.smiles.tolist(),['new;x;y;z','z;x;y','x;y;z'])
        self.assertEqual(result.molecule_id.tolist(),baseline.molecule_id.tolist())
    def test_precursor_union_matches_direct_original_bounds_including_upper_tie(self):
        threshold=dict(precursor_ppm=10,precursor_absolute_da=.002)
        q=pd.DataFrame(dict(precursor_mz=[100.,500.]))
        width=100.*10e-6+.002
        values=np.asarray([501.,100.+width,100.-width,np.nextafter(100.+width,np.inf),500.,np.nan,90.])
        expected=np.zeros(len(values),bool)
        for center in q.precursor_mz:
            limit=center*10e-6+.002;expected|=(values>=center-limit)&(values<=center+limit)
        np.testing.assert_array_equal(c45.precursor_reachability(q,values,threshold),expected)


if __name__=='__main__':unittest.main()
