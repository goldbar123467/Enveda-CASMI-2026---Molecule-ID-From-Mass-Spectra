import sys
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from public_deployment_support import intake,protect,finalize


class DeploymentSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('_public_intake_features',ROOT/'outputs/local/kit/features.py')
        cls.features=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.features)

    def query(self,mid,**extra):
        d=dict(molecule_id=mid,ionization_mode='negative',adduct='[M+FA-H]-',precursor_mz=300.,
            ms2_mzs=np.array([50.,100.,200.]),ms2_normalized_intensities=np.array([1.,.5,.2]),collision_energy_ev=np.nan)
        return dict(d,**extra)

    def test_explicit_fallbacks_and_exact_parent_strings(self):
        ids=['valid','absent','unknown','mixed','short','adduct','precursor','malformed']
        parent=pd.DataFrame(dict(molecule_id=ids,smiles=['C;CC']*len(ids)))
        q=pd.DataFrame([self.query('valid'),self.query('unknown',ionization_mode='?'),
            self.query('mixed',adduct='[M-H]-',precursor_mz=298.992724),self.query('mixed',ionization_mode='positive',adduct='[M+H]+',precursor_mz=301.007276),
            self.query('short',ms2_mzs=[50.,100.],ms2_normalized_intensities=[1.,.5]),
            self.query('adduct',adduct='unknown'),self.query('precursor',precursor_mz=np.nan),
            self.query('malformed',ms2_normalized_intensities=[1.])])
        eligible,audit=intake(parent,q,self.features,[])
        self.assertEqual(eligible,{'valid','mixed'})
        reasons=dict(zip(audit.molecule_id,audit.reason))
        self.assertEqual(reasons['mixed'],'supported');self.assertEqual(reasons['unknown'],'unknown_mode')
        self.assertEqual(audit.set_index('molecule_id').loc['mixed','mode'],'mixed')
        self.assertEqual(reasons['absent'],'absent');self.assertEqual(reasons['adduct'],'unknown_adduct')
        candidate=parent.copy();candidate['smiles']='CC;C'
        final=protect(parent,candidate,eligible)
        self.assertEqual(final.smiles.tolist(),['CC;C','C;CC','C;CC','CC;C','C;CC','C;CC','C;CC','C;CC'])

    def test_recognized_engine_formate_alias_keeps_original_neural_encoding(self):
        parent=pd.DataFrame(dict(molecule_id=['a'],smiles=['C;CC']))
        q=pd.DataFrame([self.query('a',adduct='[M+CH2O2-H]-')])
        self.assertEqual(self.features.metadata(300.,'[M+CH2O2-H]-','negative',np.nan)[1],0)
        eligible,_=intake(parent,q,self.features,['[M+CH2O2-H]-'])
        self.assertEqual(eligible,{'a'})

    def test_boundary_whitespace_is_recognized_without_changing_neural_token(self):
        parent=pd.DataFrame(dict(molecule_id=['a'],smiles=['C;CC']))
        q=pd.DataFrame([self.query('a',adduct='[M+FA-H]- ')])
        self.assertEqual(self.features.metadata(300.,q.adduct.iloc[0],'negative',np.nan)[1],0)
        eligible,_=intake(parent,q,self.features,[])
        self.assertEqual(eligible,{'a'})
        self.assertEqual(q.adduct.iloc[0],'[M+FA-H]- ')

    def test_one_short_acquisition_does_not_drop_a_supported_identity(self):
        parent=pd.DataFrame(dict(molecule_id=['a'],smiles=['C;CC']))
        q=pd.DataFrame([self.query('a'),self.query('a',ms2_mzs=[50.],ms2_normalized_intensities=[1.])])
        eligible,audit=intake(parent,q,self.features,[])
        self.assertEqual(eligible,{'a'});self.assertEqual(audit.usable_spectrum_rows.iloc[0],1)
        self.assertEqual(audit.spectrum_rows.iloc[0],2)

    def test_identity_dedup_backfills_from_prior_in_order(self):
        prior=pd.DataFrame(dict(molecule_id=['a'],smiles=['C;CC;CO']))
        raw=pd.DataFrame(dict(molecule_id=['a'],smiles=['[CH4];C;CC']))
        out,_=finalize(raw,prior,lambda s:self.features.structure(s)[0],prior)
        self.assertEqual(out.smiles.iloc[0],'[CH4];CC;CO')

    def test_invalid_reference_identity_preserves_exact_parent(self):
        prior=pd.DataFrame(dict(molecule_id=['a'],smiles=['C;CC;CO']))
        parent=pd.DataFrame(dict(molecule_id=['a'],smiles=['CO;CC;C']))
        raw=pd.DataFrame(dict(molecule_id=['a'],smiles=['INVALID;C;CC']))
        out,report=finalize(raw,prior,lambda s:self.features.structure(s)[0],parent)
        self.assertTrue(out.equals(parent));self.assertEqual(report['invalid_structure_parent_fallback_ids'],['a'])


if __name__=='__main__':unittest.main()
