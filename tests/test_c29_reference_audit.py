import sys
import unittest
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from c29_reference_audit import reference_view


class ReferenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('_c29_test_features',ROOT/'outputs/local/kit/features.py')
        cls.features=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.features)

    def test_scaled_identical_acquisition_excluded_but_distinct_reference_kept(self):
        mz=np.array([50.,80.,100.,140.]);it=np.array([1.,3.,8.,2.])
        refs=[(1,200.,mz,it*100,'a','C',199.),(1,200.,mz+.1,it,'b','CC',199.)]
        queries={'q':[dict(mz=mz,it=it,prec=200.)]}
        all_refs,a=reference_view(refs,queries,self.features,False)
        clean,b=reference_view(refs,queries,self.features,True)
        self.assertEqual(len(all_refs),2);self.assertEqual(a['input_alias_rows'],1)
        self.assertEqual([r[4] for r in clean],['b']);self.assertEqual(b['retained_rows'],1)

    def test_raw_signature_survives_public_precursor_peak_filter(self):
        mz=np.array([50.,80.,100.,210.]);it=np.array([1.,3.,8.,2.])
        refs=[(1,200.,mz,it,'a','C',199.)]
        queries={'q':[dict(mz=mz[:3],it=it[:3],prec=200.)]}
        raw=pd.DataFrame([dict(ms2_mzs=mz,ms2_normalized_intensities=it)])
        clean,a=reference_view(refs,queries,self.features,True,raw)
        self.assertEqual(len(clean),0);self.assertEqual(a['input_alias_rows'],1)


if __name__=='__main__':unittest.main()
