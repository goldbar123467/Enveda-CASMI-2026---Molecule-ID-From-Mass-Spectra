"""Keep mixed acquisition queries and zero-support strata visible."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_c37_population import query_table, support


class PopulationAuditTests(unittest.TestCase):
    def source(self):
        return pd.DataFrame([
            dict(molecule_id='A',ionization_mode='positive',adduct='[M+H]+',instrument_type='FT',precursor_mz=200.,ms2_mzs=[50.,70.]),
            dict(molecule_id='A',ionization_mode='negative',adduct='[M-H]-',instrument_type='FT',precursor_mz=198.,ms2_mzs=[40.]),
            dict(molecule_id='B',ionization_mode='negative',adduct='[M+FA-H]-',instrument_type='FT',precursor_mz=300.,ms2_mzs=[70.]),
        ])

    def test_mixed_acquisitions_stay_one_query(self):
        f=query_table(self.source()).set_index('molecule_id')
        self.assertEqual(len(f),2)
        self.assertEqual(f.loc['A','mode'],'mixed')
        self.assertEqual(f.loc['A','spectra'],2)
        self.assertEqual(f.loc['A','adduct_count'],2)
        self.assertEqual(f.loc['B','dominant_adduct'],'[M+CH2O2-H]-')

    def test_unsupported_target_stratum_is_not_dropped_or_given_infinite_weight(self):
        target=query_table(self.source())
        panel=target[target.molecule_id.eq('B')].copy()
        result=support(target,panel)
        self.assertEqual(result['target_mode_adduct_support_fraction'],.5)
        missing=[r for r in result['strata'] if r['panel_queries']==0]
        self.assertEqual(len(missing),1)
        self.assertIsNone(missing[0]['importance_weight'])
        self.assertEqual(result['importance_effective_queries'],1.)
        panel['instruments']='timsTOF'
        self.assertEqual(support(target,panel)['exact_target_instrument_query_support'],0.)

    def test_label_columns_are_rejected(self):
        f=self.source();f['truth']='hidden'
        with self.assertRaisesRegex(AssertionError,'Query metadata only'):query_table(f)


if __name__=='__main__':unittest.main()
