"""Regression checks for alias compatibility without shifting model tokens."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from legacy_formate_compat import patch_chem_source, patch_frag_source, scalar_namespace, fragment_offsets


# Small scalar fixture with the same source lookup boundaries as the pinned
# credited chem.py. Full-table parity is independently checked by audit_c37_formate.
CHEM = '''
MASS = dict(H=1.00782503207,C=12.,O=15.9949146196,Cl=34.96885268)
ELECTRON = 0.00054857990
PROTON = MASS['H']-ELECTRON
ADDUCTS = {'[M+H]+':(1,1,PROTON,1),'[M-H]-':(1,1,-PROTON,-1),
           '[M+CH2O2-H]-':(1,1,12.+2*MASS['O']+MASS['H']+ELECTRON,-1),
           '[M+2H]2+':(1,2,2*PROTON,1)}
TEST_ADDUCTS = list(ADDUCTS)
ADDUCT_LIST = TEST_ADDUCTS+['<unk>']
ADDUCT_IX = {a:i for i,a in enumerate(ADDUCT_LIST)}
def adduct_index(a):
    return ADDUCT_IX.get(a, ADDUCT_IX["<unk>"])
def neutral_mass(mz, adduct):
    mz = np.asarray(mz, dtype=np.float64)
    ad = np.asarray(adduct, dtype=object)
    out = np.full(len(mz),np.nan)
    for a,(n,z,d,_) in ADDUCTS.items():
        m=ad==a
        out[m]=(mz[m]*z-d)/n
    return out
def precursor_mz_from_mass(mass,adduct):
    n,z,d,_ = ADDUCTS[adduct]
    return (n*mass+d)/z
def adduct_delta(adduct):
    return ADDUCTS[adduct]
'''

FRAG = '''
def ion_deltas(adduct: str) -> List[float]:
    n, z, d, sign = chem.ADDUCTS.get(adduct, chem.ADDUCTS['[M+H]+' if adduct.endswith('+') else '[M-H]-'])
    if sign>0:return [PROTON]
    out=[-PROTON]
    if 'CH2O2' in adduct or 'HCOO' in adduct:
        out.append(chem.MASS['C']+2*H_MASS+2*chem.MASS['O']-PROTON)
    return out
'''


class LegacyFormateTests(unittest.TestCase):
    def setUp(self):
        self.old=scalar_namespace(CHEM)
        self.new=scalar_namespace(patch_chem_source(CHEM))

    def test_alias_uses_canonical_embedding_without_changing_vocabulary(self):
        self.assertEqual(self.old['ADDUCT_LIST'],self.new['ADDUCT_LIST'])
        self.assertEqual(self.old['ADDUCT_IX'],self.new['ADDUCT_IX'])
        self.assertNotEqual(self.old['adduct_index']('[M+FA-H]-'),self.old['adduct_index']('[M+CH2O2-H]-'))
        self.assertEqual(self.new['adduct_index']('[M+FA-H]-'),self.new['adduct_index']('[M+CH2O2-H]-'))

    def test_supported_mass_bits_and_multi_charge_roundtrip(self):
        adducts=np.asarray(list(self.old['ADDUCTS']),object)
        np.testing.assert_array_equal(self.old['neutral_mass']([500.]*4,adducts),self.new['neutral_mass']([500.]*4,adducts))
        for a in list(adducts)+['[M+FA-H]-']:
            mz=self.new['precursor_mz_from_mass'](431.23456,a)
            self.assertAlmostEqual(self.new['neutral_mass']([mz],[a])[0],431.23456,places=11)
            self.assertEqual(self.new['adduct_delta'](a),self.old['adduct_delta']('[M+CH2O2-H]-' if a=='[M+FA-H]-' else a))

    def test_unrecognized_tokens_remain_unknown_without_proton_default(self):
        for token in ['unknown','[M+madeup]-',None,' [M+H]+ ']:
            self.assertTrue(np.isnan(self.new['neutral_mass']([500.],[token])[0]))
            self.assertEqual(self.new['adduct_index'](token),self.old['ADDUCT_IX']['<unk>'])

    def test_fragment_formate_branch_matches_canonical(self):
        old=fragment_offsets(FRAG,self.old);new=fragment_offsets(patch_frag_source(FRAG),self.new)
        self.assertNotEqual(old('[M+FA-H]-'),old('[M+CH2O2-H]-'))
        self.assertEqual(new('[M+FA-H]-'),new('[M+CH2O2-H]-'))
        for a in self.old['ADDUCTS']:self.assertEqual(old(a),new(a))

    def test_source_drift_and_double_patch_are_rejected(self):
        with self.assertRaises(AssertionError):patch_chem_source(CHEM.replace('ADDUCTS[adduct]','ADDUCTS.get(adduct)'))
        with self.assertRaises(AssertionError):patch_chem_source(patch_chem_source(CHEM))
        with self.assertRaises(AssertionError):patch_frag_source(patch_frag_source(FRAG))


if __name__=='__main__':unittest.main()
