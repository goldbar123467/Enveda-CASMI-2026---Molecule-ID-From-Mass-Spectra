import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adduct_aliases import normalize_pv_adduct


class AdductAliasesTests(unittest.TestCase):
    def test_formate_and_boundary_whitespace(self):
        for text in ['[M+FA-H]-','[M+FA-H]- ',' [M+CH2O2-H]-']:
            self.assertEqual(normalize_pv_adduct(text),'[M+CH2O2-H]-')

    def test_other_recognized_names_remain_exact(self):
        for text in ['[M+H]+','[M+NH4]+','[M+Na]+','[M-H]-','[M+Cl]-','[M+2H]2+','[M+CH2O2-H]-']:
            self.assertEqual(normalize_pv_adduct(text),text)

    def test_unknown_has_no_proton_default_and_normalization_is_idempotent(self):
        for text in ['unknown','[M+weird]-','',None]:
            result=normalize_pv_adduct(text)
            self.assertNotEqual(result,'[M+H]+');self.assertEqual(normalize_pv_adduct(result),result)


if __name__=='__main__':unittest.main()
