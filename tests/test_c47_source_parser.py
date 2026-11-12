"""Malformed and ambiguous source records cannot silently become validation."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inventory_c47_public_plasticizers import parse_mgf,one


class SourceParserTests(unittest.TestCase):
    def test_export_typo_is_recorded_not_silently_accepted(self):
        rows=parse_mgf('BEGIN IONS\nTITLE=x\n10 1\n20 2\n30 3\nENDS IONS\n')
        self.assertEqual(len(rows),1);self.assertIn('_source_missing_END_IONS',rows[0][1])
    def test_only_finite_positive_peaks_retained(self):
        row=parse_mgf('BEGIN IONS\nTITLE=x\n10 1 "note"\n20 0\n30 -2\nnan 1\n40 inf\nEND IONS\n')[0]
        self.assertEqual(row[2],[(10.,1.)]);self.assertEqual(one(row[1],'TITLE'),'x')
    def test_repeated_metadata_requires_exact_agreement(self):
        self.assertEqual(one({'Precursor_type':['[M+H]+','[M+H]+']},'Precursor_type'),'[M+H]+')
        self.assertIsNone(one({'Precursor_type':['[M+H]+','[M+Na]+']},'Precursor_type'))
        self.assertIsNone(one({},'Precursor_type'))


if __name__=='__main__':unittest.main()
