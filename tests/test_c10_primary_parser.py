import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_c10_primary_references import header_record


class PrimaryRecordParserTests(unittest.TestCase):
    def test_repeated_links_and_optional_metadata_survive_peak_parsing(self):
        fields, peaks = header_record('ACCESSION: example\nAUTHORS: A, B\nLICENSE: CC BY\nCH$LINK: INCHIKEY ABC-DEF\nCH$LINK: PUBCHEM CID:1\nPK$PEAK: m/z int. rel.int.\n  50.25 2e4 999\n  70.5 100 5\n//\n')
        self.assertEqual(fields['CH$LINK'], ['INCHIKEY ABC-DEF', 'PUBCHEM CID:1'])
        self.assertEqual(fields['LICENSE'], ['CC BY'])
        np.testing.assert_array_equal(peaks, np.array([[50.25, 20000.], [70.5, 100.]], np.float32))

    def test_malformed_peak_line_is_not_silently_skipped(self):
        with self.assertRaises(AssertionError): header_record('PK$PEAK: m/z int. rel.int.\n50.25 100\n//\n')


if __name__ == '__main__': unittest.main()
