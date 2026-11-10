import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from casmi2022_data import parse_msp, scan_rank


class CasmiAssignmentTests(unittest.TestCase):
    def test_unmatched_target_never_receives_the_first_scan(self):
        scan = dict(name='wrong', precursor_mz=200., rt=1., spectrum_type='MS2', peaks=[(50., 1000.)])
        self.assertIsNone(scan_rank(scan, 300., 1.))
        self.assertIsNone(scan_rank(scan, 200., 2.))

    def test_assignment_uses_only_matching_ms2_and_total_intensity(self):
        low = dict(name='low', precursor_mz=300.001, rt=1.02, spectrum_type='MS2', peaks=[(50., 2.), (80., 3.)])
        high = dict(low, name='high', peaks=[(50., 20.), (80., 30.)])
        self.assertLess(scan_rank(high, 300., 1.), scan_rank(low, 300., 1.))
        self.assertIsNone(scan_rank(dict(high, spectrum_type='MS1'), 300., 1.))

    def test_peak_counts_are_checked_even_at_eof(self):
        text = 'Name: scan\nSpectrum_type: MS2\nPRECURSORMZ: 300\nRETENTIONTIME: 1\nNum peaks: 2\n50 3\n80 4\n'
        self.assertEqual(len(list(parse_msp(io.StringIO(text)))), 1)
        with self.assertRaises(ValueError):
            list(parse_msp(io.StringIO(text.rsplit('80 4', 1)[0])))


if __name__ == '__main__':
    unittest.main()
