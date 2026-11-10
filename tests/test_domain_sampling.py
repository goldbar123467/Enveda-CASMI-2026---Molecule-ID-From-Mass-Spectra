import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from domain_sampling import balanced_ids


class DomainSamplingTests(unittest.TestCase):
    def test_exact_strata_unique_and_no_outside_identity(self):
        members=np.arange(10);others=np.arange(20,50)
        out=balanced_ids(members,others,16,np.random.default_rng(11))
        self.assertEqual(len(set(out)),16);self.assertEqual(int(np.isin(out,members).sum()),8)
        self.assertTrue(set(out)<=set(members)|set(others))

    def test_restored_rng_repeats_exact_and_global_rng_is_untouched(self):
        rng=np.random.default_rng(23);before=np.random.get_state()
        balanced_ids(np.arange(20),np.arange(30,80),16,rng);state=copy.deepcopy(rng.bit_generator.state)
        expected=balanced_ids(np.arange(20),np.arange(30,80),16,rng)
        resumed=np.random.default_rng();resumed.bit_generator.state=state
        np.testing.assert_array_equal(expected,balanced_ids(np.arange(20),np.arange(30,80),16,resumed))
        after=np.random.get_state();self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_insufficient_or_overlapping_strata_fail_closed(self):
        with self.assertRaises(AssertionError):balanced_ids(np.arange(2),np.arange(10,20),8,np.random.default_rng(1))
        with self.assertRaises(AssertionError):balanced_ids(np.arange(4),np.arange(4),8,np.random.default_rng(1))


if __name__=='__main__':unittest.main()
