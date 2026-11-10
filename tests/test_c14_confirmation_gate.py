import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c14_confirmation_gate import gate


class C14ConfirmationGateTests(unittest.TestCase):
    def setUp(self):
        self.full={n:dict(n=120,delta=.01,ci95=[.001,.03]) for n in
            ['candidate_vs_P01','candidate_vs_P16','alias_excluded_vs_P01','alias_excluded_vs_P16']}
        self.modes={v+':'+m+':'+p:dict(n=60,delta=.001) for v in
            ['all_reference','input_alias_excluded'] for m in ['positive','negative'] for p in ['P01','P16']}

    def test_supported_full_and_both_polarities_pass(self):
        self.assertTrue(gate(self.full,self.modes)['passed'])

    def test_zero_main_lowerbound_rejects_but_zero_alias_is_allowed(self):
        full=copy.deepcopy(self.full);full['alias_excluded_vs_P01'].update(delta=0.,ci95=[0.,0.])
        self.assertTrue(gate(full,self.modes)['passed'])
        full['candidate_vs_P01']['ci95'][0]=0.
        self.assertFalse(gate(full,self.modes)['passed'])

    def test_overall_gain_cannot_hide_a_negative_polarity_mean(self):
        for name in self.modes:
            modes=copy.deepcopy(self.modes);modes[name]['delta']=-.00001
            self.assertIn(name,gate(self.full,modes)['failed_comparisons'])

    def test_alias_harm_rejects(self):
        full=copy.deepcopy(self.full);full['alias_excluded_vs_P16']['ci95'][0]=-.00001
        self.assertFalse(gate(full,self.modes)['passed'])

    def test_partial_or_nonfinite_inputs_fail_closed(self):
        for name,value in [('n',119),('delta',float('nan'))]:
            full=copy.deepcopy(self.full);full['candidate_vs_P01'][name]=value
            with self.assertRaises(AssertionError):gate(full,self.modes)
        modes=copy.deepcopy(self.modes);modes['all_reference:negative:P01']['n']=59
        with self.assertRaises(AssertionError):gate(self.full,modes)


if __name__=='__main__':unittest.main()
