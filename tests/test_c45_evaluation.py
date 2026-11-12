"""Protect exact directional gates and prospective parent selection."""
import sys,unittest
from pathlib import Path
from fractions import Fraction
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_c45_complete import point,select_variants,sources_admit,ranks
import pandas as pd


def value(delta):
    x=Fraction(delta)
    return dict(numerator=x.numerator,denominator=x.denominator,strictly_positive=x>0,nonnegative=x>=0)


def choices(cpu01,cpu16,gpu01,gpu16):
    return {name:dict(complete=value(full),incremental=value(increment)) for name,(full,increment) in zip(
        ['cpu_P01','cpu_P16','gpu_P01','gpu_P16'],[cpu01,cpu16,gpu01,gpu16])}


class EvaluationTests(unittest.TestCase):
    def test_absent_truths_are_zero_and_remain_in_denominator(self):
        p=point(np.array([0,4,0]),np.array([0,2,0]),np.ones(3,bool))
        self.assertEqual(Fraction(p['numerator'],p['denominator']),Fraction(1,12))
        self.assertEqual(p['n'],3)

    def test_exact_tie_prefers_p01_for_both_heads(self):
        self.assertEqual(select_variants(choices((1,1),(1,1),(1,1),(1,1))),dict(cpu='P01',gpu='P01'))

    def test_parent_gain_cannot_qualify_harmful_incremental_head(self):
        self.assertEqual(select_variants(choices((1,1),(3,-1),(0,0),(2,0))),dict(cpu='P01',gpu=None))

    def test_choose_only_development_best_eligible_parent(self):
        self.assertEqual(select_variants(choices((1,1),(2,1),(4,4),(2,1))),dict(cpu='P16',gpu='P01'))

    def test_no_source_rounding_or_overall_gain_can_hide_negative_slice(self):
        report=dict(development=dict(point=value(1)),source={'a':dict(point=value(1))},
                    query_mode={'negative':dict(point=value(Fraction(-1,10**30)))})
        self.assertFalse(sources_admit(report))
        report['query_mode']['negative']['point']=value(0)
        self.assertTrue(sources_admit(report))
        report['development']['point']=value(0)
        self.assertFalse(sources_admit(report))

    def test_development_phase_does_not_match_source_answers(self):
        labels=pd.DataFrame(dict(key=['a','b','c']))
        keys={'P01':[['x','a'],['b'],['c']]}
        out=ranks(keys,labels,[0],['P01'])['P01']
        np.testing.assert_array_equal(out,[2,0,0])


if __name__=='__main__':unittest.main()
