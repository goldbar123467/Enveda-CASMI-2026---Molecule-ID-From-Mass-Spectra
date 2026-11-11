"""Guard against dropped queries, chemistry leakage and false promotion."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validation_repair import (evaluate_orders,formula_order,folds_for,crossfit,
    clustered_summary,relative,validation_gate)
from validation_admission import check_validation_report
from review_c36_validation import neutral_mass


class ValidationRepairTests(unittest.TestCase):
    def block(self,i,truth='B'):
        return dict(mid=str(i),group=str(i//2),mode='positive' if i%2 else 'negative',
            keys=['A','B','C'],truth=truth,formulas=['F','F','G'],library=0.,
            x=np.asarray([[0.,1.],[1.,0.],[.3,.5]]))

    def config(self):
        return dict(outer_folds=4,inner_folds=2,split_salt='synthetic',ridge_l2=.1,
            alpha_candidates=[0.,.25,.5,1.],protect_library_at=.9,
            minimum_primary_delta=.003,minimum_primary_ci_lower=0.)

    def test_missing_truth_is_zero_and_missing_query_is_error(self):
        blocks=[self.block(0),self.block(1,'absent')]
        np.testing.assert_allclose(evaluate_orders(blocks,[[0,1,2],[0,1,2]]),[.5,0.])
        with self.assertRaises(AssertionError):evaluate_orders(blocks,[[0,1,2]])

    def test_equivalent_identity_cannot_take_two_slots(self):
        b=self.block(0);b['keys']=['A','A','C']
        with self.assertRaisesRegex(AssertionError,'Duplicate official identity'):
            evaluate_orders([b],[[0,1,2]])

    def test_isomer_promotion_preserves_formula_slots_and_membership(self):
        b=dict(keys=['a','b','c','d'],formulas=['F1','F2','F1','F2'],library=0.)
        order=formula_order(b,np.asarray([0.,0.,1.,1.]),1.)
        self.assertEqual(order,[2,3,0,1])
        self.assertEqual([b['formulas'][i] for i in order],b['formulas'])
        self.assertEqual(formula_order(b,np.ones(4),0.),[0,1,2,3])
        b['library']=.95
        self.assertEqual(formula_order(b,np.asarray([0.,0.,1.,1.]),1.),[0,1,2,3])

    def test_missing_and_constant_channels_do_not_invent_a_rank(self):
        present,z,ranks=relative([np.nan,0.,0.])
        np.testing.assert_equal(present,[0,1,1]);np.testing.assert_equal(z,[0,0,0]);np.testing.assert_equal(ranks,[0,0,0])

    def test_chemistry_components_stay_in_one_fold(self):
        blocks=[self.block(i) for i in range(16)]
        folds=folds_for(blocks,list(range(16)),4,'synthetic')
        for i in range(0,16,2):self.assertEqual(folds[i],folds[i+1])

    def test_heldout_targets_do_not_change_their_predictions_or_alpha(self):
        blocks=[self.block(i) for i in range(16)];cfg=self.config()
        folds=folds_for(blocks,list(range(16)),4,cfg['split_salt'])
        orders,audits=crossfit(blocks,[0,1],cfg,folds)
        changed=copy.deepcopy(blocks);held=[i for i in folds if folds[i]==0]
        for i in held:changed[i]['truth']='A'
        other,other_audits=crossfit(changed,[0,1],cfg,folds)
        self.assertEqual([orders[i] for i in held],[other[i] for i in held])
        self.assertEqual(audits[0]['alpha'],other_audits[0]['alpha'])
        np.testing.assert_array_equal(audits[0]['beta'],other_audits[0]['beta'])
        for a in audits:
            self.assertFalse(set(a['train_ids'])&set(a['test_ids']))
            self.assertFalse(set(a['train_groups'])&set(a['test_groups']))
            for inner in a['selection']['audits']:
                self.assertFalse(set(inner['train_groups'])&set(inner['test_groups']))

    def test_grouped_bootstrap_keeps_query_weighted_point_estimate(self):
        report=clustered_summary(np.zeros(4),np.asarray([1.,0.,0.,0.]),['a','b','b','b'],1000,2)
        self.assertEqual(report['delta'],.25)
        self.assertEqual(report['independent_groups'],2)
        self.assertEqual(report['ci95'],[0.,1.])

    def test_wrong_sign_public_control_blocks_an_optimistic_candidate(self):
        cfg=self.config();primary=dict(delta=.02,ci95=[.01,.03]);polarity={'positive':dict(delta=.02)}
        report=validation_gate(primary,polarity,{'P14':dict(delta=.005)},cfg,True)
        self.assertFalse(report['passed']);self.assertFalse(report['leaderboard_calibrated'])
        self.assertIn('known_harmful_P14_has_wrong_local_sign',report['reasons'])
        report=validation_gate(primary,polarity,{'P14':dict(delta=-.005)},cfg,False)
        self.assertFalse(report['passed']);self.assertIn('no_untouched_independent_confirmation',report['reasons'])

    def test_clean_source_harm_cannot_be_hidden_by_an_aggregate_gain(self):
        cfg=self.config();cfg['primary_queries']=240
        primary=dict(all=dict(n=240,delta=.03,ci95=[.01,.05]),polarity={'positive':dict(delta=.03),'negative':dict(delta=.01)})
        stress=dict(all=dict(n=80,delta=.01,ci95=[.001,.03]),raw_alias_free=dict(n=60,delta=-.02),polarity={'positive':dict(delta=.04),'negative':dict(delta=-.03)})
        r=dict(primary_variant='primary',results={'primary':primary},source_stress={'primary':stress},controls={'P14':dict(delta=-.02)},
               all_queries_retained=True,missing_truths_score_zero=True,truth_injection=False,no_outer_or_stress_parameter_selection=True,no_test_queries_in_fit=True)
        admission=check_validation_report(r,cfg,True,True)
        self.assertFalse(admission['passed'])
        self.assertIn('raw_alias_free_source_regression',admission['reasons'])
        self.assertIn('source_stress_polarity_regression',admission['reasons'])

    def test_formate_mass_diagnostic_does_not_default_unknown_adducts(self):
        self.assertAlmostEqual(neutral_mass(244.998201,'[M+FA-H]-','negative'),200.)
        self.assertAlmostEqual(neutral_mass(101.007276,'[M+2H]2+','positive'),200.)
        self.assertIsNone(neutral_mass(201.,'unknown','positive'))
        self.assertIsNone(neutral_mass(244.998201,'[M+FA-H]-','positive'))


if __name__=='__main__':unittest.main()
