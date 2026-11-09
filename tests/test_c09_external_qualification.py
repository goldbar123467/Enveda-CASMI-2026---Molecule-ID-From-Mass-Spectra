import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from submit_c09_external_once import MARKER, verify_review


class ExternalQualificationTests(unittest.TestCase):
    def setUp(self):
        self.review = dict(decision='qualified_sequential_trial', marker=MARKER, maximum_pending=1,
            champion_retained=.480, negative_control_interpretation=dict(detects_known_P14_regression=False,
                assessment='The control failed to detect the actual regression, so predictive validity is limited.'),
            limitations=['External-source evidence'], rationale='Supported measured-reference trial')
        self.pin = dict(policy_sha256='policy', source_files={'a': 'source'}, reference_files={'b': 'reference'})
        self.audit = dict(execution_qualified=True, changed_path_exercised=True, changed_rows=2,
            valid_rows=400, real_queries=12, official_identity_duplicates=0, exact_independent_merge=True,
            real_queries_supported_by_both_P01_models=12,
            independent_one_to_one_peak_assignment=True, local_cloud_peak_score_max_abs_difference=0.,
            fixture_real_queries=30, fixture_changed_rows=7, fixture_inserted_identities=2,
            fixture_local_cloud_peak_score_max_abs_difference=0., invalid_structure_fallback_passed=True,
            unchanged_P01_primary_error=0., source_files=self.pin['source_files'], reference_files=self.pin['reference_files'])
        summary = dict(delta=.1, ci95=[.02, .2])
        self.confirmation = dict(policy_sha256='policy', truth_absent_mrr=0, confirmation_recipe_selection=False,
            detects_known_P14_regression=False, cohorts=dict(full80=dict(queries=80,
                comparisons=dict(all_reference=copy.deepcopy(summary), input_alias_excluded=copy.deepcopy(summary))),
                no_raw_competition_signature_overlap=dict(queries=60, comparisons=dict(all_reference=copy.deepcopy(summary)))))

    def test_full_panel_gain_cannot_hide_clean_cohort_regression(self):
        verify_review(self.review, self.audit, self.confirmation, self.pin)
        self.confirmation['cohorts']['no_raw_competition_signature_overlap']['comparisons']['all_reference']['delta'] = -.1
        with self.assertRaises(AssertionError):
            verify_review(self.review, self.audit, self.confirmation, self.pin)

    def test_successful_notebook_without_changed_path_is_insufficient(self):
        self.audit.update(changed_path_exercised=False, changed_rows=0, fixture_changed_rows=0)
        with self.assertRaises(AssertionError):
            verify_review(self.review, self.audit, self.confirmation, self.pin)

    def test_protected_preview_with_real_changed_path_fixtures_is_covered(self):
        self.audit['changed_rows'] = 0
        verify_review(self.review, self.audit, self.confirmation, self.pin)

    def test_known_negative_control_cannot_be_misreported(self):
        self.review['negative_control_interpretation']['detects_known_P14_regression'] = True
        with self.assertRaises(AssertionError):
            verify_review(self.review, self.audit, self.confirmation, self.pin)


if __name__ == '__main__':
    unittest.main()
