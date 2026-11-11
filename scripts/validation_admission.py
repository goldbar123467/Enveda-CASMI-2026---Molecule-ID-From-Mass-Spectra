"""Require independent-source support before a validation gain can be promoted."""


def check_validation_report(report,config,independent_confirmation=False,pretraining_audited=False):
    primary=report['results'][report['primary_variant']];stress=report['source_stress'][report['primary_variant']]
    reasons=[]
    if not (report['all_queries_retained'] and report['missing_truths_score_zero'] and not report['truth_injection']):
        reasons.append('incomplete_or_truth_forced_evaluation')
    if not (report['no_outer_or_stress_parameter_selection'] and report['no_test_queries_in_fit']):
        reasons.append('heldout_selection_or_fit_leakage')
    if primary['all']['n']!=config['primary_queries']:reasons.append('wrong_primary_query_count')
    if primary['all']['delta']<config['minimum_primary_delta'] or primary['all']['ci95'][0]<=0:
        reasons.append('primary_gain_not_material_and_supported')
    if any(r['delta']<0 for r in primary['polarity'].values()):reasons.append('primary_polarity_regression')
    if stress['all']['n']!=80 or stress['raw_alias_free']['n']!=60:reasons.append('source_stress_query_coverage_changed')
    if stress['all']['ci95'][0]<=0:reasons.append('source_stress_gain_not_supported')
    if stress['raw_alias_free']['delta']<0:reasons.append('raw_alias_free_source_regression')
    if any(r['delta']<0 for r in stress['polarity'].values()):reasons.append('source_stress_polarity_regression')
    if report['controls']['P14']['delta']>=0:reasons.append('proxy_does_not_detect_known_P14_regression')
    if not independent_confirmation:reasons.append('untouched_confirmation_not_consumed')
    if not pretraining_audited:reasons.append('external_model_pretraining_exposure_unresolved')
    return dict(passed=not reasons,reasons=reasons,automatic_gpu_work=False,automatic_submission=False,
                interpretation='An exploratory development gain alone cannot authorize deployment or further compute.')
