"""Describe source sensitivity of sealed C09 results; never select or submit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tail_fusion import paired_summary

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'outputs/local/c09-followup-evaluation-20261109'
EXTERNAL = ROOT / 'outputs/local/c09-external-reference-20261109'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned(path, index):
    frame = pd.read_csv(path, dtype={'molecule_id': str}).set_index('molecule_id')
    assert frame.index.is_unique and set(frame.index) == set(index), path
    return frame.loc[index]


def summarize_sources(labels, values, raw_overlap):
    """Keep absent truths in every denominator, including each source/cohort."""
    baseline = np.asarray(values['P01'], dtype=float)
    for name, vector in values.items():
        vector = np.asarray(vector, dtype=float)
        assert vector.shape == baseline.shape == (len(labels),), name
        assert np.isfinite(vector).all() and ((vector >= 0) & (vector <= 1)).all(), name
    assert not labels.source.isna().any()
    result = {}
    for cohort, mask in [('full80', np.ones(len(labels), dtype=bool)),
                         ('no_raw_competition_signature_overlap', ~raw_overlap)]:
        groups = {'all_sources': mask}
        groups.update({source: mask & labels.source.eq(source).to_numpy()
                       for source in sorted(labels.source.unique())})
        result[cohort] = {}
        for source, selected in groups.items():
            if not selected.any():
                result[cohort][source] = dict(queries=0, comparisons=None)
                continue
            result[cohort][source] = dict(
                queries=int(selected.sum()),
                baseline_truths=int((baseline[selected] > 0).sum()),
                candidate_truths={name: int((np.asarray(vector)[selected] > 0).sum())
                                  for name, vector in values.items() if name != 'P01'},
                comparisons={name: paired_summary(baseline[selected], np.asarray(vector)[selected])
                             for name, vector in values.items() if name != 'P01'})
    return result


def main():
    paths = {
        'labels': ROOT / 'outputs/publish/c09-validation-20261109/labels.csv',
        'raw_spectrum_matches': ROOT / 'outputs/forks/c09-spectrum-audit-v3/c09_spectrum_matches.csv',
        'original_report': ORIGINAL / 'confirmation.json',
        'original_predictions': ORIGINAL / 'confirmation-predictions.csv',
        'external_report': EXTERNAL / 'confirmation.json',
        'external_predictions': EXTERNAL / 'confirmation-predictions.csv',
    }
    # Require both completed reports. Never infer quality from partial outputs.
    original_report = json.loads(paths['original_report'].read_text(encoding='utf-8'))
    external_report = json.loads(paths['external_report'].read_text(encoding='utf-8'))
    assert not original_report['reserved_recipe_reselection']
    assert not external_report['confirmation_recipe_selection']
    assert original_report['truth_absent_mrr'] == external_report['truth_absent_mrr'] == 0
    labels = pd.read_csv(paths['labels'], dtype=str).set_index('molecule_id')
    assert len(labels) == 80 and labels.index.is_unique and labels.partition.eq('confirmation').all()
    original = aligned(paths['original_predictions'], labels.index)
    external = aligned(paths['external_predictions'], labels.index)
    assert np.allclose(original.P01, external.P01, rtol=0, atol=1e-14)
    assert np.allclose(original.P14, external.P14, rtol=0, atol=1e-14)
    overlaps = set(pd.read_csv(paths['raw_spectrum_matches'], dtype={'molecule_id': str}).molecule_id)
    raw_overlap = labels.index.isin(overlaps)
    assert np.array_equal(original.raw_overlap.astype(str).str.lower().to_numpy(),
                          np.where(raw_overlap, 'true', 'false'))
    values = {'P01': original.P01.to_numpy()}
    values.update({'C09::' + name: original[name].to_numpy()
                   for name in original if name not in ['P01', 'raw_overlap']})
    values.update({'C09E::' + name: external[name].to_numpy()
                   for name in ['all_reference', 'input_alias_excluded']})
    cohorts = summarize_sources(labels, values, raw_overlap)
    # Independent aggregation must agree with the sealed primary reports.
    for cohort in cohorts:
        all_sources = cohorts[cohort]['all_sources']
        sealed = external_report['cohorts'][cohort]
        assert all_sources['queries'] == sealed['queries']
        for view in ['all_reference', 'input_alias_excluded']:
            measured = all_sources['comparisons']['C09E::' + view]
            expected = sealed['comparisons'][view]
            for field in ['baseline_mrr25', 'candidate_mrr25', 'delta']:
                assert np.isclose(measured[field], expected[field], rtol=0, atol=1e-14)
    report = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        cohorts=cohorts,
        input_sha256={name: sha(path) for name, path in paths.items()},
        reporter_sha256=sha(__file__),
        recipe_selection=False, automatic_submission=False, truth_absent_mrr=0,
        limitation='Descriptive source sensitivity of the fixed candidate, not an estimate of hidden source mix or universal pretraining/acquisition exclusion. No per-source policy is selected.')
    target = EXTERNAL / 'confirmation-source-sensitivity.json'
    assert not target.exists(), 'Existing source report is sealed'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('DONE C09 SOURCE SENSITIVITY', str(target), flush=True)


if __name__ == '__main__':
    main()
