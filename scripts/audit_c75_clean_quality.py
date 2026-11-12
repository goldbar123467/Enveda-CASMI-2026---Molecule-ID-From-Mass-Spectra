"""Independent ranks, exact fractions, denominators and gate audit for C72."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name, '2')
import torch
torch.set_num_threads(2)
import numpy as np
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp
from complete_c75_local_target import OUT, PAIR, PANEL, native
from evaluate_c75_clean_target import paths, FREEZE, RESULT
from audit_c45_quality import check_point


def main():
    dest = OUT / 'clean-independent-quality-audit.json'; assert not dest.exists()
    result = read(RESULT); frozen = read(FREEZE)
    assert result['source_sha256'] == frozen['source_sha256'] == sha(ROOT / 'scripts/evaluate_c75_clean_target.py')
    assert result['evaluation_freeze_sha256'] == sha(FREEZE)
    assert result['consumption_sha256'] == sha(PANEL / 'evaluation-consumption.json')
    assert sha(PANEL / 'labels.csv') == frozen['labels_sha256']
    for name, pin in frozen['files_sha256'].items(): assert sha(native(name)) == pin
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    frames = {name: pd.read_csv(path, dtype=str, keep_default_na=False) for name, path in paths().items()}
    mids = frames['P01'].molecule_id.tolist(); assert len(mids) == len(set(mids)) == 512
    labels = pd.read_csv(PANEL / 'labels.csv', dtype=str).set_index('molecule_id').loc[mids]
    assert labels.source.value_counts().to_dict() == {'enveda-180': 512} and labels.group.nunique() == 471
    identities = pd.read_parquet(OUT / 'clean-quality-identities.parquet').set_index('smiles').key.to_dict()
    ranks = {}; lists = {}; stored = np.load(OUT / 'clean-source-ranks.npz')
    for name, frame in frames.items():
        assert frame.molecule_id.tolist() == mids
        lists[name] = [[identities[s] for s in text.split(';')] for text in frame.smiles]
        assert all(1 <= len(row) <= 25 and len(set(row)) == len(row) for row in lists[name])
        ranks[name] = np.asarray([next((i for i,k in enumerate(row, 1) if k==key), 0)
                                  for row,key in zip(lists[name], labels.key)], dtype=np.int16)
        np.testing.assert_array_equal(ranks[name], stored[name])
    q = pd.read_parquet(OUT / 'queries.parquet'); assert len(q) == 2215
    modes = q.groupby('molecule_id').ionization_mode.agg(lambda a: 'mixed' if len(set(a))>1 else a.iloc[0]).loc[mids].to_numpy()
    assert pd.Series(modes).value_counts().to_dict() == result['actual_query_modes'] == {'positive': 353, 'mixed': 124, 'negative': 35}
    assert (modes == labels.query_mode.to_numpy()).all()
    whole = np.ones(512, bool)
    def check(value, base, trial):
        sign = check_point(value['all']['point'], ranks[base], ranks[trial], whole)
        assert set(value['actual_query_mode']) == set(modes)
        mode_signs = [check_point(entry['point'], ranks[base], ranks[trial], modes==mode)
                      for mode,entry in value['actual_query_mode'].items()]
        assert value['recovered_truths'] == int(((ranks[base]==0)&(ranks[trial]>0)).sum())
        assert value['displaced_truths'] == int(((ranks[base]>0)&(ranks[trial]==0)).sum())
        assert value['truth_absent_queries'] == int((ranks[trial]==0).sum())
        assert value['rank1_changes'] == sum(a[0]!=b[0] for a,b in zip(lists[base], lists[trial]))
        assert value['candidate_membership_changes'] == sum(set(a)!=set(b) for a,b in zip(lists[base], lists[trial]))
        before = np.divide(1., ranks[base], out=np.zeros(512), where=ranks[base]>0)
        after = np.divide(1., ranks[trial], out=np.zeros(512), where=ranks[trial]>0)
        assert value['all']['improved'] == int((after>before).sum())
        assert value['all']['harmed'] == int((after<before).sum())
        return sign > 0 and all(v >= 0 for v in mode_signs)
    admissions = {}
    for trial in ['cpu', 'gpu']:
        entry = result['results'][trial]
        assert set(entry['comparisons']) == {'P01', 'P16', 'physical_parent'}
        admissions[trial] = all([check(value, base, trial) for base,value in entry['comparisons'].items()])
        assert admissions[trial] == entry['admission']['passed']
        assert all(set(a)==set(b) for a,b in zip(lists['physical_parent'], lists[trial]))
    check(result['P16_overlay'], 'P01', 'P16')
    check(result['physical_adapter'], 'P16', 'physical_parent')
    check(result['P14_negative_control'], 'P01', 'P14')
    assert result['passed_pair'] == all(admissions.values())
    assert result['absent_truths_zero'] and result['all_queries_retained']
    assert not result['qualified_for_scoring'] and result['scoring_entries'] == 0
    write(dest, dict(completed_utc=stamp(), source_sha256=sha(__file__),
        quality_results_sha256=sha(RESULT), evaluation_freeze_sha256=sha(FREEZE),
        complete_pipeline_binding_sha256=sha(PAIR / 'complete-pipeline-binding.json'),
        queries=512, acquisitions=2215, chemistry_groups=471, exact_official_rank_vectors=True,
        exact_fraction_points=True, all_original_mode_denominators_exact=True,
        recovered_displaced_counts_exact=True, absent_truths_zero=True,
        admissions=admissions, passed_pair=all(admissions.values()),
        C13_read=False, scoring_entries=0, production_and_once_confirmation_still_required=True))
    print('C75 INDEPENDENT WHOLE512 QUALITY AUDIT', admissions, flush=True)


if __name__ == '__main__': main()
