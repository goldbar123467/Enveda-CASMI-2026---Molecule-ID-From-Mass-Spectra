"""Independent P16 peak assignment, merge, identity and local/cloud parity audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import bisect
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from precision_cache import ROOT, sha, write


def independent_cosine(qm, qi, rm, ri, precursor):
    qm, qi, rm, ri = [np.asarray(v, dtype=np.float64) for v in [qm, qi, rm, ri]]
    qkeep = np.isfinite(qm) & np.isfinite(qi) & (qm > 0) & (qi > 0) & (qm <= precursor + .5)
    rkeep = np.isfinite(rm) & np.isfinite(ri) & (rm > 0) & (ri > 0)
    qm, qi, rm, ri = qm[qkeep], qi[qkeep], rm[rkeep], ri[rkeep]
    qo, ro = np.argsort(qm, kind='stable'), np.argsort(rm, kind='stable')
    qm, qi, rm, ri = qm[qo], np.sqrt(qi[qo]), rm[ro], np.sqrt(ri[ro])
    qi /= max(float(np.linalg.norm(qi)), 1e-12); ri /= max(float(np.linalg.norm(ri)), 1e-12)
    pairs = []
    for i, mass in enumerate(qm):
        for j in range(bisect.bisect_left(rm, mass - .015), bisect.bisect_right(rm, mass + .015)):
            pairs.append((float(qi[i] * ri[j]), i, j))
    used_q, used_r, total, count = set(), set(), 0., 0
    for value, i, j in sorted(pairs, reverse=True):
        if i not in used_q and j not in used_r:
            total += value; count += 1; used_q.add(i); used_r.add(j)
    return min(total, 1.), count


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('folder', type=Path); args = parser.parse_args()
    folder = args.folder; assert rdBase.rdkitVersion == '2026.03.3'
    pin = json.loads((ROOT / 'kaggle/forks/p16-external-reference-c09/external-pin.json').read_text())
    assert sha(ROOT / 'kaggle/forks/p16-external-reference-c09/notebook.ipynb') == pin['notebook_sha256']
    receipt = json.loads((folder / 'p16_execution_receipt.json').read_text())
    assert receipt['required_stage_completed'] and receipt['merge_status'] == 'applied' and not receipt['is_hidden_rerun']
    assert receipt['valid_rows'] == 400 and receipt['search']['queries'] == 12
    assert receipt['official_rdkit'] == '2026.03.3' and receipt['official_identity_duplicates'] == 0
    assert receipt['policy_sha256'] == pin['policy_sha256'] and receipt['reference_files'] == pin['reference_files']
    assert receipt['source_files'] == pin['source_files'] and receipt['invalid_external_structures'] == 0
    for name, expected in pin['source_files'].items():
        assert sha(folder / 'c09e_code' / name) == expected, name
    assert sha(folder / 'c09e_code/c09e_policy.json') == pin['policy_sha256']
    reference = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    for name, expected in pin['reference_files'].items():
        assert sha(reference / name) == expected
    baseline = pd.read_csv(folder / 'c09e_p01_submission.csv', dtype=str, keep_default_na=False)
    final = pd.read_csv(folder / 'submission.csv', dtype=str, keep_default_na=False)
    assert final.columns.tolist() == baseline.columns.tolist() == ['molecule_id', 'smiles']
    assert len(final) == 400 and final.molecule_id.is_unique and final.molecule_id.tolist() == baseline.molecule_id.tolist()
    assert sha(folder / 'submission.csv') == receipt['csv_sha256']
    assert sha(folder / 'c09e_p01_submission.csv') == receipt['baseline_sha256']
    assert sha(folder / 'effective_query_spectra.parquet') == receipt['query_sha256']
    queries = pd.read_parquet(folder / 'effective_query_spectra.parquet'); assert queries.molecule_id.nunique() == 12
    meta = pd.read_parquet(reference / 'ext_meta.parquet'); mz = np.load(reference / 'ext_mz.npy', mmap_mode='r'); intensity = np.load(reference / 'ext_in.npy', mmap_mode='r')
    precursor = meta.precursor_mz.to_numpy(dtype=float); order = np.argsort(precursor, kind='stable'); sorted_precursor = precursor[order]
    best = {str(mid): (0., 0, '', None) for mid in queries.molecule_id.unique()}
    for query in queries.itertuples(index=False):
        width = query.precursor_mz * 10e-6 + .002
        lo, hi = bisect.bisect_left(sorted_precursor, query.precursor_mz - width), bisect.bisect_right(sorted_precursor, query.precursor_mz + width)
        for pos in order[lo:hi]:
            ref = meta.iloc[pos]
            if ('pos' in str(query.ionization_mode).lower()) != ('pos' in str(ref.ionization_mode).lower()):
                continue
            score, count = independent_cosine(query.ms2_mzs, query.ms2_normalized_intensities, mz[pos], intensity[pos], query.precursor_mz)
            mid = str(query.molecule_id)
            if count >= 4 and score > best[mid][0]:
                best[mid] = (score, count, str(ref.normalized_smiles), int(pos))
    remote = pd.read_parquet(folder / 'c09e_matches.parquet'); remote = remote[remote.view.eq('all_reference')].set_index('molecule_id')
    maximum_error = 0.
    for mid, (score, count, smiles, pos) in best.items():
        value = remote.loc[mid]; maximum_error = max(maximum_error, abs(score - float(value.cosine)))
        assert int(value.matched_peaks) == count and value.smiles == smiles
        if pos is not None:
            assert int(value.reference_row) == pos
    assert maximum_error < 1e-12
    evidence = pd.read_parquet(folder / 'frozen_candidate_evidence.parquet', columns=['molecule_id', 'library_max'])
    confidence = evidence.groupby('molecule_id').library_max.first().to_dict()
    identities = {}
    for name in ['neural_candidate_features.parquet', 'outsider_neural_features.parquet']:
        records = pd.read_parquet(folder / name, columns=['smiles', 'metric_key']).dropna()
        identities.update(dict(zip(records.smiles.astype(str), records.metric_key.astype(str))))
    tautomer = rdMolStandardize.TautomerEnumerator()
    @lru_cache(maxsize=30000)
    def identity(smiles):
        if smiles not in identities:
            mol = Chem.MolFromSmiles(smiles); assert mol is not None; Chem.RemoveStereochemistry(mol)
            identities[smiles] = Chem.MolToInchiKey(tautomer.Canonicalize(mol))[:14]
        return identities[smiles]
    changed = 0; passthrough = 0
    for old, actual in zip(baseline.itertuples(index=False), final.itertuples(index=False)):
        candidates = old.smiles.split(';'); expected = candidates.copy(); mid = str(old.molecule_id)
        if mid in best and np.isfinite(confidence.get(mid, np.nan)) and confidence[mid] < .9 and best[mid][0] >= .8:
            keys = [identity(s) for s in candidates]; key = identity(best[mid][2])
            if key != keys[0]:
                representation = candidates[keys.index(key)] if key in keys else best[mid][2]
                expected = ([representation] + [s for s, k in zip(candidates, keys) if k != key])[:25]
        else:
            passthrough += 1
        assert actual.smiles.split(';') == expected, mid
        keys = [identity(s) for s in expected]; assert 1 <= len(keys) <= 25 and len(keys) == len(set(keys))
        changed += int(old.smiles != actual.smiles)
    assert changed == receipt['changed_rows']
    index = ['molecule_id', 'smiles']
    public = pd.read_parquet(folder / 'neural_public.parquet').set_index(index)
    ours = pd.read_parquet(folder / 'neural_ours.parquet').set_index(index).reindex(public.index)
    primary = pd.read_parquet(folder / 'neural_candidate_features.parquet').set_index(index).reindex(public.index)
    def z(frame):
        values = frame.neural_score.fillna(0); groups = values.groupby(level=0)
        return (values - groups.transform('mean')) / groups.transform(lambda v: v.std(ddof=0)).clip(lower=1e-6)
    p01_error = float(np.abs(primary.neural_score - .5 * z(public) - .5 * z(ours)).max()); assert p01_error < 1e-6
    report = dict(valid_rows=400, real_queries=12, official_identity_duplicates=0, changed_rows=changed,
        exact_independent_merge=True, verified_passthrough_rows=passthrough, local_cloud_peak_score_max_abs_difference=maximum_error,
        independent_one_to_one_peak_assignment=True, unchanged_P01_primary_error=p01_error,
        source_files=pin['source_files'], reference_files=pin['reference_files'], csv_sha256=sha(folder / 'submission.csv'),
        changed_path_exercised=changed > 0, execution_qualified=changed > 0,
        scientific_confirmation_separate=True, automatic_submission=False)
    write(folder / 'independent_external_smoke_audit.json', report)
    print('DONE INDEPENDENT P16 AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
