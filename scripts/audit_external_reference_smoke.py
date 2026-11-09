"""Independent P16 peak assignment, merge, identity and local/cloud parity audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import ast
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


def independent_best(queries, meta, mz, intensity):
    precursor = meta.precursor_mz.to_numpy(dtype=float); order = np.argsort(precursor, kind='stable'); sorted_precursor = precursor[order]
    best = {str(mid): (0., 0, '', None) for mid in queries.molecule_id.unique()}
    for query in queries.itertuples(index=False):
        # The sealed serving feature parser converts query arrays to float32
        # before matching promotes them to float64. Parquet can store float64;
        # audit the served representation rather than its extra storage bits.
        query_mz = np.asarray(query.ms2_mzs, dtype=np.float32).reshape(-1)
        query_intensity = np.asarray(query.ms2_normalized_intensities, dtype=np.float32).reshape(-1)
        width = query.precursor_mz * 10e-6 + .002
        lo, hi = bisect.bisect_left(sorted_precursor, query.precursor_mz - width), bisect.bisect_right(sorted_precursor, query.precursor_mz + width)
        for pos in order[lo:hi]:
            ref = meta.iloc[pos]
            if ('pos' in str(query.ionization_mode).lower()) != ('pos' in str(ref.ionization_mode).lower()):
                continue
            score, count = independent_cosine(query_mz, query_intensity, mz[pos], intensity[pos], query.precursor_mz)
            mid = str(query.molecule_id)
            if count >= 4 and score > best[mid][0]:
                best[mid] = (score, count, str(ref.normalized_smiles), int(pos))
    return best


def compare_peak_matches(best, remote):
    remote = remote[remote.view.eq('all_reference')].set_index('molecule_id')
    assert set(remote.index) == set(best)
    maximum_error = 0.
    for mid, (score, count, smiles, pos) in best.items():
        value = remote.loc[mid]; maximum_error = max(maximum_error, abs(score - float(value.cosine)))
        assert int(value.matched_peaks) == count and value.smiles == smiles
        if pos is not None:
            assert int(value.reference_row) == pos
    assert maximum_error < 1e-12
    return maximum_error


def verify_lists(baseline, final, best, confidence, identity):
    assert baseline.molecule_id.tolist() == final.molecule_id.tolist()
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
    return changed, passthrough


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
    public_pins = None
    parent = json.loads((ROOT / 'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    for cell in parent['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MODEL_PINS' for t in node.targets):
                public_pins = ast.literal_eval(node.value)
    assert public_pins and receipt['baseline_model_source_audit']['public_files'] == public_pins
    assert receipt['baseline_model_source_audit']['second_checkpoint_sha256'] == 'c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46'
    assert receipt['baseline_model_source_audit']['second_shares_public_architecture_features_config']
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
    best = independent_best(queries, meta, mz, intensity)
    maximum_error = compare_peak_matches(best, pd.read_parquet(folder / 'c09e_matches.parquet'))
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
    changed, passthrough = verify_lists(baseline, final, best, confidence, identity)
    assert changed == receipt['changed_rows']
    fixture = json.loads((folder / 'p16_fixture_receipt.json').read_text())
    assert fixture == receipt['changed_path_fixture']
    assert fixture['real_queries'] == 30 and fixture['changed_rows'] == 7 and fixture['inserted_identities'] == 2
    assert fixture['invalid_structure_fallback_passed'] and fixture['exact_expected_lists']
    assert not fixture['query_truth_labels_used'] and not fixture['scoring_csv_modified']
    fixture_package = ROOT / 'outputs/publish/c09e-smoke-fixtures-20261109'
    assert sha(fixture_package / 'c09e-fixture-contract.json') == pin['fixture_contract_sha256'] == fixture['fixture_contract_sha256']
    contract = json.loads((fixture_package / 'c09e-fixture-contract.json').read_text())
    for name, expected in contract['files'].items():
        assert sha(fixture_package / name) == expected
    for name in ['c09e_fixture_queries.parquet', 'c09e_fixture_baseline.csv', 'c09e_fixture_library.csv', 'c09e_fixture_identities.parquet']:
        assert sha(folder / name) == contract['files'][name]
    fixture_queries = pd.read_parquet(folder / 'c09e_fixture_queries.parquet')
    fixture_before = pd.read_csv(folder / 'c09e_fixture_baseline.csv', dtype=str, keep_default_na=False)
    fixture_after = pd.read_csv(folder / 'c09e_fixture_final.csv', dtype=str, keep_default_na=False)
    assert set(fixture_before.molecule_id).isdisjoint(set(final.molecule_id))
    assert fixture_after.equals(pd.read_csv(fixture_package / 'c09e_fixture_expected.csv', dtype=str, keep_default_na=False))
    fixture_identities = pd.read_parquet(folder / 'c09e_fixture_identities.parquet')
    for smiles, key in zip(fixture_identities.smiles, fixture_identities.metric_key):
        assert smiles not in identities or identities[smiles] == key
        identities[smiles] = key
    fixture_best = independent_best(fixture_queries, meta, mz, intensity)
    fixture_error = compare_peak_matches(fixture_best, pd.read_parquet(folder / 'c09e_fixture_matches.parquet'))
    fixture_confidence = pd.read_csv(folder / 'c09e_fixture_library.csv').set_index('molecule_id').library_max.to_dict()
    fixture_changed, fixture_passthrough = verify_lists(fixture_before, fixture_after, fixture_best, fixture_confidence, identity)
    assert fixture_changed == fixture['changed_rows']
    index = ['molecule_id', 'smiles']
    public = pd.read_parquet(folder / 'neural_public.parquet').set_index(index)
    ours = pd.read_parquet(folder / 'neural_ours.parquet').set_index(index).reindex(public.index)
    primary = pd.read_parquet(folder / 'neural_candidate_features.parquet').set_index(index).reindex(public.index)
    assert not public.index.has_duplicates and not ours.index.has_duplicates and not primary.index.has_duplicates
    real = public.index.get_level_values('molecule_id').astype(str).isin(set(queries.molecule_id.astype(str)))
    assert real.any() and public.loc[real].index.get_level_values('molecule_id').nunique() == 12
    for frame in [public, ours, primary]:
        assert frame.loc[real, 'neural_ok'].astype(bool).all(), 'Real-query neural scoring fell back'
        assert not frame.loc[~real, 'neural_ok'].astype(bool).any()
        assert np.isfinite(frame.loc[real, 'neural_score'].to_numpy()).all()
    def z(frame):
        values = frame.neural_score.fillna(0); groups = values.groupby(level=0)
        return (values - groups.transform('mean')) / groups.transform(lambda v: v.std(ddof=0)).clip(lower=1e-6)
    p01_error = float(np.abs(primary.neural_score - .5 * z(public) - .5 * z(ours)).max()); assert p01_error < 1e-6
    report = dict(valid_rows=400, real_queries=12, official_identity_duplicates=0, changed_rows=changed,
        exact_independent_merge=True, verified_passthrough_rows=passthrough, local_cloud_peak_score_max_abs_difference=maximum_error,
        independent_one_to_one_peak_assignment=True, unchanged_P01_primary_error=p01_error,
        source_files=pin['source_files'], reference_files=pin['reference_files'], csv_sha256=sha(folder / 'submission.csv'),
        fixture_real_queries=30, fixture_changed_rows=fixture_changed, fixture_inserted_identities=2,
        fixture_local_cloud_peak_score_max_abs_difference=fixture_error, fixture_passthrough_rows=fixture_passthrough,
        invalid_structure_fallback_passed=True, baseline_model_source_audit=receipt['baseline_model_source_audit'],
        real_queries_supported_by_both_P01_models=12,
        independent_audit_source_sha256=sha(Path(__file__)),
        changed_path_exercised=fixture_changed > 0, execution_qualified=fixture_changed > 0,
        scientific_confirmation_separate=True, automatic_submission=False)
    write(folder / 'independent_external_smoke_audit.json', report)
    print('DONE INDEPENDENT P16 AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
