"""Four predeclared, label-blind C10 reference searches; P16 stays immutable."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa

from precision_cache import ROOT, sha, write, features
from external_reference_overlay import prepare, cosine, search as parent_search
from c09_spectrum_audit import signatures
from build_external_cache import normalize_adduct

OUT = ROOT / 'outputs/local/c10-candidates-20261110'
POOL = ROOT / 'outputs/local/c10-reference-pool-20261110'
QUERY = ROOT / 'outputs/publish/c10-casmi2022-20261110/validation-queries.parquet'
CONFIG = ROOT / 'configs/c10-reference-candidates-20261110.json'
POLICY = ROOT / 'configs/c09-external-reference-20261109.json'


def search_candidates(queries, reference_dir, parent_count, recipes, feature_module, policy):
    """Global input-alias exclusion, stable P16 peak matching and fixed adduct filters."""
    assert not {'key', 'smiles', 'inchikey14', 'partition'} & set(queries.columns)
    reference_dir = Path(reference_dir)
    meta = pd.read_parquet(reference_dir / 'ext_meta.parquet')
    mz = np.load(reference_dir / 'ext_mz.npy', mmap_mode='r')
    intensity = np.load(reference_dir / 'ext_in.npy', mmap_mode='r')
    assert mz.shape == intensity.shape == (len(meta), 128)
    assert 0 < parent_count <= len(meta)
    threshold = policy['thresholds']
    precursor = meta.precursor_mz.to_numpy(dtype=np.float64)
    order = np.argsort(precursor, kind='stable'); sorted_precursor = precursor[order]
    all_signatures, wanted, windows = set(), set(), []
    for row in queries.itertuples(index=False):
        sig = signatures(row, feature_module)
        if sig: all_signatures.update(sig)
        width = row.precursor_mz * threshold['precursor_ppm'] * 1e-6 + threshold['precursor_absolute_da']
        lo = np.searchsorted(sorted_precursor, row.precursor_mz - width, side='left')
        hi = np.searchsorted(sorted_precursor, row.precursor_mz + width, side='right')
        positions = order[lo:hi]; windows.append((row, positions)); wanted.update(positions.tolist())
    refs, aliases = {}, set()
    for pos in sorted(wanted):
        refs[pos] = prepare(mz[pos], intensity[pos])
        sig = signatures(SimpleNamespace(ms2_mzs=mz[pos], ms2_normalized_intensities=intensity[pos]), feature_module)
        if sig and any(value in all_signatures for value in sig): aliases.add(pos)
    matches = {(recipe['name'], str(mid), view): dict(cosine=0., matched_peaks=0, smiles='', reference_row=None)
        for recipe in recipes for mid in queries.molecule_id.unique()
        for view in ['all_reference', 'input_alias_excluded']}
    comparisons = 0
    for row, positions in windows:
        query = prepare(feature_module.array_value(row.ms2_mzs), feature_module.array_value(row.ms2_normalized_intensities), row.precursor_mz)
        positive = 'pos' in str(row.ionization_mode).lower()
        query_adduct = normalize_adduct(row.adduct)
        for pos in positions:
            ref = meta.iloc[pos]
            if ('pos' in str(ref.ionization_mode).lower()) != positive: continue
            ref_adduct = normalize_adduct(ref.adduct)
            score, count = cosine(query, refs[pos], threshold['fragment_tolerance_da']); comparisons += 1
            if count < threshold['minimum_matched_peaks']: continue
            for recipe in recipes:
                assert recipe['reference_pool'] in ['parent', 'expanded']
                assert recipe['adduct_rule'] in ['any_positive', 'same_normalized']
                if recipe['reference_pool'] == 'parent' and pos >= parent_count: continue
                if recipe['adduct_rule'] == 'same_normalized' and query_adduct != ref_adduct: continue
                for view in ['all_reference', 'input_alias_excluded']:
                    if view == 'input_alias_excluded' and pos in aliases: continue
                    best = matches[recipe['name'], str(row.molecule_id), view]
                    if score > best['cosine']:
                        best.update(cosine=score, matched_peaks=count, smiles=str(ref.normalized_smiles),
                            reference_row=int(pos), input_alias=pos in aliases,
                            same_adduct=str(row.adduct) == str(ref.adduct), normalized_adduct_match=query_adduct == ref_adduct,
                            reference_pool='parent' if pos < parent_count else 'massbank_curated')
    evidence = pd.DataFrame([dict(recipe=recipe, molecule_id=mid, view=view, **value)
        for (recipe, mid, view), value in matches.items()])
    report = dict(queries=int(queries.molecule_id.nunique()), spectra=len(queries), reference_rows=len(meta),
        precursor_candidate_rows=len(wanted), query_reference_comparisons=comparisons,
        query_input_alias_reference_rows=len(aliases), query_labels_used=False,
        limitation='Bounded full-round/FP16top96 input aliases only; no universal historical-pretraining exclusion')
    return evidence, report


def main():
    pa.set_cpu_count(2); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'search-report.json').exists(), 'Completed search is sealed'
    assert not (OUT / 'development.json').exists()
    manifest = json.loads((POOL / 'reference-manifest.json').read_text())
    assert sha(CONFIG) == manifest['candidate_config_sha256']
    for name, pin in manifest['expanded_sha256'].items(): assert sha(POOL / 'expanded' / name) == pin
    parent = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    for name, pin in manifest['original_parent_reference_sha256'].items(): assert sha(parent / name) == pin
    sources = ['c10_reference_search.py', 'external_reference_overlay.py', 'c09_spectrum_audit.py', 'build_external_cache.py']
    freeze = dict(prepared_utc=datetime.now(timezone.utc).isoformat(), config_sha256=sha(CONFIG),
        policy_sha256=sha(POLICY), query_sha256=sha(QUERY), reference_manifest_sha256=sha(POOL / 'reference-manifest.json'),
        source_sha256={name: sha(ROOT / 'scripts' / name) for name in sources},
        feature_sha256=sha(ROOT / 'outputs/local/kit/features.py'), labels_read=False, metrics_read=False)
    write(OUT / 'search-freeze.json', freeze)
    queries = pd.read_parquet(QUERY); policy = json.loads(POLICY.read_text())
    recipes = json.loads(CONFIG.read_text())['candidates_in_tie_order']
    matches, report = search_candidates(queries, POOL / 'expanded', manifest['parent_records'], recipes, features, policy)
    original, original_report = parent_search(queries, parent, features, policy)
    columns = ['molecule_id', 'view', 'cosine', 'matched_peaks', 'smiles', 'reference_row']
    actual = matches[matches.recipe.eq('P16')][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, original[columns], check_dtype=False, check_exact=True)
    matches.to_parquet(OUT / 'matches.parquet', index=False)
    report.update(completed_utc=datetime.now(timezone.utc).isoformat(), parent_exact_parity=True,
        parent_search=original_report, freeze_sha256=sha(OUT / 'search-freeze.json'),
        matches_sha256=sha(OUT / 'matches.parquet'), metrics_computed=False, recipe_selection=False)
    write(OUT / 'search-report.json', report)
    print('DONE C10 LABEL BLIND SEARCH', json.dumps(report), flush=True)


if __name__ == '__main__': main()
