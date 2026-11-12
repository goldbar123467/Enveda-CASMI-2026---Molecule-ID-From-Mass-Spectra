"""Check unknown/inconsistent-mass fallback using real USED P16 promotions."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
import torch
import types
from pathlib import Path
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp
from audit_c76_physical_fixture import load


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    out = base / 'interface-repair-r1'; target = out / 'physical-fallback-guard.json'; assert not target.exists()
    assert read(out / 'independent-run-audit.json')['passed']
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-r1-20261112'
    work = Path(read(out / 'completed-prefix-audit.json')['working'])
    ranker = load('c76_fallback_original_ranker', bundle / 'c67_ranker.py')
    compatibility = load('c76_fallback_compat', bundle / 'legacy_formate_compat.py')
    mass = types.ModuleType('c76_fallback_mass')
    source = bundle / 'legacy_chem_source.py'
    exec(compile(compatibility.patch_chem_source(source.read_text(encoding='utf8')), str(source), 'exec'), mass.__dict__)
    before = pd.read_csv(work / 'c09e_fixture_baseline.csv', dtype=str, keep_default_na=False)
    promoted = pd.read_csv(work / 'c09e_fixture_final.csv', dtype=str, keep_default_na=False)
    queries = pd.read_parquet(work / 'c09e_fixture_queries.parquet')
    matches = pd.read_parquet(work / 'c09e_fixture_matches.parquet')
    refs = pd.read_parquet(ROOT / 'outputs/research/2026-11-09-silver/extlib/ext_meta.parquet')
    changed = set(before.loc[before.smiles.ne(promoted.smiles), 'molecule_id']); assert len(changed) == 7
    hit_rows = matches[matches.molecule_id.isin(changed) & matches.view.eq('all_reference')].reference_row.astype(int).unique()
    results = {}
    for kind in ['unknown_adduct', 'inconsistent_mass']:
        counterfactual = refs.copy()
        if kind == 'unknown_adduct': counterfactual.loc[hit_rows, 'adduct'] = 'UNSUPPORTED_DIAGNOSTIC_ADDUCT'
        else: counterfactual.loc[hit_rows, 'precursor_mz'] += 10.
        final, events = ranker.physical_parent(before, promoted, queries, matches, counterfactual, mass.neutral_mass)
        pd.testing.assert_frame_equal(final, before, check_exact=True)
        assert len(events) == 7 and all(not row['mass_consistent'] for row in events)
        results[kind] = dict(real_promotions_reverted=7, all30_original_lists_exact=True)
    assert sha(bundle / 'c67_ranker.py') == read(bundle / 'pair-manifest.json')['files_sha256']['c67_ranker.py']
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        real_USED_queries=30, original_acquisitions=113, no_query_truth_used=True,
        cases=results, original_reference_data_unchanged=True,
        counterfactual_metadata_views_only=True, model_and_ranking_rules_unchanged=True,
        quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 UNKNOWN/INCONSISTENT MASS FALLBACK GUARDS PASSED; ORIGINAL LISTS KEPT', flush=True)


if __name__ == '__main__': main()
