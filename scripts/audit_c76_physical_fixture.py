"""Independently check the physical rule on seven real P16 promotions."""
import importlib.util
import types
import numpy as np
import pandas as pd


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def audit(working, bundle, reference, output):
    ranker = load('c76_physical_checked_ranker', bundle / 'c67_ranker.py')
    compatibility = load('c76_physical_checked_compat', bundle / 'legacy_formate_compat.py')
    legacy = bundle / 'legacy_chem_source.py'
    chemistry = types.ModuleType('c76_physical_checked_mass')
    exec(compile(compatibility.patch_chem_source(legacy.read_text(encoding='utf8')), str(legacy), 'exec'), chemistry.__dict__)
    old = pd.read_csv(working / 'c09e_fixture_baseline.csv', dtype=str, keep_default_na=False)
    new = pd.read_csv(working / 'c09e_fixture_final.csv', dtype=str, keep_default_na=False)
    queries = pd.read_parquet(working / 'c09e_fixture_queries.parquet')
    matches = pd.read_parquet(working / 'c09e_fixture_matches.parquet')
    refs = pd.read_parquet(reference / 'ext_meta.parquet')
    chosen, events = ranker.physical_parent(old, new, queries, matches, refs, chemistry.neutral_mass)
    assert len(old) == len(new) == 30 and len(queries) == 113 and len(events) == 7
    rebuilt = new.copy(); accepted = rejected = 0
    aliases = {'[M+H-H2O]+': '[M-H2O+H]+', '[M+H-2H2O]+': '[M-2H2O+H]+'}
    def inferred(mz, token):
        data = chemistry.ADDUCTS.get(compatibility.normalize_legacy_adduct(token))
        if data is None: return float('nan')
        number, charge, delta, _ = data
        return (float(mz)*charge-delta)/number
    for i, row in old.iterrows():
        if row.smiles == new.smiles.iloc[i]: continue
        group = queries[queries.molecule_id.eq(row.molecule_id)]
        masses = np.array([inferred(r.precursor_mz, r.adduct) for r in group.itertuples(index=False)])
        masses = masses[np.isfinite(masses)]
        target = float(np.median(masses)) if len(masses) else float('nan')
        hit = matches[matches.molecule_id.eq(row.molecule_id) & matches.view.eq('all_reference')].iloc[0]
        ref = refs.iloc[int(hit.reference_row)]
        mass = inferred(ref.precursor_mz, aliases.get(ref.adduct, ref.adduct))
        keep = bool(np.isfinite(target) and np.isfinite(mass) and abs(target-mass) <= max(.002, target*1e-5))
        if keep: accepted += 1
        else: rejected += 1; rebuilt.at[i, 'smiles'] = row.smiles
    assert accepted + rejected == 7
    pd.testing.assert_frame_equal(chosen, rebuilt, check_exact=True)
    chosen.to_csv(output / 'independent-physical-fixture.csv', index=False)
    from c75_port_utils import write
    write(output / 'independent-physical-fixture-events.json', events)
    return dict(real_queries=30, original_acquisitions=113, promoted_lists_checked=7,
        mass_consistent_promotions_kept=accepted, inconsistent_or_unknown_promotions_reverted=rejected,
        independent_scalar_mass_and_final_list_reconstruction_exact=True, query_truth_labels_used=False)
