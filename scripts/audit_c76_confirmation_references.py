"""Independently rebuild reserved alias census and all retained reference rows.

Uses direct mass-window comparisons and a graph traversal instead of the
preparer's sorted searches and iterative closure. No query truth is opened.
"""
import os
from collections import defaultdict, deque
from types import SimpleNamespace
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_confirmation_preparation import C73, ORIGINAL, PREPARATION, prior, read_queries
from c76_submission_contract import BINDING, POLICY


def graph_closure(edges, old_raw, old_official):
    graph = defaultdict(set); seeds = {('r', key) for key in old_raw} | {('o', key) for key in old_official}
    for raw, official in edges:
        if official is None: seeds.add(('r', raw)); continue
        a, b = ('r', raw), ('o', official); graph[a].add(b); graph[b].add(a)
    seen = set(seeds); queue = deque(seeds)
    while queue:
        for node in graph.get(queue.popleft(), ()):
            if node not in seen: seen.add(node); queue.append(node)
    return {key for kind, key in seen if kind == 'r'}, {key for kind, key in seen if kind == 'o'}


def main():
    prior(); target = PREPARATION / 'independent-reference-audit.json'; assert not target.exists()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
    import torch
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from c74_complete_target import CODE, frozen_contract, module
    pa.set_cpu_count(2); RDLogger.DisableLog('rdApp.*'); assert rdBase.rdkitVersion == '2026.03.3'
    raw_path = ROOT / 'outputs/competition-raw/train.parquet'
    main_reference = ROOT / 'outputs/research/c08-pipeline/reference/train_structs.parquet'
    reference = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    q = read_queries(); source_pin = sha(__file__)
    feature = module('c76_c13_independent_mask_identity', 'c09e_features.py')
    signatures = module('c76_c13_independent_mask_signatures', 'c09_spectrum_audit.py').signatures
    raw_audit_path = PREPARATION / 'raw-input-alias-audit.json'; raw_audit = read(raw_audit_path)
    preparer = ROOT / 'scripts/prepare_c76_confirmation_references.py'
    assert raw_audit['source_sha256'] == sha(preparer)
    assert raw_audit['query_sha256'] == sha(ORIGINAL / 'validation-queries.parquet')
    assert raw_audit['raw_training_sha256'] == sha(raw_path)
    for name, pin in raw_audit['files_sha256'].items(): assert sha(PREPARATION / name) == pin
    lookup = [{}, {}]; all_signatures = set()
    for row in q.itertuples(index=False):
        values = signatures(row, feature); assert values and len(values) == 2
        for kind, value in enumerate(values): lookup[kind].setdefault(value, set()).add(row.molecule_id)
        all_signatures.update(values)
    mass = pd.read_parquet(raw_path, columns=['precursor_mz']).precursor_mz.to_numpy(float)
    wanted = np.zeros(len(mass), bool)
    for center in q.precursor_mz:
        width = center*20e-6 + .003; wanted |= (mass >= center-width) & (mass <= center+width)
    assert int(wanted.sum()) == raw_audit['candidate_rows_checked'] and len(mass) == raw_audit['raw_rows']
    parquet = pq.ParquetFile(raw_path); rebuilt = set(); offset = 0
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows; positions = np.flatnonzero(wanted[offset:offset+count])
        if len(positions):
            table = parquet.read_row_group(group, columns=['ms2_mzs', 'ms2_normalized_intensities', 'inchikey14'], use_threads=False)
            frame = table.take(pa.array(positions)).to_pandas(); del table
            for pos, row in zip(positions, frame.itertuples(index=False)):
                values = signatures(row, feature)
                if values:
                    for kind, value in enumerate(values):
                        rebuilt.update((mid, offset+int(pos), str(row.inchikey14), kind) for mid in lookup[kind].get(value, []))
            del frame
        offset += count
    saved = pd.read_parquet(PREPARATION / 'raw-input-alias-matches.parquet')
    recorded = set(saved[['molecule_id', 'raw_row', 'raw_key', 'signature_kind']].itertuples(index=False, name=None))
    assert offset == len(mass) and len(saved) == len(recorded) == len(rebuilt) == raw_audit['match_records']
    assert recorded == rebuilt
    added = set(read(PREPARATION / 'additional-raw-alias-quarantine.json')['keys'])
    assert added == {row[2] for row in rebuilt} and len(added) == raw_audit['matched_raw_keys']
    main_path = PREPARATION / 'reference-exclusion-freeze.json'; main = read(main_path)
    assert main['source_sha256'] == sha(preparer) and main['raw_input_alias_audit_sha256'] == sha(raw_audit_path)
    original = read(C73 / 'reference-exclusion-freeze.json')
    assert main['original_C73_reference_freeze_sha256'] == sha(C73 / 'reference-exclusion-freeze.json')
    assert main['original_C73_independent_audit_sha256'] == sha(C73 / 'independent-reference-mask-audit.json')
    assert read(C73 / 'independent-reference-mask-audit.json')['retained_global_excluded_identity_formula_scaffold_or_near_chemistry'] == 0
    assert main['original_structure_sha256'] == original['reference_structure_sha256'] == sha(main_reference)
    for name, pin in original['files_sha256'].items(): assert sha(C73 / name) == pin
    for name, pin in main['files_sha256'].items(): assert sha(PREPARATION / name) == pin
    old_witness = pd.read_parquet(C73 / 'original-structure-identity-witnesses.parquet')
    witness = pd.read_parquet(PREPARATION / 'original-structure-identity-witnesses.parquet')
    pd.testing.assert_frame_equal(old_witness.drop(columns='exclude'), witness.drop(columns='exclude'), check_exact=True)
    edges = [(str(r.raw_key), None if pd.isna(r.official_key) else str(r.official_key)) for r in old_witness.itertuples(index=False)]
    raw, official = graph_closure(edges, set(read(C73 / 'engine2-raw-key-exclusion.json')['keys']) | added,
        set(old_witness.loc[old_witness.exclude, 'official_key'].dropna()))
    assert raw == set(read(PREPARATION / 'engine2-raw-key-exclusion.json')['keys'])
    expected_exclusion = old_witness.exclude | old_witness.raw_key.isin(raw) | old_witness.official_key.isin(official) | old_witness.official_key.isna()
    np.testing.assert_array_equal(witness.exclude, expected_exclusion)
    assert not witness.groupby('raw_key').exclude.nunique().gt(1).any()
    assert not witness[witness.official_key.notna()].groupby('official_key').exclude.nunique().gt(1).any()
    st = pd.read_parquet(main_reference); old_mask = np.load(C73 / 'main-reference-sid-exclusion.npy', allow_pickle=False)
    mask = np.load(PREPARATION / 'main-reference-sid-exclusion.npy', allow_pickle=False)
    mapping = dict(zip(old_witness.smiles, old_witness.official_key))
    expected = np.asarray([bool(old_mask[i]) or str(row.inchikey14) in raw or pd.isna(mapping.get(str(row.smiles)))
        or mapping.get(str(row.smiles)) in official for i, row in enumerate(st.itertuples(index=False))])
    assert np.array_equal(st.sid, np.arange(len(st))) and len(st) == 275810
    np.testing.assert_array_equal(mask, expected); assert mask[old_mask].all()
    assert int(mask.sum()) == main['excluded_main_SIDs'] and len(raw) == main['excluded_raw_keys']
    p16 = PREPARATION / 'masked-P16-reference'; manifest_path = p16 / 'mask-manifest.json'; manifest = read(manifest_path)
    assert manifest['source_sha256'] == sha(preparer) and manifest['policy_sha256'] == sha(POLICY)
    assert manifest['query_sha256'] == sha(ORIGINAL / 'validation-queries.parquet')
    assert manifest['main_reference_freeze_sha256'] == sha(main_path)
    contract = frozen_contract(); assert manifest['original_reference_files'] == contract['reference_files']
    for name, pin in manifest['original_reference_files'].items(): assert sha(reference / name) == pin
    for name, pin in {**manifest['reference_files'], **manifest['audit_files']}.items(): assert sha(p16 / name) == pin
    global_path = C73 / 'global-exclusion-chemistry.parquet'
    assert manifest['global_exclusion_chemistry_sha256'] == sha(global_path)
    classification = {name.replace('\\', '/'): pin for name, pin in read(C73 / 'chemistry-classification-freeze.json')['files_sha256'].items()}
    assert classification[global_path.relative_to(ROOT).as_posix()] == sha(global_path)
    held = pd.read_parquet(global_path)
    def describe(text):
        key, canonical, formula, _, bits = feature.structure(text)
        fp = DataStructs.ExplicitBitVect(len(bits)); fp.SetBitsFromList(np.flatnonzero(bits).tolist())
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=canonical, includeChirality=False)
        return key, formula, sc, fp
    descriptions = [describe(text) for text in held.smiles]
    assert [d[0] for d in descriptions] == held.key.tolist() and [d[1] for d in descriptions] == held.formula.tolist()
    keys = set(held.key); forms = set(held.formula); scaffolds = {d[2] for d in descriptions if d[2]}; fps = [d[3] for d in descriptions]
    official |= keys
    meta = pd.read_parquet(reference / 'ext_meta.parquet'); selected = pd.read_parquet(p16 / 'ext_meta.parquet')
    rows = pd.read_parquet(p16 / 'row-exclusion-audit.parquet'); descriptors = pd.read_parquet(p16 / 'identity-audit.parquet').set_index('smiles')
    assert len(meta) == len(rows) == manifest['original_rows'] == 200724
    np.testing.assert_array_equal(rows.original_row, np.arange(len(meta)))
    thresholds = read(CODE / 'c09e_policy.json')['thresholds']; mass = meta.precursor_mz.to_numpy(float); reachable = np.zeros(len(meta), bool)
    for center in q.precursor_mz:
        width = center*thresholds['precursor_ppm']*1e-6 + thresholds['precursor_absolute_da']
        reachable |= (mass >= center-width) & (mass <= center+width)
    np.testing.assert_array_equal(rows.outside_all_frozen_precursor_windows, ~reachable)
    assert int(reachable.sum()) == manifest['exact_precursor_reachable_rows']
    structural = ['invalid_identity', 'query_identity', 'known_official_alias', 'same_formula', 'same_nonempty_scaffold', 'near_query_tanimoto']
    assert set(descriptors.index) == set(meta.loc[reachable, 'normalized_smiles'].astype(str)) and descriptors.index.is_unique
    for text, saved in descriptors.iterrows():
        try:
            key, formula, sc, fp = describe(text)
            similarity = max(DataStructs.BulkTanimotoSimilarity(fp, fps), default=0.)
            expected = [False, key in keys, key in official, formula in forms, bool(sc and sc in scaffolds), similarity >= .7]
            assert (saved.official_key, saved.formula, saved.scaffold) == (key, formula, sc)
            assert abs(saved.maximum_query_tanimoto - similarity) < 1e-14
        except (ValueError, TypeError, RuntimeError): expected = [True, False, False, False, False, False]
        assert saved[structural].tolist() == expected
    np.testing.assert_array_equal(rows.loc[reachable, structural].to_numpy(bool),
        descriptors.loc[meta.loc[reachable, 'normalized_smiles'].astype(str), structural].to_numpy(bool))
    assert not rows.loc[~reachable, structural].any().any()
    np.testing.assert_array_equal(rows.known_raw_alias, meta.inchikey14.astype(str).isin(raw))
    mz = np.load(reference / 'ext_mz.npy', mmap_mode='r'); intensity = np.load(reference / 'ext_in.npy', mmap_mode='r')
    aliases = np.zeros(len(meta), bool)
    for pos in np.flatnonzero(reachable): aliases[pos] = bool(all_signatures.intersection(signatures(
        SimpleNamespace(ms2_mzs=mz[pos], ms2_normalized_intensities=intensity[pos]), feature) or []))
    np.testing.assert_array_equal(rows.query_input_alias, aliases)
    flags = rows.drop(columns='original_row'); keep = np.flatnonzero(~flags.any(axis=1).to_numpy())
    assert len(keep) == len(selected) == manifest['retained_rows'] and len(keep) > 0
    assert {name: int(flags[name].sum()) for name in flags} == manifest['reason_counts']
    pd.testing.assert_frame_equal(selected, meta.iloc[keep].reset_index(drop=True), check_exact=True)
    for name in ['ext_mz.npy', 'ext_in.npy']:
        original_array = np.load(reference / name, mmap_mode='r'); actual = np.load(p16 / name, mmap_mode='r')
        assert original_array.shape == (len(meta), 128) and actual.shape == (len(keep), 128) and actual.dtype == original_array.dtype
        for start in range(0, len(keep), 2048): np.testing.assert_array_equal(actual[start:start+2048], original_array[keep[start:start+2048]])
    assert sha(__file__) == source_pin
    files = [raw_audit_path, main_path, manifest_path, ORIGINAL / 'validation-queries.parquet', global_path]
    files += [PREPARATION / name for name in raw_audit['files_sha256']]
    files += [PREPARATION / name for name in main['files_sha256']]
    files += [p16 / name for name in {**manifest['reference_files'], **manifest['audit_files']}]
    write(target, dict(completed_utc=stamp(), source_sha256=source_pin, passed=True,
        complete_pipeline_binding_sha256=sha(BINDING), policy_sha256=sha(POLICY), queries=120, acquisitions=120,
        raw_alias_census_independently_recomputed=True, every_raw_and_official_alias_component_closed=True,
        all_prior_and_global231_chemistry_and_alias_masks_preserved=True, main_SID_mask_exact=True,
        original_full200724_reference_used=True, all_reachable_structure_and_peak_alias_flags_independently_rebuilt=True,
        retained_reference_order_metadata_and_peak_arrays_exact=True, forbidden_reference_rows=0,
        artifact_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files},
        query_labels_read=False, query_removal=0, scoring_entries=0))
    print('C76 RESERVED REFERENCE AND ALL120 INPUT ALIAS AUDIT PASSED', flush=True)


if __name__ == '__main__': main()
