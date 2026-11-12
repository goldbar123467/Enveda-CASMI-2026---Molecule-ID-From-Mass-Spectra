"""Strict C13 input-alias/reference masks, gated before reading its query map.

Preserve all C73 quarantine. P16 starts from the original complete reference;
no C72-specific precursor exclusions or selected rows are reused.
"""
import argparse
import os
import shutil
from types import SimpleNamespace
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_confirmation_preparation import (C73, ORIGINAL, PACKAGE, PREPARATION,
    prior, read_queries)
from c76_submission_contract import BINDING, POLICY

RAW = ROOT / 'outputs/competition-raw/train.parquet'
REFERENCE = ROOT / 'outputs/research/2026-11-09-silver/extlib'
MAIN_REFERENCE = ROOT / 'outputs/research/c08-pipeline/reference/train_structs.parquet'


def alias_closure(edges, old_raw, old_official):
    """A pure bipartite closure, including invalid official identities."""
    raw, official = set(old_raw), set(old_official); rounds = 0
    while True:
        previous = len(raw), len(official)
        for r, o in edges:
            if o is None or r in raw or o in official:
                raw.add(r)
                if o is not None: official.add(o)
        rounds += 1
        if previous == (len(raw), len(official)): return raw, official, rounds


def runtime():
    prior()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
    import torch
    global np, pd, pa, pq
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(2)
    from rdkit import RDLogger, rdBase
    RDLogger.DisableLog('rdApp.*'); assert rdBase.rdkitVersion == '2026.03.3'
    PREPARATION.mkdir(parents=True, exist_ok=True)
    return sha(__file__)


def raw_aliases():
    source_pin = runtime(); target = PREPARATION / 'raw-input-alias-audit.json'; assert not target.exists()
    assert not (PREPARATION / 'raw-input-alias-matches.parquet').exists(), 'Inspect a partial census rather than overwrite'
    q = read_queries()
    pin = read(ROOT / 'outputs/local/c39-source-supported-validation-20261111/preparation-protocol.json')['raw_training_sha256']
    assert sha(RAW) == pin
    from c74_complete_target import module
    feature = module('c76_c13_alias_features', 'c09e_features.py')
    signatures = module('c76_c13_alias_signatures', 'c09_spectrum_audit.py').signatures
    lookup = [{}, {}]
    for row in q.itertuples(index=False):
        values = signatures(row, feature); assert values and len(values) == 2
        for kind, value in enumerate(values): lookup[kind].setdefault(value, set()).add(row.molecule_id)
    precursor = pd.read_parquet(RAW, columns=['precursor_mz']).precursor_mz.to_numpy(float)
    order = np.argsort(precursor, kind='stable'); sorted_mass = precursor[order]; wanted = np.zeros(len(precursor), bool)
    for center in q.precursor_mz:
        width = center * 20e-6 + .003
        lo = np.searchsorted(sorted_mass, center-width, side='left'); hi = np.searchsorted(sorted_mass, center+width, side='right')
        wanted[order[lo:hi]] = True
    parquet = pq.ParquetFile(RAW); assert parquet.metadata.num_rows == len(precursor)
    records = []; offset = checked = 0
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows; positions = np.flatnonzero(wanted[offset:offset+count])
        if len(positions):
            table = parquet.read_row_group(group, columns=['ms2_mzs', 'ms2_normalized_intensities', 'inchikey14'], use_threads=False)
            frame = table.take(pa.array(positions)).to_pandas(); del table
            for pos, row in zip(positions, frame.itertuples(index=False)):
                checked += 1; values = signatures(row, feature)
                if not values: continue
                for kind, value in enumerate(values):
                    for mid in sorted(lookup[kind].get(value, [])):
                        records.append(dict(molecule_id=mid, raw_row=offset+int(pos), raw_key=str(row.inchikey14), signature_kind=kind))
            del frame
        offset += count
        write(PREPARATION / 'raw-input-alias-progress.json', dict(checked_utc=stamp(), pid=os.getpid(),
            raw_rows_completed=offset, raw_rows=len(precursor), candidate_rows_checked=checked, match_records=len(records)))
    assert offset == len(precursor) and checked == int(wanted.sum())
    assert sha(__file__) == source_pin
    frame = pd.DataFrame(records, columns=['molecule_id', 'raw_row', 'raw_key', 'signature_kind'])
    frame.to_parquet(PREPARATION / 'raw-input-alias-matches.parquet', index=False)
    write(PREPARATION / 'additional-raw-alias-quarantine.json', dict(keys=sorted(set(frame.raw_key)),
        scope='Global input spectral aliases; no query-to-truth mapping and no query removal'))
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), queries=120, acquisitions=120,
        query_sha256=sha(ORIGINAL / 'validation-queries.parquet'), raw_training_sha256=pin,
        raw_rows=len(precursor), candidate_rows_checked=checked, match_records=len(records), matched_raw_keys=frame.raw_key.nunique(),
        bounded_window='20ppm+0.003Da', full_and_top96_signatures=True, query_removal=0, query_labels_read=False,
        files_sha256={name: sha(PREPARATION / name) for name in ['raw-input-alias-matches.parquet', 'additional-raw-alias-quarantine.json']}))
    print('C76 ALL120 RESERVED INPUT ALIAS CENSUS COMPLETE', checked, len(records), flush=True)


def main_mask():
    source_pin = runtime(); target = PREPARATION / 'reference-exclusion-freeze.json'; assert not target.exists()
    raw_audit = read(PREPARATION / 'raw-input-alias-audit.json')
    assert raw_audit['source_sha256'] == sha(__file__)
    for name, pin in raw_audit['files_sha256'].items(): assert sha(PREPARATION / name) == pin
    original = read(C73 / 'reference-exclusion-freeze.json')
    for name, pin in original['files_sha256'].items(): assert sha(C73 / name) == pin
    previous_audit = read(C73 / 'independent-reference-mask-audit.json')
    assert previous_audit['passed'] and previous_audit['reference_exclusion_freeze_sha256'] == sha(C73 / 'reference-exclusion-freeze.json')
    assert previous_audit['retained_global_excluded_identity_formula_scaffold_or_near_chemistry'] == 0
    assert sha(MAIN_REFERENCE) == original['reference_structure_sha256']
    witness = pd.read_parquet(C73 / 'original-structure-identity-witnesses.parquet')
    edges = [(str(r.raw_key), None if pd.isna(r.official_key) else str(r.official_key)) for r in witness.itertuples(index=False)]
    old_raw = set(read(C73 / 'engine2-raw-key-exclusion.json')['keys'])
    added = set(read(PREPARATION / 'additional-raw-alias-quarantine.json')['keys'])
    old_official = set(witness.loc[witness.exclude, 'official_key'].dropna())
    raw, official, rounds = alias_closure(edges, old_raw | added, old_official)
    st = pd.read_parquet(MAIN_REFERENCE, columns=['sid', 'inchikey14', 'smiles'])
    old_mask = np.load(C73 / 'main-reference-sid-exclusion.npy', allow_pickle=False)
    assert old_mask.dtype == bool and len(st) == len(old_mask) == 275810
    assert np.array_equal(st.sid, np.arange(len(st)))
    lookup = dict(zip(witness.smiles, witness.official_key))
    mask = old_mask | st.inchikey14.astype(str).isin(raw).to_numpy() | np.array([
        pd.isna(lookup.get(str(s))) or lookup.get(str(s)) in official for s in st.smiles])
    assert mask[old_mask].all()
    assert sha(__file__) == source_pin
    witness['exclude'] = witness.exclude | witness.raw_key.isin(raw) | witness.official_key.isin(official) | witness.official_key.isna()
    witness.to_parquet(PREPARATION / 'original-structure-identity-witnesses.parquet', index=False)
    np.save(PREPARATION / 'main-reference-sid-exclusion.npy', mask)
    write(PREPARATION / 'engine2-raw-key-exclusion.json', dict(keys=sorted(raw), scope='Preserve all C73 chemistry/prior aliases plus all120 reserved input aliases; no query truth map'))
    names = ['original-structure-identity-witnesses.parquet', 'main-reference-sid-exclusion.npy', 'engine2-raw-key-exclusion.json']
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), complete_pipeline_binding_sha256=sha(BINDING),
        original_C73_reference_freeze_sha256=sha(C73 / 'reference-exclusion-freeze.json'),
        original_C73_independent_audit_sha256=sha(C73 / 'independent-reference-mask-audit.json'),
        raw_input_alias_audit_sha256=sha(PREPARATION / 'raw-input-alias-audit.json'),
        original_structure_sha256=sha(MAIN_REFERENCE), alias_closure_rounds=rounds,
        reference_SIDs=len(st), excluded_main_SIDs=int(mask.sum()), excluded_raw_keys=len(raw),
        all_prior_and_global231_chemistry_exclusions_preserved=True, queries=120, acquisitions=120,
        query_removal=0, query_labels_read=False, files_sha256={name: sha(PREPARATION / name) for name in names}))
    print('C76 RESERVED MAIN/ENGINE2 MASKS EXTENDED WITHOUT UNEXCLUDING ROWS', int(mask.sum()), flush=True)


def p16_mask():
    source_pin = runtime(); q = read_queries(); target = PREPARATION / 'masked-P16-reference'; assert not target.exists()
    from c74_complete_target import CODE, frozen_contract, module, descriptor, chemistry_flags, precursor_reachability
    contract = frozen_contract()
    for name, pin in contract['reference_files'].items(): assert sha(REFERENCE / name) == pin
    main = read(PREPARATION / 'reference-exclusion-freeze.json')
    assert main['source_sha256'] == sha(__file__)
    for name, pin in main['files_sha256'].items(): assert sha(PREPARATION / name) == pin
    features = module('c76_c13_reference_identity', 'c09e_features.py')
    signatures = module('c76_c13_reference_signatures', 'c09_spectrum_audit.py').signatures
    global_path = C73 / 'global-exclusion-chemistry.parquet'
    classification = {name.replace('\\', '/'): pin for name, pin in read(C73 / 'chemistry-classification-freeze.json')['files_sha256'].items()}
    assert classification[global_path.relative_to(ROOT).as_posix()] == sha(global_path)
    held = pd.read_parquet(global_path); described = [descriptor(s, features) for s in held.smiles]
    assert [d['key'] for d in described] == held.key.tolist() and [d['formula'] for d in described] == held.formula.tolist()
    keys = {d['key'] for d in described}; forms = {d['formula'] for d in described}
    scaffolds = {d['scaffold'] for d in described if d['scaffold']}; fps = [d['fp'] for d in described]
    raw = set(read(PREPARATION / 'engine2-raw-key-exclusion.json')['keys'])
    witness = pd.read_parquet(PREPARATION / 'original-structure-identity-witnesses.parquet')
    official = set(witness.loc[witness.exclude, 'official_key'].dropna()) | keys
    query_signatures = set()
    for row in q.itertuples(index=False): query_signatures.update(signatures(row, features) or [])
    meta = pd.read_parquet(REFERENCE / 'ext_meta.parquet'); assert len(meta) == 200724
    mz = np.load(REFERENCE / 'ext_mz.npy', mmap_mode='r'); intensity = np.load(REFERENCE / 'ext_in.npy', mmap_mode='r')
    assert mz.shape == intensity.shape == (len(meta), 128)
    reachable = precursor_reachability(q, meta.precursor_mz, read(CODE / 'c09e_policy.json')['thresholds'])
    structural = ['invalid_identity', 'query_identity', 'known_official_alias', 'same_formula', 'same_nonempty_scaffold', 'near_query_tanimoto']
    records = []
    for i, text in enumerate(dict.fromkeys(meta.loc[reachable, 'normalized_smiles'].astype(str)), 1):
        try:
            d = descriptor(text, features)
            row = dict(smiles=text, official_key=d['key'], formula=d['formula'], scaffold=d['scaffold'], invalid_identity=False,
                **chemistry_flags(d, keys, forms, scaffolds, fps, official))
        except (ValueError, TypeError, RuntimeError):
            row = dict(smiles=text, official_key=None, formula=None, scaffold=None, invalid_identity=True,
                maximum_query_tanimoto=None, **{name: False for name in structural if name != 'invalid_identity'})
        records.append(row)
        if i % 2500 == 0: print('C76 RESERVED P16 REFERENCE IDENTITIES', i, flush=True)
    descriptors = pd.DataFrame(records).set_index('smiles')
    flags = pd.DataFrame(False, index=np.arange(len(meta)), columns=structural)
    flags.loc[reachable, structural] = descriptors.loc[meta.loc[reachable, 'normalized_smiles'].astype(str), structural].to_numpy(bool)
    flags['outside_all_frozen_precursor_windows'] = ~reachable
    flags['known_raw_alias'] = meta.inchikey14.astype(str).isin(raw)
    aliases = np.zeros(len(meta), bool)
    for pos in np.flatnonzero(reachable):
        aliases[pos] = bool(query_signatures.intersection(signatures(SimpleNamespace(ms2_mzs=mz[pos], ms2_normalized_intensities=intensity[pos]), features) or []))
    flags['query_input_alias'] = aliases; keep = np.flatnonzero(~flags.any(axis=1).to_numpy())
    assert len(keep) > 0 and sha(__file__) == source_pin
    target.mkdir(); descriptors.reset_index().to_parquet(target / 'identity-audit.parquet', index=False)
    flags.insert(0, 'original_row', np.arange(len(meta))); flags.to_parquet(target / 'row-exclusion-audit.parquet', index=False)
    meta.iloc[keep].reset_index(drop=True).to_parquet(target / 'ext_meta.parquet', index=False)
    for name, original in [('ext_mz.npy', mz), ('ext_in.npy', intensity)]:
        output = np.lib.format.open_memmap(target / name, mode='w+', dtype=original.dtype, shape=(len(keep), 128))
        for start in range(0, len(keep), 2048): output[start:start+2048] = original[keep[start:start+2048]]
        output.flush(); del output
    write(target / 'mask-manifest.json', dict(frozen_utc=stamp(), source_sha256=sha(__file__), policy_sha256=sha(POLICY),
        original_reference_files=contract['reference_files'], global_exclusion_chemistry_sha256=sha(global_path),
        main_reference_freeze_sha256=sha(PREPARATION / 'reference-exclusion-freeze.json'), query_sha256=sha(ORIGINAL / 'validation-queries.parquet'),
        original_rows=len(meta), exact_precursor_reachable_rows=int(reachable.sum()), retained_rows=len(keep),
        queries=120, acquisitions=120, original_full_reference_used=True, C72_outside_window_flags_reused=False,
        all_prior_and_global231_chemistry_and_alias_masks_preserved=True, query_labels_read=False,
        reason_counts={name: int(flags[name].sum()) for name in flags if name != 'original_row'},
        reference_files={name: sha(target / name) for name in contract['reference_files']},
        audit_files={name: sha(target / name) for name in ['identity-audit.parquet', 'row-exclusion-audit.parquet']}))
    print('C76 RESERVED P16 MASK PREPARED FROM COMPLETE ORIGINAL REFERENCE', len(keep), flush=True)


def package():
    prior(); audit = read(PREPARATION / 'independent-reference-audit.json')
    assert audit['passed'] and audit['forbidden_reference_rows'] == 0
    for name, pin in audit['artifact_sha256'].items(): assert sha(ROOT / name) == pin
    assert not PACKAGE.exists(); PACKAGE.mkdir()
    for name in ['validation-queries.parquet', 'validation-sample.csv']: shutil.copyfile(ORIGINAL / name, PACKAGE / name)
    for name in ['main-reference-sid-exclusion.npy', 'engine2-raw-key-exclusion.json']: shutil.copyfile(PREPARATION / name, PACKAGE / name)
    (PACKAGE / 'ATTRIBUTION.txt').write_text('Private original CASMI2022 FT/HCD120 stress confirmation,60positive/60negative,one acquisition per query,unknown CE; original tokens and peak arrays preserved. DOI10.5281/zenodo.8371051,CC-BY-4.0. Global C73 USED/fresh512/reserved231 chemistry and prior aliases remain quarantined, plus every bounded C13 input alias. No per-query structure labels or truth maps included. Public-parent/pretraining exposure remains disclosed; this is not target timsTOF support or public-score calibration.\n', encoding='utf8')
    names = ['validation-queries.parquet', 'validation-sample.csv', 'main-reference-sid-exclusion.npy', 'engine2-raw-key-exclusion.json', 'ATTRIBUTION.txt']
    write(PACKAGE / 'validation-contract.json', dict(prepared_utc=stamp(), queries=120, spectra=120,
        complete_pipeline_binding_sha256=sha(BINDING), reference_mask_required=True, truth_injection=False,
        query_labels_visible_to_pipeline=False, independent_reference_audit_sha256=sha(PREPARATION / 'independent-reference-audit.json'),
        files={name: sha(PACKAGE / name) for name in names}))
    from c76_confirmation_preparation import DATASET
    write(PACKAGE / 'dataset-metadata.json', dict(id=DATASET, title='CASMI26 C76 reserved original120 masked inputs', licenses=[dict(name='CC-BY-4.0')]))
    print('C76 RESERVED QUERY-ONLY MASKED PACKAGE PREPARED; NOT PUBLISHED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['raw', 'main', 'p16', 'package']); args = parser.parse_args()
    {'raw': raw_aliases, 'main': main_mask, 'p16': p16_mask, 'package': package}[args.phase]()
