"""Build an instrument-matched NP diagnostic with genuinely untrained decoys.

Public inputs only. Formula-oracle retrieval diagnostic, not a hidden-test score.
Canonical features: Saner Turhaner's Apache-2.0 kit, pinned RDKit 2026.03.3.
"""
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()


def encode(item):
    key, smiles, formula = item
    try:
        from features import structure
        canonical, _, actual_formula, _, fp = structure(smiles)
        return key, canonical, actual_formula, fp, None
    except Exception as exc:
        return key, '', '', None, type(exc).__name__


def main():
    import subprocess
    import torch
    try:
        from rdkit import rdBase
    except ImportError:
        wheel = [p for p in Path('/kaggle/input').rglob('rdkit-*.whl') if f'cp{sys.version_info.major}{sys.version_info.minor}' in p.name][0]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)], check=True)
        from rdkit import rdBase
    assert rdBase.rdkitVersion == '2026.03.3', rdBase.rdkitVersion
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    kit = next(p.parent for p in Path('/kaggle/input').rglob('features.py') if p.parent.name == 'casmi_spectrum_model_v1')
    sys.path.insert(0, str(kit))
    from features import structure
    mol = pd.read_parquet(kit / 'molecules.parquet')
    raw_np = pd.read_parquet(next(Path('/kaggle/input').rglob('np_raw_queries.parquet')))
    keys = set(raw_np.molecule_id.astype(str))
    assert len(keys) == 234
    targets = mol[mol.key.isin(keys)].sort_values('key', kind='stable')
    assert len(targets) == 234
    known, formulas = set(mol.key.astype(str)), set(targets.formula.astype(str))
    pool = next(Path('/kaggle/input').rglob('pool_popularity.csv'))
    parts = []
    for chunk in pd.read_csv(pool, usecols=['inchikey14', 'smiles', 'formula', 'np_coconut'], chunksize=100000):
        selected = chunk[chunk.formula.isin(formulas) & chunk.np_coconut.astype(bool) & ~chunk.inchikey14.isin(known)]
        parts.append(selected)
    eligible = pd.concat(parts, ignore_index=True).drop_duplicates('inchikey14')
    selected = []
    for formula, group in eligible.groupby('formula', sort=True):
        order = sorted(range(len(group)), key=lambda j: hashlib.sha256(('unseen-formula-v1|' + formula + '|' + str(group.inchikey14.iloc[j])).encode()).hexdigest())
        selected.extend(group.iloc[order[:256]][['inchikey14', 'smiles', 'formula']].itertuples(index=False, name=None))
    print('DECOYS raw eligible', len(eligible), 'deterministic pre-canonical subset', len(selected), flush=True)
    with mp.get_context('fork').Pool(4) as workers:
        encoded = list(workers.imap(encode, selected, chunksize=10))
    by_formula, rejected, seen = {}, {}, set()
    for raw_key, key, formula, fp, error in encoded:
        reason = error or ('training_identity' if key in known else 'duplicate_identity' if key in seen else '')
        if not reason and formula not in formulas:
            reason = 'formula_changed'
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        seen.add(key)
        by_formula.setdefault(formula, []).append((key, fp))
    arrays, key_arrays, counts = {}, {}, []
    for row in targets.itertuples(index=False):
        true_key, _, formula, _, fp = structure(row.smiles)
        assert true_key == row.key and formula == row.formula
        decoys = by_formula.get(formula, [])[:127]
        assert all(key not in known and key != true_key for key, _ in decoys)
        arrays[row.key] = np.stack([fp] + [value for _, value in decoys]).astype(np.uint8)
        key_arrays[row.key] = np.array([true_key] + [key for key, _ in decoys], dtype='U14')
        counts.append(len(decoys) + 1)
    np.savez_compressed('/kaggle/working/np_unseen_fingerprints.npz', **arrays)
    np.savez_compressed('/kaggle/working/np_unseen_candidate_keys.npz', **key_arrays)
    train = next(Path('/kaggle/input').rglob('train.parquet'))
    meta = pd.read_parquet(train, columns=['ingest_lib', 'instrument_type'])
    source_counts = meta.groupby('ingest_lib', observed=True).size().to_dict()
    native_libs = sorted(str(s) for s in meta.ingest_lib.dropna().unique() if 'enveda' in str(s).lower())
    del meta
    columns = ['inchikey14', 'normalized_smiles', 'ingest_lib', 'instrument_type', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev']
    native = pd.read_parquet(train, columns=columns, filters=[('ingest_lib', 'in', native_libs)])
    native.to_parquet('/kaggle/working/native_training_spectra.parquet', index=False)
    audit = dict(protocol='NP native timsTOF queries; truth plus up to 127 canonical exact-formula COCONUT decoys absent from ALL kit cache identities',
                 formula_oracle=True, target_count=234, fresh_targets=False, decoy_candidates_untrained=True,
                 raw_eligible_decoys=len(eligible), canonicalized=len(encoded), rejected=rejected,
                 unique_untrained_decoys=len(seen), candidate_count_quantiles={str(q): float(np.quantile(counts, q)) for q in [0, .25, .5, .75, 1]},
                 ambiguous_at_least_8=int(np.sum(np.array(counts) >= 8)),
                 source_counts={str(k): int(v) for k, v in source_counts.items()}, native_libraries=native_libs,
                 native_training_rows=len(native), native_raw_keys=int(native.inchikey14.nunique()),
                 rdkit=rdBase.rdkitVersion, seconds=time.time() - T0)
    Path('/kaggle/working/unseen_native_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    print('UNSEEN_NATIVE_PREP_DONE', round(time.time() - T0), 's', flush=True)


if __name__ == '__main__':
    main()
