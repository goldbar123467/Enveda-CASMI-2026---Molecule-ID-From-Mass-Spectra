"""Count unused source identities for a fresh mixed-polarity confirmation panel.

Uses source truth structures only for identity exclusion. No prediction files,
coverage scores or model metrics are read. Does not create a qualified panel.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import json
import tarfile
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa

from precision_cache import ROOT, sha, write, features
from training_monitor import system_memory

OUT = ROOT / 'outputs/local/c13-confirmation-20261110'


def main():
    pa.set_cpu_count(2); assert system_memory()['free_gib'] >= 4
    assert not (OUT / 'inventory.json').exists()
    OUT.mkdir(exist_ok=True)
    archive = ROOT / 'outputs/research/c10-casmi2022-20261110/casmi_2022.tgz'
    assert sha(archive) == 'febe375faf786ab65950c50ec8f0c4ee8ea9c8ce6ff10b4cd68f25fcd5ce1e31'
    raw = ROOT / 'outputs/competition-raw/train.parquet'
    production = ROOT / 'outputs/research/c08-pipeline/reference/train_structs.parquet'
    forbidden = set(pd.read_parquet(ROOT / 'outputs/local/n0/cache/molecules.parquet', columns=['key']).key)
    for name in ['key', 'inchikey14', 'rawkey']:
        forbidden.update(pd.read_parquet(production, columns=[name])[name].dropna().astype(str).str[:14])
    forbidden.update(pd.read_parquet(raw, columns=['inchikey14']).inchikey14.dropna().astype(str).str[:14])
    prior = [ROOT / 'outputs/publish' / n / 'labels.csv' for n in ['c08-validation-20261108', 'c09-validation-20261109']]
    prior.append(ROOT / 'outputs/local/c10-casmi2022-20261110/labels.csv')
    for path in prior: forbidden.update(pd.read_csv(path, usecols=['key']).key)
    used_numbers = set(pd.read_csv(ROOT / 'outputs/local/c10-casmi2022-20261110/provenance.csv').source_number.astype(int))
    with tarfile.open(archive, 'r:gz') as source:
        meta = pd.read_csv(source.extractfile('casmi_2022/all_meta_results.csv'), header=1)
    counts = collections.Counter(source_rows=len(meta)); eligible = []
    for row in meta.to_dict('records'):
        number = int(row['Compound Number'])
        if number in [81, 282, 432, 476] or number in used_numbers:
            counts['known_error_or_previous_challenge'] += 1; continue
        mode = 'positive' if '_pos' in str(row['File']).lower() else 'negative' if '_neg' in str(row['File']).lower() else None
        if mode is None:
            counts['unrecognized_mode'] += 1; continue
        if str(row['InChIKey'])[:14] in forbidden:
            counts['raw_identity_excluded'] += 1; continue
        try: key, smiles, formula, mass, _ = features.structure(row['SMILES'])
        except (ValueError, TypeError):
            counts['invalid_structure'] += 1; continue
        if key in forbidden:
            counts['canonical_identity_excluded'] += 1; continue
        eligible.append(dict(source_number=number, mode=mode, key=key))
    frame = pd.DataFrame(eligible).drop_duplicates('key')
    by_mode = frame['mode'].value_counts().to_dict()
    frame.to_parquet(OUT / 'identity-inventory.parquet', index=False)
    write(OUT / 'inventory.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), source_sha256=sha(archive), raw_training_sha256=sha(raw),
        prior_label_sha256={str(p.relative_to(ROOT)): sha(p) for p in prior}, counts=dict(counts),
        unique_eligible_before_scan_and_raw_signature_audit=len(frame), mode_counts=by_mode,
        prospective_design='60positive/60negative confirmation identities, deterministic hash draw after strict scan/identity/raw-signature exclusions; fail if either stratum is too small.',
        minimum_counts_sufficient=all(by_mode.get(mode, 0) >= 60 for mode in ['positive', 'negative']),
        inventory_sha256=sha(OUT / 'identity-inventory.parquet'), identity_keys_used_only_for_exclusion=True,
        prediction_files_read=False, query_metrics_computed=False, qualified_panel=False,
        limitation='Upper-bound metadata inventory only; scan quality, source assignment and spectrum overlap still require validation. C10/C12 reserved metrics have been consumed and cannot be reused as fresh confirmation.'))
    print('DONE C13 UNUSED VALIDATION INVENTORY', json.dumps(by_mode), flush=True)


if __name__ == '__main__': main()
