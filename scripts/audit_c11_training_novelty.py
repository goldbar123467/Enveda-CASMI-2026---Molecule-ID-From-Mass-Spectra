"""Check training novelty before proposing an RTX2070 data-expansion pilot.

Reference lookup coverage and new training identities are distinct quantities.
Prior panel keys are used only for exclusion, never for recipe/score selection.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import json
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa

from precision_cache import ROOT, sha, write


def main():
    pa.set_cpu_count(2)
    out = ROOT / 'outputs/local/c11-new-source-20261110'
    assert not (out / 'novelty-audit.json').exists(), 'Completed inventory is sealed'
    source = ROOT / 'outputs/local/c10-reference-pool-20261110/supplement/ext_meta.parquet'
    raw_path = ROOT / 'outputs/competition-raw/train.parquet'
    source_audit = ROOT / 'outputs/research/c10-massbank-current-primary-20261110/audit.json'
    audited = json.loads(source_audit.read_text()); assert audited['all_records_verified']
    protocol = json.loads((source_audit.parent / 'protocol.json').read_text())
    assert sha(source) == protocol['supplemental_sha256']['ext_meta.parquet']
    assert sha(raw_path) == '9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98'
    frame = pd.read_parquet(source)
    raw = set(pd.read_parquet(raw_path, columns=['inchikey14']).inchikey14.dropna().astype(str).str[:14])
    paths = [ROOT / 'outputs/publish/c08-validation-20261108/labels.csv',
        ROOT / 'outputs/publish/c09-validation-20261109/labels.csv',
        ROOT / 'outputs/local/c10-casmi2022-20261110/labels.csv']
    prior = set()
    for path in paths: prior.update(pd.read_csv(path, usecols=['key']).key)
    absent = ~frame.inchikey14.isin(raw)
    allowed = ~frame.inchikey14.isin(raw | prior)
    initial = json.loads((out / 'feasibility.json').read_text())
    counts = dict(reference_rows=len(frame), raw_reference_keys=int(frame.inchikey14.nunique()),
        raw_training_keys=len(raw), absent_raw_training_rows=int(absent.sum()),
        absent_raw_training_keys=int(frame.loc[absent, 'inchikey14'].nunique()),
        rawkey_not_in_raw_training_or_previous_panels_keys=int(frame.loc[allowed, 'inchikey14'].nunique()))
    for key in ['raw_reference_keys', 'raw_training_keys', 'absent_raw_training_rows',
        'absent_raw_training_keys', 'rawkey_not_in_raw_training_or_previous_panels_keys']:
        assert counts[key] == initial[key], 'Inventory changed: ' + key
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), counts=counts,
        original_inventory_sha256=sha(out / 'feasibility.json'), auditor_sha256=sha(__file__),
        source_sha256=sha(source), original_source_audit_sha256=sha(source_audit), raw_training_sha256=sha(raw_path),
        prior_panel_labels_sha256={str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in paths},
        canonical_identity_audit_performed=False, novelty_counts_are_upper_bounds=True,
        novel_acquisition_audit_performed=False, previous_panel_identity_keys_used_only_for_exclusion=True,
        C10_reserved_identity_keys_read_for_exclusion=True, C10_prediction_metrics_read=False,
        C10_candidate_config_sha256=sha(ROOT / 'configs/c10-reference-candidates-20261110.json'),
        C10_development_protocol_sha256=sha(ROOT / 'outputs/local/c10-candidates-20261110/development-protocol.json'),
        C10_selection_changed=False, training_started=False, submission=False,
        decision='Do not launch a new-identity training expansion from this inventory. Await current full-pipeline evidence; the existing RTX2070 P15-control continuation remains scheduled.',
        limitation='Absence from the Takumu lookup library does not imply training novelty. Raw-key exclusions can still miss canonical aliases; new-acquisition novelty is not established. This is an inventory, not a model metric or a claim that all previously seen-identity acquisitions are equivalent.')
    write(out / 'novelty-audit.json', report)
    print('DONE C11 TRAINING NOVELTY AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__': main()
