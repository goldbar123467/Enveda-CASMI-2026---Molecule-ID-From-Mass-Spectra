"""Resolve mirror provenance against the pinned November official MassBank data."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, rdBase

from precision_cache import ROOT, sha, write
from audit_c10_primary_references import header_record, POOL
from build_external_cache import normalize_adduct
from training_monitor import system_memory

OUT = ROOT / 'outputs/research/c10-massbank-current-primary-20261110'
COMMIT = '72980b3af504cfd1e89ac54d7698d3da45d85a58'
URL = 'https://codeload.github.com/MassBank/MassBank-data/zip/' + COMMIT


def main():
    pa.set_cpu_count(2); assert rdBase.rdkitVersion == '2026.03.3'
    assert system_memory()['free_gib'] >= 4
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'audit.json').exists(), 'Completed audit is sealed'
    write(OUT / 'protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        official_tag='2026.11', official_commit=COMMIT, url=URL,
        published_utc='2026-11-09T12:18:27Z', auditor_sha256=sha(__file__),
        first_audit_sha256=sha(ROOT / 'outputs/research/c10-massbank-primary-20261110/audit.json'),
        supplemental_sha256={name: sha(POOL / 'supplement' / name) for name in ['ext_meta.parquet', 'ext_mz.npy', 'ext_in.npy']},
        identity_check='RDKit raw key from official SMILES/InChI; any nonempty official declared key must agree. Missing optional declared keys are counted explicitly.',
        query_labels_used=False, prediction_metrics_read=False, reference_pool_changed=False))
    archive = OUT / 'MassBank-data-2026.11-pinned.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.zip.partial')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk); size += len(chunk); assert size <= 512 * 1024**2
                if size % (32 * 1024**2) == 0: print('C10 CURRENT PRIMARY DOWNLOAD', size, flush=True)
        temporary.replace(archive)
    frame = pd.read_parquet(POOL / 'supplement/ext_meta.parquet')
    mz = np.load(POOL / 'supplement/ext_mz.npy', mmap_mode='r')
    intensity = np.load(POOL / 'supplement/ext_in.npy', mmap_mode='r')
    counts = collections.Counter(selected_records=len(frame)); licenses = collections.Counter(); issues = []; attribution = []
    with zipfile.ZipFile(archive) as data:
        assert {name.split('/')[0] for name in data.namelist()} == {'MassBank-data-' + COMMIT}
        assert sum(item.file_size for item in data.infolist()) < 4 * 1024**3
        lookup = collections.defaultdict(list)
        for member in data.infolist():
            if not member.is_dir() and member.filename.endswith('.txt'): lookup[Path(member.filename).name].append(member)
        for pos, row in enumerate(frame.itertuples(index=False)):
            members = lookup[row.spectrum_id + '.txt']; problems = []
            if len(members) != 1:
                counts['missing_or_ambiguous'] += 1; issues.append(dict(spectrum_id=row.spectrum_id, issue='missing_or_ambiguous')); continue
            member = members[0]; assert member.file_size < 2 * 1024**2
            raw = data.read(member); fields, peaks = header_record(raw.decode('utf-8-sig')); counts['found'] += 1
            def one(name):
                values = fields.get(name, []); return values[0] if len(values) == 1 else None
            license_value = one('LICENSE'); licenses[str(license_value)] += 1
            if one('ACCESSION') != row.spectrum_id: problems.append('accession')
            if license_value != row.license or license_value not in ['CC0', 'CC BY']: problems.append('license')
            if not one('AUTHORS') or one('AUTHORS') != row.authors: problems.append('authors')
            declared = [value.split(' ', 1)[1][:14] for value in fields.get('CH$LINK', []) if value.startswith('INCHIKEY ')]
            if not declared: counts['optional_declared_key_absent'] += 1
            elif declared != [row.inchikey14]: problems.append('declared_structure_key')
            mol = Chem.MolFromSmiles(one('CH$SMILES') or '')
            if mol is None and one('CH$IUPAC'):
                mol = Chem.MolFromInchi(one('CH$IUPAC')); counts['identity_from_primary_inchi'] += 1
            if mol is None or Chem.MolToInchiKey(mol)[:14] != row.inchikey14: problems.append('primary_structure')
            focused = dict(value.split(' ', 1) for value in fields.get('MS$FOCUSED_ION', []) if ' ' in value)
            if normalize_adduct(focused.get('PRECURSOR_TYPE')) != row.normalized_adduct: problems.append('adduct')
            try:
                if abs(float(focused['PRECURSOR_M/Z']) - float(row.precursor_mz)) > 1e-6: problems.append('precursor')
            except (KeyError, ValueError): problems.append('precursor')
            peaks = peaks[np.isfinite(peaks).all(1) & (peaks[:, 0] > 0) & (peaks[:, 1] > 0)]
            chosen = np.argsort(-peaks[:, 1], kind='stable')[:128]; chosen = chosen[np.argsort(peaks[chosen, 0], kind='stable')]
            a, b = np.zeros(128, np.float32), np.zeros(128, np.float32)
            a[:len(chosen)], b[:len(chosen)] = peaks[chosen, 0], peaks[chosen, 1]
            if not np.array_equal(a, mz[pos]): problems.append('mz_peaks')
            if not np.array_equal(b, intensity[pos]): problems.append('intensity_peaks')
            if problems:
                issues.append(dict(spectrum_id=row.spectrum_id, issue=','.join(problems)))
                for problem in problems: counts['mismatch_' + problem] += 1
            else: counts['verified_records'] += 1
            attribution.append(dict(spectrum_id=row.spectrum_id, official_path=member.filename,
                official_record_sha256=hashlib.sha256(raw).hexdigest(),
                license=license_value, authors=one('AUTHORS'), publication=one('PUBLICATION'), copyright=one('COPYRIGHT')))
            if (pos + 1) % 5000 == 0: print('C10 CURRENT PRIMARY CHECK', pos + 1, '/', len(frame), flush=True)
    pd.DataFrame(issues, columns=['spectrum_id', 'issue']).to_csv(OUT / 'issues.csv', index=False)
    pd.DataFrame(attribution).to_parquet(OUT / 'original-attribution.parquet', index=False)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), official_tag='2026.11', official_commit=COMMIT,
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, protocol_sha256=sha(OUT / 'protocol.json'),
        counts=dict(counts), original_licenses=dict(licenses), all_records_verified=counts['verified_records'] == len(frame),
        issue_rows=len(issues), issue_csv_sha256=sha(OUT / 'issues.csv'), original_attribution_sha256=sha(OUT / 'original-attribution.parquet'),
        query_labels_used=False, prediction_metrics_read=False, reference_pool_changed=False,
        source_review_complete=False,
        limitation='Original-record data/attribution verification only; not blanket legal clearance or exclusion of historical pretraining. Parent Takumu record provenance is still incomplete; mirror compilation notice is retained. No automatic candidate adoption/submission.')
    write(OUT / 'audit.json', report)
    print('DONE C10 CURRENT PRIMARY AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__': main()
