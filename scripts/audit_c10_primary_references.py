"""Compare the frozen supplemental records with the official MassBank release.

Reads plain data from a checksum-verified ZIP, never extracts or executes code.
Does not use query labels/metrics or change any reference pool or serving policy.
"""
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

from precision_cache import ROOT, sha, write
from build_external_cache import normalize_adduct
from training_monitor import system_memory

OUT = ROOT / 'outputs/research/c10-massbank-primary-20261110'
POOL = ROOT / 'outputs/local/c10-reference-pool-20261110'
META_URL = 'https://zenodo.org/api/records/19073053'
KEY = 'MassBank/MassBank-data-2026.03.zip'
MD5 = '51904e29ee8717756153a8d31726789d'
SIZE = 239602156


def header_record(text):
    fields = collections.defaultdict(list); peaks = []; in_peaks = False
    for line in text.splitlines():
        if line.startswith('PK$PEAK:'): in_peaks = True; continue
        if line.startswith('//'): break
        if in_peaks:
            parts = line.split()
            if parts:
                assert len(parts) == 3
                peaks.append((float(parts[0]), float(parts[1])))
        elif ': ' in line:
            name, value = line.split(': ', 1); fields[name].append(value.strip())
    return dict(fields), np.asarray(peaks, dtype=np.float32).reshape(-1, 2)


def main():
    pa.set_cpu_count(2); OUT.mkdir(parents=True, exist_ok=True)
    assert system_memory()['free_gib'] >= 4
    assert not (OUT / 'audit.json').exists(), 'Completed source audit is sealed'
    write(OUT / 'protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        source= META_URL, official_release='2026.03', expected_md5=MD5, expected_bytes=SIZE,
        auditor_sha256=sha(__file__), supplemental_meta_sha256=sha(POOL / 'supplement/ext_meta.parquet'),
        query_labels_used=False, prediction_metrics_read=False, reference_pool_changed=False))
    with urllib.request.urlopen(META_URL, timeout=60) as stream: metadata = json.load(stream)
    entry = next(file for file in metadata['files'] if file['key'] == KEY)
    assert entry['size'] == SIZE and entry['checksum'] == 'md5:' + MD5
    assert metadata['metadata']['doi'] == '10.5281/zenodo.19073053'
    write(OUT / 'zenodo-metadata.json', metadata)
    archive = OUT / 'MassBank-data-2026.03.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.zip.partial')
        request = urllib.request.Request(entry['links']['self'], headers={'User-Agent': 'CASMI26-source-reproducibility-audit/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk); downloaded += len(chunk)
                assert downloaded <= SIZE
                if downloaded % (16 * 1024 * 1024) == 0: print('C10 PRIMARY DOWNLOAD', downloaded, '/', SIZE, flush=True)
        assert temporary.stat().st_size == SIZE
        temporary.replace(archive)
    digest = hashlib.md5()
    with archive.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024): digest.update(chunk)
    assert archive.stat().st_size == SIZE and digest.hexdigest() == MD5
    frame = pd.read_parquet(POOL / 'supplement/ext_meta.parquet')
    assert frame.spectrum_id.is_unique
    frozen_mz = np.load(POOL / 'supplement/ext_mz.npy', mmap_mode='r')
    frozen_in = np.load(POOL / 'supplement/ext_in.npy', mmap_mode='r')
    issues = []; counts = collections.Counter(selected_records=len(frame)); licenses = collections.Counter()
    with zipfile.ZipFile(archive) as data:
        assert sum(item.file_size for item in data.infolist()) <= 4 * 1024**3
        lookup = collections.defaultdict(list)
        for item in data.infolist():
            if not item.is_dir() and item.filename.endswith('.txt'): lookup[Path(item.filename).name].append(item)
        for pos, row in enumerate(frame.itertuples(index=False)):
            names = lookup.get(row.spectrum_id + '.txt', [])
            if len(names) != 1:
                counts['missing_or_ambiguous_primary_record'] += 1
                issues.append(dict(spectrum_id=row.spectrum_id, issue='primary_record_not_unique', found=len(names))); continue
            member = names[0]; assert member.file_size < 2 * 1024**2
            fields, peaks = header_record(data.read(member).decode('utf-8-sig'))
            counts['found'] += 1
            def one(key):
                values = fields.get(key, []); return values[0] if len(values) == 1 else None
            license_value = one('LICENSE'); licenses[str(license_value)] += 1
            problems = []
            if one('ACCESSION') != row.spectrum_id: problems.append('accession')
            if license_value != row.license or license_value not in ['CC0', 'CC BY']: problems.append('license')
            if not one('AUTHORS') or one('AUTHORS') != row.authors: problems.append('authors')
            keys = [value.split(' ', 1)[1][:14] for value in fields.get('CH$LINK', []) if value.startswith('INCHIKEY ')]
            if keys != [row.inchikey14]: problems.append('structure_key')
            focused = dict(value.split(' ', 1) for value in fields.get('MS$FOCUSED_ION', []) if ' ' in value)
            if normalize_adduct(focused.get('PRECURSOR_TYPE')) != row.normalized_adduct: problems.append('adduct')
            try:
                if abs(float(focused['PRECURSOR_M/Z']) - float(row.precursor_mz)) > 1e-6: problems.append('precursor')
            except (KeyError, ValueError): problems.append('precursor')
            keep = np.isfinite(peaks).all(1) & (peaks[:, 0] > 0) & (peaks[:, 1] > 0)
            peaks = peaks[keep]
            chosen = np.argsort(-peaks[:, 1], kind='stable')[:128]
            chosen = chosen[np.argsort(peaks[chosen, 0], kind='stable')]
            mz, intensity = np.zeros(128, np.float32), np.zeros(128, np.float32)
            mz[:len(chosen)] = peaks[chosen, 0]; intensity[:len(chosen)] = peaks[chosen, 1]
            if not np.array_equal(mz, frozen_mz[pos]): problems.append('mz_peaks')
            if not np.array_equal(intensity, frozen_in[pos]): problems.append('intensity_peaks')
            if problems:
                issues.append(dict(spectrum_id=row.spectrum_id, issue=','.join(problems)))
                for problem in problems: counts['mismatch_' + problem] += 1
            else: counts['exact_checked_records'] += 1
            if (pos + 1) % 5000 == 0: print('C10 PRIMARY CHECK', pos + 1, '/', len(frame), flush=True)
    pd.DataFrame(issues, columns=['spectrum_id', 'issue', 'found']).to_csv(OUT / 'issues.csv', index=False)
    optional_missing = {name: int((frame[name].isna() | frame[name].astype(str).str.strip().isin(['', 'None', 'nan'])).sum()) for name in ['publication', 'copyright']}
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(OUT / 'protocol.json'),
        source_doi='10.5281/zenodo.19073053', official_release='2026.03', archive_sha256=sha(archive),
        archive_md5=MD5, counts=dict(counts), original_record_licenses=dict(licenses),
        exact_all_records=counts['exact_checked_records'] == len(frame), optional_mirror_fields_missing=optional_missing,
        issue_rows=len(issues), issue_csv_sha256=sha(OUT / 'issues.csv'),
        query_labels_used=False, prediction_metrics_read=False, reference_pool_changed=False,
        source_review_complete=False,
        limitation='Record-level provenance/integrity comparison, not blanket legal clearance. Parent Takumu library attribution remains incomplete; mirror compilation terms remain retained. Primary mismatches require review before any expanded-reference deployment.')
    write(OUT / 'audit.json', report)
    print('DONE C10 PRIMARY REFERENCE AUDIT', json.dumps(report), flush=True)


if __name__ == '__main__': main()
