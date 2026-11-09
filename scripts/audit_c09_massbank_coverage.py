"""Inventory a backup reference source without changing the sealed C09E rule."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pa.set_cpu_count(2)
    folder = ROOT / 'outputs/research/2026-11-09-silver/massbank-coverage-audit'
    path = folder / 'spectra.parquet'
    extpath = ROOT / 'outputs/research/2026-11-09-silver/extlib/ext_meta.parquet'
    frame = pd.read_parquet(path, columns=['spectrum_id', 'license', 'inchikey',
        'source_institution', 'ion_mode', 'ms_type', 'precursor_mz', 'num_peaks'])
    known = set(pd.read_parquet(extpath, columns=['inchikey14']).inchikey14.dropna().astype(str))
    # Literal observed record metadata only; the compilation has a separate
    # CC-BY-NC-SA-4.0 declaration. This is an inventory, not a license clearance.
    selected = frame[frame.license.eq('CC0') & frame.ion_mode.eq('POSITIVE') &
                     frame.ms_type.eq('MS2') & frame.num_peaks.ge(4) &
                     np.isfinite(frame.precursor_mz) & frame.precursor_mz.gt(0)].copy()
    selected['key14'] = selected.inchikey.str.slice(0, 14)
    selected = selected[selected.key14.str.fullmatch('[A-Z]{14}', na=False)]
    novel = selected[~selected.key14.isin(known)]
    report = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        dataset='samartalwar/casmi-2026-spectral-library-massbankharmonized',
        source_url='https://www.kaggle.com/datasets/samartalwar/casmi-2026-spectral-library-massbankharmonized',
        dataset_declared_license='CC-BY-NC-SA-4.0',
        source_sha256=sha(path), current_reference_metadata_sha256=sha(extpath),
        records=len(frame), record_license_counts=frame.license.fillna('<missing>').value_counts().to_dict(),
        selected_definition='Literal CC0 record license, positive MS2, >=4 peaks, finite positive precursor and valid raw connectivity key',
        selected_records=len(selected), selected_connectivity_keys=int(selected.key14.nunique()),
        selected_institutions=selected.source_institution.value_counts().to_dict(),
        absent_from_current_reference_records=len(novel),
        absent_from_current_reference_connectivity_keys=int(novel.key14.nunique()),
        query_labels_used=False, candidate_policy_changed=False, reference_data_adopted=False,
        reporter_sha256=sha(Path(__file__)),
        limitation='Raw connectivity-key coverage only; not official tautomer identity, spectral novelty, validation gain or clearance of the derivative compilation license. No new data is attached to P16.')
    (folder / 'coverage.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('DONE C09 MASSBANK COVERAGE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
