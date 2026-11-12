"""Fetch and audit actual unlabeled production inputs for a bounded preview."""
import argparse
import zipfile
from c75_port_utils import ROOT, read, write, sha, stamp

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'production-inputs'


def fetch():
    assert not OUT.exists()
    assert read(ROOT / '.local/c38-active-goal.json')['active']
    OUT.mkdir()
    competition = 'enveda-CASMI26-molecule-id-mass-spectra'
    names = ['test.parquet', 'sample_submission.csv']
    write(OUT / 'download-intent.json', dict(created_utc=stamp(), source_sha256=sha(__file__),
        official_competition=competition, filenames=names, read_only=True, scoring_entries=0))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    for name in names:
        api.competition_download_file(competition, name, path=str(OUT.resolve()), quiet=True)
        if not (OUT / name).exists():
            archive = OUT / (name + '.zip'); assert archive.is_file()
            with zipfile.ZipFile(archive) as package:
                assert package.namelist() == [name], 'Do not unpack unexpected paths'
                (OUT / name).write_bytes(package.read(name))
        assert (OUT / name).is_file()
    write(OUT / 'download-complete.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        official_competition=competition, files_sha256={name: sha(OUT / name) for name in names},
        unlabeled_inputs_only=True, C13_read=False, scoring_entries=0))
    print('C76 ACTUAL UNLABELED PRODUCTION TEST/SAMPLE DOWNLOADED', flush=True)


def audit():
    import torch
    import numpy as np
    import pandas as pd
    target = OUT / 'input-audit.json'; assert not target.exists()
    done = read(OUT / 'download-complete.json')
    assert done['source_sha256'] == sha(__file__)
    for name, pin in done['files_sha256'].items(): assert sha(OUT / name) == pin, name
    q = pd.read_parquet(OUT / 'test.parquet')
    sample = pd.read_csv(OUT / 'sample_submission.csv', dtype=str, keep_default_na=False)
    assert len(sample) == sample.molecule_id.nunique() == q.molecule_id.nunique() == 400
    assert set(sample.molecule_id) == set(q.molecule_id) and q.spectrum_id.is_unique
    assert not {'smiles', 'normalized_smiles', 'key', 'inchikey14', 'formula'} & set(q.columns)
    assert np.isfinite(q.precursor_mz).all() and q.precursor_mz.gt(0).all()
    prior = ROOT / 'outputs/local/c37-selection-trace-20261111/unlabelled-target/test.parquet'
    assert sha(OUT / 'test.parquet') == sha(prior), 'Reconcile any change in official public test inputs'
    modes = q.groupby('molecule_id').ionization_mode.agg(lambda v: 'mixed' if v.nunique() > 1 else v.iloc[0]).value_counts().to_dict()
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), queries=400,
        acquisitions=len(q), actual_query_modes=modes, official_API_download_verified=True,
        prior_unlabeled_test_bytes_exact=True, unique_original_spectra_and_sample_ids=True,
        all_original_acquisitions_retained=True, no_query_structures_in_test=True,
        download_complete_sha256=sha(OUT / 'download-complete.json'),
        potential_use='Bounded original production preview only; cannot replace actual T4 source/production qualification',
        no_inference_yet=True, quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 ACTUAL PRODUCTION INPUT AUDIT PASSED', len(q), 'acquisitions/400 queries', modes, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['fetch', 'audit']); args = parser.parse_args()
    (fetch if args.phase == 'fetch' else audit)()
