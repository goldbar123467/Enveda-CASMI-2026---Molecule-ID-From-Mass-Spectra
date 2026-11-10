"""Prepare a prospective, label-separated CASMI2022 panel; never infer/submit.

Raw source: Young/Wang/Roest, Zenodo8371051 (CC-BY-4.0), originally UC Davis
CASMI2022. Strict .005Da/.1min, maximum-TIC scan assignment follows the public
MassFormer preparation method, with no arbitrary fallback for unmatched targets.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase

from precision_cache import ROOT, sha, write, features
from casmi2022_data import parse_msp, scan_rank
from c09_spectrum_audit import signatures
from training_monitor import system_memory

SOURCE = ROOT / 'outputs/research/c10-casmi2022-20261110'
OUT = ROOT / 'outputs/local/c10-casmi2022-20261110'
PUBLISH = ROOT / 'outputs/publish/c10-casmi2022-20261110'
PARAMETERS = dict(positive_only=True, excluded_known_errors=[81, 282, 432, 476],
    mass_assignment_da=.005, rt_assignment_minutes=.1, selection='maximum positive finite total intensity; deterministic RT/mass/name ties',
    minimum_usable_peaks=3, raw_signature_precursor_window='20ppm +.003Da',
    draw_salt='C10-CASMI2022-prospective-v1:', maximum_queries=160, development=40,
    minimum_confirmation=80, unmatched_target_fallback=False, unknown_collision_energy='NaN, not imputed')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def raw_overlap(queries, reference):
    """Same bounded label-blind signatures as C09, before panel selection."""
    precursor = pd.read_parquet(reference, columns=['precursor_mz']).precursor_mz.to_numpy(dtype=np.float64)
    order = np.argsort(precursor, kind='stable'); sorted_mz = precursor[order]
    wanted, indices = set(), [collections.defaultdict(set), collections.defaultdict(set)]
    for row in queries.itertuples(index=False):
        sig = signatures(row, features); assert sig is not None
        for i, value in enumerate(sig): indices[i][value].add(row.molecule_id)
        width = row.precursor_mz * 20e-6 + .003
        lo = np.searchsorted(sorted_mz, row.precursor_mz - width, side='left')
        hi = np.searchsorted(sorted_mz, row.precursor_mz + width, side='right')
        wanted.update(order[lo:hi].tolist())
    cross = set().union(*(ids for index in indices for ids in index.values() if len(ids) > 1))
    matches, checked, offset = [], 0, 0
    columns = ['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
    for batch in pq.ParquetFile(reference).iter_batches(batch_size=4096, columns=columns, use_threads=False):
        positions = [i for i in range(len(batch)) if offset + i in wanted]
        if positions:
            selected = batch.take(pa.array(positions)).to_pandas()
            for pos, row in zip(positions, selected.itertuples(index=False)):
                values = signatures(row, features); checked += 1
                if values is None: continue
                for i, value in enumerate(values):
                    for mid in indices[i].get(value, []):
                        matches.append(dict(molecule_id=mid, reference_row=offset+pos, signature=i))
        offset += len(batch)
        if offset % 262144 < 4096:
            print('C10 RAW AUDIT', offset, '/', len(precursor), 'checked', checked, flush=True)
    flagged = {m['molecule_id'] for m in matches}
    pd.DataFrame(matches, columns=['molecule_id', 'reference_row', 'signature']).to_csv(OUT / 'raw-matches.csv', index=False)
    return flagged | cross, dict(raw_rows=len(precursor), candidate_rows=len(wanted), checked_rows=checked,
        reference_overlap_queries=len(flagged), cross_identity_query_aliases=len(cross), match_records=len(matches),
        labels_used=False, limitation='Bounded rounded/full and FP16top96 signatures, not universal near-duplicate or historical-pretraining exclusion')


def main():
    pa.set_cpu_count(2)
    assert rdBase.rdkitVersion == '2026.03.3' and system_memory()['free_gib'] >= 4
    assert not (PUBLISH / 'validation-contract.json').exists(), 'Completed dataset is sealed'
    OUT.mkdir(parents=True, exist_ok=True); PUBLISH.mkdir(parents=True, exist_ok=True)
    archive = SOURCE / 'casmi_2022.tgz'
    archive_sha = sha(archive)
    assert archive_sha == 'febe375faf786ab65950c50ec8f0c4ee8ea9c8ce6ff10b4cd68f25fcd5ce1e31'
    write(OUT / 'preparation-protocol.json', dict(prepared_utc=stamp(), parameters=PARAMETERS,
        source_sha256=archive_sha, preparer_sha256=sha(__file__), parser_sha256=sha(ROOT / 'scripts/casmi2022_data.py'),
        feature_sha256=sha(ROOT / 'outputs/local/kit/features.py'),
        author_method_url='https://github.com/Roestlab/massformer/blob/master/preproc_scripts/prepare_casmi22_data.py',
        author_method_git_blob='0987cc5c48ea7309661ac7a05cd7df5e1fcf2464',
        no_prediction_metrics_read=True, labels_excluded_from_runtime=True))
    reference = ROOT / 'outputs/research/c08-pipeline/reference/train_structs.parquet'
    assert sha(reference) == 'f20aa1cac23af42e430a6633707fda34e48dea97114bdc3e443897f692df3656'
    raw_train = ROOT / 'outputs/competition-raw/train.parquet'
    forbidden = set(pd.read_parquet(ROOT / 'outputs/local/n0/cache/molecules.parquet', columns=['key']).key)
    for name in ['key', 'inchikey14', 'rawkey']:
        forbidden.update(pd.read_parquet(reference, columns=[name])[name].dropna().astype(str).str[:14])
    forbidden.update(pd.read_parquet(raw_train, columns=['inchikey14']).inchikey14.dropna().astype(str).str[:14])
    for name in ['c08-validation-20261108', 'c09-validation-20261109']:
        forbidden.update(pd.read_csv(ROOT / 'outputs/publish' / name / 'labels.csv', usecols=['key']).key)
    with tarfile.open(archive, 'r:gz') as data:
        meta = pd.read_csv(data.extractfile('casmi_2022/all_meta_results.csv'), header=1)
    meta = meta[meta.File.astype(str).str.contains('_pos', case=False, regex=False)]
    meta = meta[~pd.to_numeric(meta['Compound Number']).isin(PARAMETERS['excluded_known_errors'])]
    targets, counts = {}, collections.Counter(positive_rows_after_known_error_exclusion=len(meta))
    for row in meta.to_dict('records'):
        rawkey = str(row['InChIKey'])[:14]
        if rawkey in forbidden:
            counts['raw_key_excluded'] += 1; continue
        try: key, smiles, formula, mass, _ = features.structure(row['SMILES'])
        except (ValueError, TypeError):
            counts['invalid_structure'] += 1; continue
        if key in forbidden:
            counts['canonical_key_excluded'] += 1; continue
        number = int(row['Compound Number'])
        mid = 'C10Q_' + hashlib.sha256((archive_sha + ':' + str(number)).encode()).hexdigest()[:20]
        targets[number] = dict(molecule_id=mid, key=key, smiles=smiles, formula=formula, mass=mass,
            source='CASMI2022', source_number=number, source_file=str(row['File']), adduct=str(row['Adduct']),
            target_mz=float(row['Precursor m/z (Da)']), target_rt=float(row['RT [min]']))
    by_file = collections.defaultdict(list)
    for number, row in targets.items(): by_file[row['source_file']].append(number)
    best, seen_scans = {}, 0
    with tarfile.open(archive, 'r:gz') as data:
        for member in data:
            if not member.isfile() or not member.name.endswith('.msp'): continue
            name = Path(member.name).stem
            if name not in by_file: continue
            for scan in parse_msp(io.TextIOWrapper(data.extractfile(member), encoding='utf-8')):
                seen_scans += 1
                for number in by_file[name]:
                    row = targets[number]; rank = scan_rank(scan, row['target_mz'], row['target_rt'])
                    if rank is not None and (number not in best or rank < best[number][0]):
                        best[number] = rank, scan
    rows, labels, provenance = [], [], []
    for number, target in targets.items():
        if number not in best:
            counts['no_matching_scan'] += 1; continue
        rank, scan = best[number]
        peaks = np.asarray(scan['peaks'], dtype=np.float64)
        keep = np.isfinite(peaks).all(1) & (peaks[:, 0] > 0) & (peaks[:, 1] > 0)
        peaks = peaks[keep]; peaks = peaks[np.argsort(peaks[:, 0], kind='stable')]
        if len(peaks) < 3:
            counts['too_few_peaks'] += 1; continue
        mz, intensity = peaks[:, 0].astype(np.float32), (peaks[:, 1] / peaks[:, 1].max()).astype(np.float32)
        if features.peaks(mz, intensity, scan['precursor_mz'], 96)[1] < 3:
            counts['too_few_model_peaks'] += 1; continue
        mid = target['molecule_id']
        sid = 'C10S_' + hashlib.sha256((target['source_file'] + ':' + scan['name']).encode()).hexdigest()[:20]
        rows.append(dict(molecule_id=mid, spectrum_id=sid, adduct=target['adduct'], precursor_mz=scan['precursor_mz'],
            ionization_mode='positive', instrument_type='FT', collision_energy_ev=float('nan'),
            ms2_mzs=mz, ms2_normalized_intensities=intensity, num_peaks=len(mz)))
        labels.append({k: target[k] for k in ['molecule_id', 'key', 'smiles', 'formula', 'mass', 'source']})
        provenance.append(dict(molecule_id=mid, spectrum_id=sid, source_number=number,
            source_file=target['source_file'], scan_name=scan['name'], rt=scan['rt'], target_rt=target['target_rt'],
            target_mz=target['target_mz'], observed_mz=scan['precursor_mz'], tic=-rank[0], license='CC-BY-4.0'))
    queries = pd.DataFrame(rows); labels = pd.DataFrame(labels); provenance = pd.DataFrame(provenance)
    # Keep one acquisition per identity without looking at prediction quality.
    ordered = labels.merge(provenance[['molecule_id', 'tic']], on='molecule_id').sort_values(['tic', 'molecule_id'], ascending=[False, True], kind='stable')
    keep_ids = set(ordered.drop_duplicates('key').molecule_id)
    queries = queries[queries.molecule_id.isin(keep_ids)].copy()
    assert not {'key', 'smiles', 'formula', 'mass', 'source_number'} & set(queries.columns)
    flagged, audit = raw_overlap(queries, raw_train)
    labels = labels[labels.molecule_id.isin(keep_ids) & ~labels.molecule_id.isin(flagged)].copy()
    labels['draw'] = labels.key.map(lambda key: hashlib.sha256((PARAMETERS['draw_salt'] + key).encode()).hexdigest())
    labels = labels.sort_values('draw', kind='stable').head(PARAMETERS['maximum_queries']).reset_index(drop=True)
    assert len(labels) >= PARAMETERS['development'] + PARAMETERS['minimum_confirmation'], 'Insufficient independent panel; do not resample'
    labels['partition'] = ['development'] * PARAMETERS['development'] + ['confirmation'] * (len(labels)-PARAMETERS['development'])
    ids = labels.molecule_id.tolist(); order = {mid: i for i, mid in enumerate(ids)}
    queries = queries[queries.molecule_id.isin(ids)].copy(); queries['order'] = queries.molecule_id.map(order)
    queries = queries.sort_values('order').drop(columns='order').reset_index(drop=True)
    assert len(queries) == len(labels) and labels.key.is_unique and queries.spectrum_id.is_unique
    labels.drop(columns='draw').to_csv(OUT / 'labels.csv', index=False)
    provenance[provenance.molecule_id.isin(ids)].to_csv(OUT / 'provenance.csv', index=False)
    queries.to_parquet(PUBLISH / 'validation-queries.parquet', index=False)
    pd.DataFrame(dict(molecule_id=ids, smiles='CCO')).to_csv(PUBLISH / 'validation-sample.csv', index=False)
    write(OUT / 'partitions.json', {name: labels[labels.partition.eq(name)].molecule_id.tolist() for name in ['development', 'confirmation']})
    report = dict(prepared_utc=stamp(), version=1, queries=len(queries), spectra=len(queries), development=PARAMETERS['development'],
        confirmation=len(labels)-PARAMETERS['development'], source='CASMI2022', source_doi='10.5281/zenodo.8371051', declared_license='CC-BY-4.0',
        parameters=PARAMETERS, counts=dict(counts), scans_read=seen_scans, audit=audit,
        raw_training_sha256=sha(raw_train), production_train_structs_sha256=sha(reference),
        prior_panel_labels_sha256={name: sha(ROOT / 'outputs/publish' / name / 'labels.csv') for name in ['c08-validation-20261108', 'c09-validation-20261109']},
        labels_sha256=sha(OUT / 'labels.csv'),
        preparation_protocol_sha256=sha(OUT / 'preparation-protocol.json'),
        query_labels_visible_to_pipeline=False, truth_injection=False, truth_absent_mrr=0,
        limitation='Historical public pretraining exposure not universally excluded. Single selected LC-MS2 scan per identity; FT/HCD metadata follows source authors, collision energy unavailable. No model metrics or recipe selection performed.',
        files={name: sha(PUBLISH/name) for name in ['validation-queries.parquet', 'validation-sample.csv']})
    write(PUBLISH / 'validation-contract.json', report); write(OUT / 'preparation-report.json', report)
    write(PUBLISH / 'dataset-metadata.json', dict(id='clarkkitchen/casmi26-c10-casmi2022-20261110', title='CASMI26 C10 unlabeled CASMI2022 evaluation queries', licenses=[dict(name='CC-BY-4.0')]))
    (PUBLISH / 'README.md').write_text('Private prospective CASMI2022 evaluation queries. No structure labels are packaged.\n\nSource: Young, Wang and Roest, Public Data files for MassFormer, Zenodo8371051, CC-BY-4.0; original UC Davis CASMI2022 workshop data. https://doi.org/10.5281/zenodo.8371051\n\nStrict mass/RT matching, no unmatched fallback. Known-error challenges, training/previous-panel identities, bounded raw-spectrum matches and within-panel aliases excluded before deterministic partition. Labels and partition membership remain local.\n', encoding='utf-8')
    print('DONE C10 CASMI2022 PREPARATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
