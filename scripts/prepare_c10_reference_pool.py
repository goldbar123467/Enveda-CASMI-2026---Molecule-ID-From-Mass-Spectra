"""Prepare an attributed reference extension without using query labels/metrics."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from precision_cache import ROOT, sha, write
from build_external_cache import OFFSETS, normalize_adduct, forbidden_library
from training_monitor import system_memory

OUT = ROOT / 'outputs/local/c10-reference-pool-20261110'


def main():
    pa.set_cpu_count(2)
    assert rdBase.rdkitVersion == '2026.03.3' and system_memory()['free_gib'] >= 4
    assert not (OUT / 'reference-manifest.json').exists(), 'Completed reference pool is sealed'
    assert not (ROOT / 'outputs/local/c10-candidates-20261110/development.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    config = ROOT / 'configs/c10-reference-candidates-20261110.json'
    source = ROOT / 'outputs/research/2026-11-09-silver/massbank-coverage-audit/spectra.parquet'
    assert sha(source) == '6ebb46b32ddd344d988c3bdb90e7160d4ce46e6d0b900a1217607aa2234c9477'
    parent = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    pins = json.loads((ROOT / 'outputs/local/c09-external-reference-20261109/development.json').read_text())['reference_sha256']
    for name, expected in pins.items(): assert sha(parent / name) == expected
    write(OUT / 'protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        candidate_config_sha256=sha(config), preparer_sha256=sha(__file__), source_sha256=sha(source),
        parent_reference_sha256=pins, query_labels_used=False, development_metrics_read=False,
        compilation_license='CC-BY-NC-SA-4.0; record license/attribution retained individually; not a CC0-only compilation'))
    frame = pd.read_parquet(source, filters=[('license', 'in', ['CC0', 'CC BY']), ('ion_mode', '=', 'POSITIVE'), ('ms_type', '=', 'MS2')])
    counts = collections.Counter(input_filtered_records=len(frame)); records, mz_rows, in_rows = [], [], []
    for row in frame.itertuples(index=False):
        if forbidden_library(str(row.source) + ' ' + str(row.source_institution)):
            counts['forbidden_source'] += 1; continue
        adduct = normalize_adduct(row.adduct)
        if adduct not in OFFSETS or not adduct.endswith('+'):
            counts['unsupported_adduct'] += 1; continue
        if not np.isfinite(row.precursor_mz) or row.precursor_mz <= 0:
            counts['invalid_precursor'] += 1; continue
        mol = Chem.MolFromSmiles(str(row.smiles))
        if mol is None or Chem.MolToInchiKey(mol)[:14] != str(row.inchikey)[:14]:
            counts['invalid_structure_or_key'] += 1; continue
        charge, offset = OFFSETS[adduct]
        expected = (rdMolDescriptors.CalcExactMolWt(mol) + offset) / charge
        if abs(row.precursor_mz - expected) / expected * 1e6 > 25:
            counts['mass_inconsistent'] += 1; continue
        mz, intensity = np.asarray(row.mzs, np.float32), np.asarray(row.intensities, np.float32)
        if mz.ndim != 1 or mz.shape != intensity.shape:
            counts['invalid_peak_shape'] += 1; continue
        keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
        mz, intensity = mz[keep], intensity[keep]
        if len(mz) < 4:
            counts['too_few_peaks'] += 1; continue
        chosen = np.argsort(-intensity, kind='stable')[:128]
        chosen = chosen[np.argsort(mz[chosen], kind='stable')]
        a, b = np.zeros(128, np.float32), np.zeros(128, np.float32)
        a[:len(chosen)], b[:len(chosen)] = mz[chosen], intensity[chosen]
        item = row._asdict(); item.pop('mzs'); item.pop('intensities')
        item.update(inchikey14=str(row.inchikey)[:14], normalized_smiles=Chem.MolToSmiles(mol),
            ionization_mode='positive', normalized_adduct=adduct, reference_pool='massbank_curated',
            parent_compilation_license='CC-BY-NC-SA-4.0')
        records.append(item); mz_rows.append(a); in_rows.append(b)
    supplementary = pd.DataFrame(records); assert len(supplementary) > 0
    supplement = OUT / 'supplement'; supplement.mkdir(exist_ok=True)
    supplementary.to_parquet(supplement / 'ext_meta.parquet', index=False)
    np.save(supplement / 'ext_mz.npy', np.asarray(mz_rows)); np.save(supplement / 'ext_in.npy', np.asarray(in_rows))
    original = pd.read_parquet(parent / 'ext_meta.parquet')
    original['reference_pool'] = 'parent'
    original['normalized_adduct'] = original.adduct.map(normalize_adduct)
    combined = pd.concat([original, supplementary], ignore_index=True)
    expanded = OUT / 'expanded'; expanded.mkdir(exist_ok=True)
    combined.to_parquet(expanded / 'ext_meta.parquet', index=False)
    for filename, extra in [('ext_mz.npy', mz_rows), ('ext_in.npy', in_rows)]:
        values = np.load(parent / filename, mmap_mode='r')
        assert values.shape == (len(original), 128)
        output = np.lib.format.open_memmap(expanded / filename, mode='w+', dtype=np.float32, shape=(len(combined), 128))
        output[:len(original)] = values
        output[len(original):] = np.asarray(extra)
        output.flush(); del output
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), parent_records=len(original),
        supplemental_records=len(supplementary), expanded_records=len(combined), filters=dict(counts),
        supplemental_licenses=supplementary.license.value_counts().to_dict(),
        supplemental_adducts=supplementary.normalized_adduct.value_counts().to_dict(),
        supplemental_connectivity_keys=int(supplementary.inchikey14.nunique()),
        added_connectivity_keys=int((~supplementary.inchikey14.isin(set(original.inchikey14))).groupby(supplementary.inchikey14).any().sum()),
        query_labels_used=False, prediction_metrics_read=False, p16_deployment_changed=False,
        candidate_config_sha256=sha(config), protocol_sha256=sha(OUT / 'protocol.json'),
        expanded_sha256={name: sha(expanded / name) for name in ['ext_meta.parquet', 'ext_mz.npy', 'ext_in.npy']},
        original_parent_reference_sha256=pins,
        limitation='License labels and contributor attribution retained, not a blanket rights/provenance guarantee. Parent reference metadata lacks record-level attribution. Coverage counts are not validation gains.')
    write(OUT / 'reference-manifest.json', report)
    (OUT / 'README.md').write_text('Private research reference extension.\n\nSources: Takumu Hata casmi26-extlib (declaredCC0 dataset); samartalwar CASMI2026 MassBank harmonized compilation (declaredCC-BY-NC-SA-4.0). Selected MassBank rows retain their literalCC0/CC BY licenses plus authors, publication, copyright and source institution. Original MassBank contributors: https://github.com/MassBank/MassBank-data . No query labels or metrics used. Do not describe this mixed compilation as CC0-only.\n', encoding='utf-8')
    print('DONE C10 REFERENCE PREPARATION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
