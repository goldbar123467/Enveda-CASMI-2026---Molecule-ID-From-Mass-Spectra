"""Stream the disclosed MassBank/GNPS mirrors into a small, attributed training cache.

The mirrors are an intake source, not a blanket provenance guarantee. NIST-named
libraries, imported/propagated annotations, invalid structures, unsupported adducts,
and mass-inconsistent records are rejected before any training file is written.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'outputs/local/kit'))
import features as F

SOURCES = {
    'massbank': ('https://scigantic-massbank.s3.amazonaws.com/massbank_open_spectra.parquet',
                 '8fbda467165353851eb112054f19b9e6', 'https://github.com/MassBank/MassBank-data/releases/tag/2026.03'),
    'gnps': ('https://scigantic-gnps-libraries.s3.amazonaws.com/gnps_community_spectra.parquet',
             '22d30be090f0b088082f5fa23de0c06e', 'https://external.gnps2.org/gnpslibrary'),
}
OFFSETS = {'[M+H]+': (1, 1.007276), '[M-H]-': (1, -1.007276),
           '[M+Na]+': (1, 22.989218), '[M+K]+': (1, 38.963158),
           '[M+NH4]+': (1, 18.033823), '[M+Cl]-': (1, 34.969402),
           '[M+FA-H]-': (1, 44.998201), '[M+H-H2O]+': (1, -17.003289),
           '[M+2H]2+': (2, 2.014552), '[M-2H]2-': (2, -2.014552)}


def forbidden_library(name):
    return bool(re.search(r'NIST|METLIN|PROP[AO]GAT|SUSPECT|DEREPLICATOR', str(name), re.I))


def collision_energy(value):
    # A percent or NCE value is not an energy in eV. Keep it unknown.
    text = str(value or '').strip()
    if re.fullmatch(r'\d+(\.\d+)?\s*(eV)?', text, re.I):
        return [float(re.sub(r'\s*eV$', '', text, flags=re.I))]
    return []


def normalize_adduct(value):
    text = str(value or '').replace(' ', '')
    if text in OFFSETS:
        return text
    aliases = {k[1:k.index(']')].upper(): k for k in OFFSETS if OFFSETS[k][0] == 1}
    aliases.update({k.upper().replace(']1', ']'): k for k in OFFSETS})
    return aliases.get(text.upper().replace(']1', ']'), text)


def digest_file(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, default=ROOT/'outputs/external/2026-11-06')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/external/cache-20261106')
    args = ap.parse_args()
    out = args.out
    assert not (out/'manifest.json').exists(), 'Completed cache exists; do not overwrite it'
    out.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog('rdApp.*')
    counts, libraries, licenses = Counter(), Counter(), Counter()
    molecules, indices, fingerprints, structure_cache = [], {}, [], {}
    records, seen = [], set()
    started = time.time()
    with (out/'peaks.bin').open('wb') as peaks_file, (out/'metadata.bin').open('wb') as meta_file:
        for source in SOURCES:
            path = args.raw/(source+'.parquet')
            assert digest_file(path, 'md5') == SOURCES[source][1], f'{source} download identity mismatch'
            pf = pq.ParquetFile(path)
            print('SOURCE', source, pf.metadata.num_rows, pf.schema_arrow.names, flush=True)
            for batch in pf.iter_batches(batch_size=1024, use_threads=False):
                for row in batch.to_pylist():
                    counts[source+'_input'] += 1
                    library = 'MassBank' if source == 'massbank' else str(row.get('library', row.get('source_library', '')))
                    if not library or forbidden_library(library):
                        counts['rejected_library:'+library] += 1
                        continue
                    if source == 'massbank':
                        license_name = str(row.get('license', ''))
                        if license_name not in {'CC0 1.0', 'CC BY 4.0', 'CC BY-SA 4.0', 'dl-de/by-2-0'}:
                            counts['rejected_license'] += 1
                            continue
                        if row.get('spectrum_type') != 'MS2':
                            counts['rejected_ms_level'] += 1
                            continue
                    else:
                        license_name = 'CC0 (direct GNPS contribution; see mirror/upstream manifests)'
                        if row.get('ms_level') != 2 or row.get('library_class') not in {1, 2}:
                            counts['rejected_ms_level_or_annotation_quality'] += 1
                            continue
                    smi = str(row.get('smiles') or '')
                    if smi not in structure_cache:
                        try:
                            structure_cache[smi] = F.structure(smi) if smi and '.' not in smi else None
                        except Exception:
                            structure_cache[smi] = None
                    structure = structure_cache[smi]
                    if structure is None:
                        counts['rejected_structure'] += 1
                        continue
                    key, canonical, formula, mass, fp = structure
                    adduct = normalize_adduct(row.get('precursor_type', row.get('adduct', '')))
                    if adduct not in OFFSETS:
                        counts['rejected_adduct'] += 1
                        continue
                    precursor = row.get('precursor_mz')
                    try:
                        precursor = float(precursor)
                        charge, offset = OFFSETS[adduct]
                        if not math.isfinite(precursor) or abs(precursor*charge-offset-mass) > max(.03, mass*50e-6):
                            raise ValueError('mass')
                        x, length = F.peaks(row['mz'], row['intensity'], precursor)
                        if length < 3:
                            raise ValueError('short')
                        meta, adduct_id = F.metadata(precursor, adduct, row.get('ion_mode', ''), collision_energy(row.get('collision_energy')))
                    except (ValueError, TypeError, KeyError):
                        counts['rejected_spectrum_or_mass'] += 1
                        continue
                    # Rounded cache features match the existing kit representation.
                    x = x.astype('<f2')
                    signature = hashlib.sha256(key.encode()+x.tobytes()+meta.astype('<f4').tobytes()+bytes([adduct_id])).digest()
                    if signature in seen:
                        counts['duplicate_external_spectrum'] += 1
                        continue
                    seen.add(signature)
                    if key not in indices:
                        indices[key] = len(molecules)
                        molecules.append(dict(key=key, smiles=canonical, formula=formula, mass=mass))
                        fingerprints.append(fp)
                    x.tofile(peaks_file)
                    meta.astype('<f4').tofile(meta_file)
                    records.append(dict(mol_idx=indices[key], length=length, adduct_id=adduct_id,
                                        source=1 if source == 'massbank' else 2, library=library,
                                        accession=str(row.get('accession', row.get('spectrum_id', ''))), license=license_name))
                    counts[source+'_accepted'] += 1
                    libraries[library] += 1
                    licenses[license_name] += 1
                if counts[source+'_input'] % 10240 == 0:
                    print('PREP', dict(counts), 'seconds', round(time.time()-started), flush=True)
    assert len(molecules) > 1000 and len(records) > 10000, 'Insufficient approved external spectra'
    pd.DataFrame(molecules).to_parquet(out/'molecules.parquet', index=False)
    pd.DataFrame(records).to_parquet(out/'spectra.parquet', index=False)
    np.save(out/'fingerprints.npy', np.asarray(fingerprints, dtype=np.uint8))
    manifest = dict(sources={k:dict(url=v[0],md5=v[1],upstream=v[2]) for k,v in SOURCES.items()},
                    counts=dict(counts), libraries=dict(libraries), licenses=dict(licenses),
                    molecules=len(molecules), spectra=len(records), seconds=round(time.time()-started),
                    feature_source_sha256=digest_file(ROOT/'outputs/local/kit/features.py'),
                    builder_sha256=digest_file(__file__), no_test_data=True,
                    exclusions='NIST/METLIN/propagated/suspect/DEREPLICATOR libraries; salts; invalid labels; unsupported adducts; mass > max(0.03Da,50ppm); <3peaks',
                    files={p.name:digest_file(p) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DONE external cache', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
