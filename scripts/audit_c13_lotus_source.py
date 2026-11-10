"""Trace the public LOTUS mirror to checksum-pinned original plain data.

No query data, labels, predictions, model execution or public upload. Original
Zenodo v11 is CC BY4.0 despite the Kaggle mirror's CC0 claim. Keep attribution.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import collections
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from precision_cache import ROOT, sha, write
from training_monitor import system_memory

SOURCE = ROOT / 'outputs/research/c12-lotus-20261110'
OUT = ROOT / 'outputs/local/c13-lotus-source-20261110'


def main():
    pa.set_cpu_count(2)
    assert rdBase.rdkitVersion == '2026.03.3' and system_memory()['free_gib'] >= 4
    assert not (OUT / 'audit.json').exists(), 'Completed source audit is sealed'
    OUT.mkdir(exist_ok=True)
    mirror = SOURCE / 'candidates_lotus.parquet'
    original = SOURCE / '260413_frozen_metadata.csv.gz'
    metadata = json.loads((SOURCE / 'zenodo-19360665.json').read_text())
    assert metadata['metadata']['license']['id'] == 'cc-by-4.0'
    assert sha(mirror) == '7684b51f72a2d4c073a8d2435fff14cce0a0e7364ac7904101b8f93990f0ac2f'
    assert sha(original) == '09885dcfb3bc0e6d700fefcd5778e91446a1f9d5d2fbbcbf00eb5d2e83b68844'
    assert hashlib.md5(original.read_bytes()).hexdigest() == 'b17048b3b77daae9ab1e480b6591aabd'
    write(OUT / 'protocol.json', dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), mirror_sha256=sha(mirror), original_sha256=sha(original),
        metadata_sha256=sha(SOURCE / 'zenodo-19360665.json'), original_license='CC-BY-4.0',
        mirror_declared_license='CC0-1.0', query_data_read=False, prediction_metrics_computed=False,
        method='Independent RDKit largest-fragment/uncharger/stereo removal; exact cleaned SMILES lookup, then raw key/formula/mass comparison. Untraced rows remain unresolved; do not guess.'))
    frame = pd.read_parquet(mirror)
    assert len(frame) == 149983 and frame.smiles.is_unique and frame.inchikey14.is_unique
    lookup = dict(zip(frame.smiles, range(len(frame))))
    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    seen = {}; traced = {}; counts = collections.Counter()
    columns = ['structure_smiles_2D', 'structure_inchikey', 'structure_wikidata', 'reference_doi']
    for chunk in pd.read_csv(original, usecols=columns, dtype=str, keep_default_na=False,
                             chunksize=20000):
        for row in chunk.itertuples(index=False, name=None):
            # Pandas preserves source column order, so use the explicit mapping.
            item = dict(zip(chunk.columns, row)); text = item['structure_smiles_2D']
            if text not in seen:
                mol = Chem.MolFromSmiles(text)
                if mol is None:
                    counts['invalid_original_smiles'] += 1; seen[text] = None
                else:
                    Chem.RemoveStereochemistry(mol)
                    mol = uncharger.uncharge(chooser.choose(mol))
                    clean = Chem.MolToSmiles(mol)
                    idx = lookup.get(clean)
                    seen[text] = idx
                    if idx is not None and idx not in traced:
                        ref = frame.iloc[idx]
                        key = Chem.MolToInchiKey(mol)[:14]
                        mass = rdMolDescriptors.CalcExactMolWt(mol)
                        formula = rdMolDescriptors.CalcMolFormula(mol)
                        assert key == ref.inchikey14 and formula == ref.formula
                        assert abs(mass - ref.monoisotopic_mass) < 1e-7
            idx = seen[text]
            if idx is not None:
                counts['traceable_original_rows'] += 1
                if idx not in traced:
                    traced[idx] = dict(mirror_row=int(idx), smiles=frame.smiles.iloc[idx],
                        original_smiles=text, original_inchikey=item['structure_inchikey'],
                        structure_wikidata=item['structure_wikidata'], reference_doi=item['reference_doi'],
                        original_record='https://zenodo.org/records/19360665', original_license='CC-BY-4.0')
        counts['original_rows'] += len(chunk)
        assert system_memory()['free_gib'] >= 4
        print('C13 ORIGINAL LOTUS ROWS', counts['original_rows'], 'TRACED MIRROR', len(traced), flush=True)
    rows = pd.DataFrame(list(traced.values())).sort_values('mirror_row', kind='stable')
    rows.to_parquet(OUT / 'original-attribution.parquet', index=False)
    missing = sorted(set(range(len(frame))) - set(traced))
    frame.iloc[missing].assign(mirror_row=missing).to_parquet(OUT / 'unresolved.parquet', index=False)
    write(OUT / 'audit.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        mirror_rows=len(frame), traced_mirror_rows=len(rows), unresolved_mirror_rows=len(missing),
        unique_original_smiles=len(seen), counts=dict(counts), protocol_sha256=sha(OUT / 'protocol.json'),
        attribution_sha256=sha(OUT / 'original-attribution.parquet'), unresolved_sha256=sha(OUT / 'unresolved.parquet'),
        original_license='CC-BY-4.0', mirror_declared_license='CC0-1.0', license_claim_conflict=True,
        original_citation=metadata['metadata'].get('creators'),
        identity_caveat='Mirror raw connectivity keys do not implement official tautomer canonicalization; compute official metric identities before candidate deduplication.',
        query_data_read=False, query_labels_read=False, prediction_metrics_computed=False,
        candidate_recipe_selected=False, uploaded=False,
        limitation='Exact traced structural provenance and original license metadata only. No chemical truth, training novelty, model gain or hidden-score claim. Unresolved structures are not silently adopted.'))
    print('DONE C13 LOTUS SOURCE AUDIT', len(rows), 'UNRESOLVED', len(missing), flush=True)


if __name__ == '__main__': main()
