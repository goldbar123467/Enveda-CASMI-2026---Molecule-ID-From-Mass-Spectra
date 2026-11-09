"""Fixed-policy external spectral matching; label-blind search and alias audit.

Idea/threshold attribution: alexchilton/casmi26-lb-0-484-peaks-to-structures-extlib
and Takumu Hata's public measured-reference library. No downloaded code runs.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from c09_spectrum_audit import signatures


def prepare(mz, intensity, precursor=None):
    mz, intensity = np.asarray(mz, dtype=np.float64), np.asarray(intensity, dtype=np.float64)
    assert mz.shape == intensity.shape and mz.ndim == 1
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    if precursor is not None:
        keep &= mz <= precursor + .5
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz, kind='stable')
    intensity = np.sqrt(intensity[order])
    return mz[order], intensity / max(float(np.linalg.norm(intensity)), 1e-12)


def cosine(query, reference, tolerance=.015):
    """Greedy maximum-product one-to-one peak assignment, with stable ties."""
    qm, qi = query
    rm, ri = reference
    if not len(qm) or not len(rm):
        return 0., 0
    q, r = np.nonzero(np.abs(qm[:, None] - rm[None, :]) <= tolerance)
    products = qi[q] * ri[r]
    order = np.lexsort((-r, -q, -products))
    used_q, used_r, score, count = set(), set(), 0., 0
    for index in order:
        a, b = int(q[index]), int(r[index])
        if a not in used_q and b not in used_r:
            used_q.add(a); used_r.add(b)
            score += float(products[index]); count += 1
    return min(score, 1.), count


def search(queries, reference_dir, features, policy):
    """Compute both views without labels, including protected pipeline queries."""
    reference_dir = Path(reference_dir)
    meta = pd.read_parquet(reference_dir / 'ext_meta.parquet')
    mz = np.load(reference_dir / 'ext_mz.npy', mmap_mode='r')
    intensity = np.load(reference_dir / 'ext_in.npy', mmap_mode='r')
    assert mz.shape == intensity.shape == (len(meta), 128)
    assert not {'key', 'smiles', 'inchikey14'} & set(queries.columns)
    threshold = policy['thresholds']
    precursor = meta.precursor_mz.to_numpy(dtype=np.float64)
    order = np.argsort(precursor, kind='stable'); sorted_precursor = precursor[order]
    all_signatures, wanted, windows = set(), set(), []
    for row in queries.itertuples(index=False):
        sig = signatures(row, features)
        if sig:
            all_signatures.update(sig)
        width = row.precursor_mz * threshold['precursor_ppm'] * 1e-6 + threshold['precursor_absolute_da']
        lo, hi = np.searchsorted(sorted_precursor, [row.precursor_mz - width, row.precursor_mz + width], side='left')
        # Include the exact upper boundary, consistently with the public rule.
        hi = np.searchsorted(sorted_precursor, row.precursor_mz + width, side='right')
        positions = order[lo:hi]
        windows.append((row, positions)); wanted.update(positions.tolist())
    refs, aliases = {}, set()
    for pos in sorted(wanted):
        refs[pos] = prepare(mz[pos], intensity[pos])
        row = SimpleNamespace(ms2_mzs=mz[pos], ms2_normalized_intensities=intensity[pos])
        sig = signatures(row, features)
        if sig and any(value in all_signatures for value in sig):
            aliases.add(pos)
    matches = {str(mid): {view: dict(cosine=0., matched_peaks=0, smiles='', reference_row=None)
                        for view in ['all_reference', 'input_alias_excluded']}
               for mid in queries.molecule_id.unique()}
    comparisons = 0
    for row, positions in windows:
        query = prepare(features.array_value(row.ms2_mzs), features.array_value(row.ms2_normalized_intensities), row.precursor_mz)
        positive = 'pos' in str(row.ionization_mode).lower()
        for pos in positions:
            ref = meta.iloc[pos]
            if ('pos' in str(ref.ionization_mode).lower()) != positive:
                continue
            score, count = cosine(query, refs[pos], threshold['fragment_tolerance_da']); comparisons += 1
            if count < threshold['minimum_matched_peaks']:
                continue
            for view in ['all_reference', 'input_alias_excluded']:
                if view == 'input_alias_excluded' and pos in aliases:
                    continue
                best = matches[str(row.molecule_id)][view]
                if score > best['cosine']:
                    best.update(cosine=score, matched_peaks=count, smiles=str(ref.normalized_smiles), reference_row=int(pos),
                                input_alias=pos in aliases, same_adduct=str(row.adduct) == str(ref.adduct))
    evidence = [dict(molecule_id=mid, view=view, **best) for mid, views in matches.items() for view, best in views.items()]
    report = dict(queries=int(queries.molecule_id.nunique()), spectra=len(queries), reference_rows=len(meta),
                  precursor_candidate_rows=len(wanted), query_reference_comparisons=comparisons,
                  query_input_alias_reference_rows=len(aliases), query_labels_used=False,
                  limitation='Rounded full peaks and FP16top96 signatures only; not universal acquisition/provenance exclusion')
    return pd.DataFrame(evidence), report


def apply_overlay(baseline, matches, library_confidence, policy, identity):
    """Promote a measured hit with official-identity dedup and exact fallback."""
    assert baseline.molecule_id.is_unique and matches.molecule_id.is_unique
    lookup = matches.set_index('molecule_id').to_dict('index')
    threshold = policy['thresholds']; result = baseline.copy(); records = []
    for index, row in baseline.iterrows():
        mid, original = str(row.molecule_id), row.smiles
        match = lookup.get(mid)
        reason, changed = 'no_qualifying_match', False
        confidence = library_confidence.get(mid)
        if confidence is None or not np.isfinite(confidence):
            reason = 'missing_pipeline_library_confidence'
        elif confidence >= threshold['protect_pipeline_library_at']:
            reason = 'protected_pipeline_library'
        elif match and match['cosine'] >= threshold['sqrt_cosine'] and match['matched_peaks'] >= threshold['minimum_matched_peaks'] and match['smiles']:
            candidates = original.split(';'); keys = [identity(s) for s in candidates]
            assert len(keys) == len(set(keys)) and len(keys) <= 25
            try:
                key = identity(match['smiles'])
            except (ValueError, TypeError):
                records.append(dict(molecule_id=mid, reason='invalid_external_structure', changed=False))
                continue
            if key != keys[0]:
                representative = candidates[keys.index(key)] if key in keys else match['smiles']
                candidates = [representative] + [s for s, k in zip(candidates, keys) if k != key]
                result.at[index, 'smiles'] = ';'.join(candidates[:25]); changed = True
                reason = 'promoted_existing_identity' if key in keys else 'inserted_external_identity'
            else:
                reason = 'same_identity_as_current_leader'
        records.append(dict(molecule_id=mid, reason=reason, changed=changed))
    return result, pd.DataFrame(records)
