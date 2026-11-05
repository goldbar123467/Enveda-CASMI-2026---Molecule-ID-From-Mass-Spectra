"""Cloud-only fragment calculations; export numeric evidence for exact in-list replay."""
import glob
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE))
import frag_rescore as frag

spec = importlib.util.spec_from_file_location('physical', CODE / 'parallax_physical.py')
post = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post)


def digest(spectra):
    h = hashlib.sha256()
    h.update(json.dumps([[s['adduct'], float(s['prec']), int(s['mode'])] for s in spectra]).encode())
    for spectrum in spectra:
        h.update(np.asarray(spectrum['mz'], dtype='<f8').tobytes())
        h.update(np.asarray(spectrum['it'], dtype='<f8').tobytes())
    return h.hexdigest()


def values(spectra, candidates, baseline=False, cache=None):
    kwargs = dict(cache=cache)
    if baseline:
        kwargs['cfg'] = frag.BASELINE_CFG
    return np.array([np.nan if x is None else x for x in frag.score_molecule(spectra, candidates, **kwargs)], dtype=float)


started = time.time()
labels = pd.read_parquet(sorted(glob.glob('/kaggle/input/**/catalogue_experiment/labelled_candidate_scores.parquet', recursive=True))[0])
query_files = sorted(glob.glob('/kaggle/input/**/catalogue_replay_checkpoint/batch_*/effective_query_spectra.parquet', recursive=True))
queries = pd.concat([pd.read_parquet(p) for p in query_files])
qmap = dict(tuple(queries.groupby('molecule_id', sort=False)))
frag.warmup()
arrays, metadata, cache, checked = {}, {}, {}, 0
for number, (mid, group) in enumerate(labels.groupby('molecule_id', sort=True)):
    candidates = group.sort_values('baseline_rank', kind='stable').smiles.astype(str).tolist()
    spectra = post.select_spectra(qmap[mid])
    selected = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in spectra]
    entry = dict(candidates=candidates, n_spectra=len(spectra), selected_spectra=selected, lookups={})
    if len(spectra) >= 2:
        for name, selected, baseline in [('opt_all', spectra, False), ('old_all', spectra, True)]:
            key = f'{number}_{name}'
            arrays[key] = values(selected, candidates, baseline, cache)
            entry['lookups'][digest(selected) + ('_old' if baseline else '_opt')] = key
        for chunk, start in enumerate(range(0, len(spectra), 2)):
            selected = spectra[start:start + 2]
            key = f'{number}_chunk{chunk}'
            arrays[key] = values(selected, candidates, False, cache)
            entry['lookups'][digest(selected) + '_opt'] = key
        # Verify that caching the full candidate set preserves actual production subset scores.
        subset = values(spectra, candidates[:12], False, cache)
        assert np.allclose(subset, arrays[f'{number}_opt_all'][:12], equal_nan=True, atol=1e-10)
        checked += 1
    metadata[str(mid)] = entry
    if (number + 1) % 10 == 0:
        print('FRAGMENT CACHE', number + 1, 'of', labels.molecule_id.nunique(), round(time.time() - started), 's', flush=True)
np.savez_compressed(ROOT / 'fragment_scores.npz', **arrays)
(ROOT / 'fragment_lookup.json').write_text(json.dumps(metadata))
audit = dict(molecules=len(metadata), candidate_rows=len(labels), subset_invariance_checks=checked,
             physical_source_sha256=hashlib.sha256((CODE / 'parallax_physical.py').read_bytes()).hexdigest(),
             query_file_sha256=sorted(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in query_files),
             seconds=time.time() - started, protocol='Numeric scores for all listed candidates; exact production spectrum selection and optimized/baseline/chunk scorers; no rank fitting')
(ROOT / 'fragment_cache_audit.json').write_text(json.dumps(audit, indent=2))
print('FRAGMENT CACHE DONE', json.dumps(audit), flush=True)
