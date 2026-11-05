"""Read-only adapter for cloud-computed fragment evidence, with exact input checks."""
import ast
import hashlib
import json
import types
from pathlib import Path

import numpy as np


def spectrum_digest(spectra):
    h = hashlib.sha256()
    h.update(json.dumps([[s['adduct'], float(s['prec']), int(s['mode'])] for s in spectra]).encode())
    for s in spectra:
        h.update(np.asarray(s['mz'], dtype='<f8').tobytes())
        h.update(np.asarray(s['it'], dtype='<f8').tobytes())
    return h.hexdigest()


def physical_module(notebook, expected_sha256):
    """Extract the pinned upstream module as literal data; never execute the notebook."""
    notebook = json.loads(Path(notebook).read_text(encoding='utf-8'))
    sources = []
    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                value = node.value.value
                if hashlib.sha256(value.encode()).hexdigest() == expected_sha256:
                    sources.append(value)
    assert len(sources) == 1, 'Expected exactly one pinned physical module'
    source = sources[0]
    assert hashlib.sha256(source.encode()).hexdigest() == expected_sha256
    module = types.ModuleType('cached_upstream_physical')
    exec(compile(source, 'upstream/parallax_physical.py', 'exec'), module.__dict__)
    return module


class CachedFragment:
    BASELINE_CFG = object()

    def __init__(self, folder):
        folder = Path(folder)
        self.lookup = json.loads((folder / 'fragment_lookup.json').read_text())
        with np.load(folder / 'fragment_scores.npz', allow_pickle=False) as stored:
            self.scores = {key:stored[key] for key in stored.files}
        self.mid = None

    def score_molecule(self, spectra, candidates, *, cache=None, cfg=None):
        entry = self.lookup[str(self.mid)]
        suffix = '_old' if cfg is self.BASELINE_CFG else '_opt'
        assert cfg is None or cfg is self.BASELINE_CFG
        key = entry['lookups'][spectrum_digest(spectra) + suffix]
        positions = {s: i for i, s in enumerate(entry['candidates'])}
        assert len(positions) == len(entry['candidates'])
        values = self.scores[key][[positions[s] for s in candidates]]
        return [float(v) if np.isfinite(v) else None for v in values]

    def selected_spectra(self, frame, max_n=12):
        """Use the cloud's exact tie ordering; raw input files are separately hash-checked."""
        assert frame.molecule_id.astype(str).nunique() == 1
        mid = str(frame.molecule_id.iloc[0])
        assert mid == str(self.mid) and max_n == 12
        return [{k: np.asarray(v, dtype=float) if k in ('mz', 'it') else v for k,v in s.items()}
                for s in self.lookup[mid]['selected_spectra']]
