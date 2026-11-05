# ---- x6: which training molecules come from which library (CPU) ----
# The scorer cache (n0) keeps no library column. This maps every spectrum library of train.parquet to molecules and,
# for the Enveda / timsTOF libraries (closest to the hidden test), to the cache's molecule key
# (kit features.structure: stereo-stripped, tautomer-canonical InChIKey14), so a local GPU job can hold them out.
import json, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
INP = Path('/kaggle/input'); W = Path('/kaggle/working')
TRAIN = next(INP.rglob('train.parquet'))
KIT = next(p.parent for p in INP.rglob('features.py') if (p.parent / 'train.py').is_file())
sys.path.insert(0, str(KIT))
from features import structure
cols = ['ingest_lib', 'instrument_type', 'ionization_mode', 'inchikey14', 'normalized_smiles']
df = pq.read_table(TRAIN, columns=cols).to_pandas()
log('ROWS', len(df), TRAIN)
summary = {}
for lib, g in df.groupby('ingest_lib', sort=False):
    summary[str(lib)] = dict(spectra=len(g), molecules=int(g.inchikey14.nunique()),
                             instruments=dict(Counter(g.instrument_type.astype(str)).most_common(6)),
                             modes=dict(Counter(g.ionization_mode.astype(str))))
summary = dict(sorted(summary.items(), key=lambda kv: -kv[1]['spectra']))
inst = df.groupby('instrument_type').inchikey14.nunique().sort_values(ascending=False)
(W / 'lib_summary.json').write_text(json.dumps(dict(libraries=summary, molecules_by_instrument=inst.astype(int).to_dict()), indent=2))
print(json.dumps(summary, indent=1)[:6000]); print(inst.head(20))
# Molecule-level library membership for every library (inchikey14 level; cheap).
mem = df.groupby(['inchikey14', 'ingest_lib', 'instrument_type'], observed=True).size().rename('spectra').reset_index()
mem.to_parquet(W / 'lib_membership.parquet', index=False)
log('MEMBERSHIP rows', len(mem))
# Exact cache keys for the NP-like subset.
sub = df[df.ingest_lib.astype(str).str.contains('enveda', case=False) | df.instrument_type.astype(str).str.contains('tims', case=False)]
smiles = sorted(sub.normalized_smiles.astype(str).unique())
log('NP-LIKE rows', len(sub), 'unique smiles', len(smiles))
def key(s):
    try: return s, structure(s)[0]
    except Exception: return s, None
with Pool(4) as p:
    keys = dict(p.map(key, smiles, chunksize=256))
sub = sub.assign(key=sub.normalized_smiles.astype(str).map(keys))
out = sub.groupby(['key', 'inchikey14', 'ingest_lib', 'instrument_type', 'ionization_mode'], observed=True, dropna=False).size().rename('spectra').reset_index()
out.to_parquet(W / 'np_like_keys.parquet', index=False)
log('NP-LIKE keys', out.key.nunique(), 'invalid', int(out.key.isna().sum()), 'rows', len(out))
