# CPU probe: signatures of the visible test file, used to detect the hidden-test rerun. No training.
import os, json, hashlib
import numpy as np, pandas as pd
from pathlib import Path
INPUT = Path('/kaggle/input')
COMP = next(p for p in (INPUT / 'competitions/enveda-CASMI26-molecule-id-mass-spectra',
                        INPUT / 'enveda-CASMI26-molecule-id-mass-spectra',
                        INPUT / 'enveda-casmi26-molecule-id-mass-spectra')
            if all((p / f).is_file() for f in ('train.parquet', 'test.parquet', 'sample_submission.csv')))
print('COMP', COMP)
print('env', sorted(k for k in os.environ if 'KAGGLE' in k))
_ids = pd.read_parquet(COMP / 'test.parquet', columns=['spectrum_id']).spectrum_id.astype(str)
print('ID_SIGNATURE', hashlib.md5(','.join(sorted(_ids)).encode()).hexdigest())

def content_signature(path):
    t = pd.read_parquet(path, columns=['spectrum_id', 'precursor_mz', 'ms2_mzs'])
    t = t.assign(spectrum_id=t.spectrum_id.astype(str)).sort_values('spectrum_id', kind='stable')
    rows = [[s, round(float(p), 4), int(len(m)), round(float(np.sum(np.asarray(m, dtype=np.float64))), 3)]
            for s, p, m in zip(t.spectrum_id, t.precursor_mz, t.ms2_mzs)]
    return hashlib.md5(json.dumps(rows).encode()).hexdigest()

print('CONTENT_SIGNATURE', content_signature(COMP / 'test.parquet'))
print('CONTENT_SIGNATURE_AGAIN', content_signature(COMP / 'test.parquet'))
te = pd.read_parquet(COMP / 'test.parquet')
print(te.shape, te.molecule_id.nunique()); print(te.dtypes.to_string()); print(te.head(3).to_string()[:1500])
print(pd.read_csv(COMP / 'sample_submission.csv').head(3).to_string())
