"""Check every actual auxiliary acquisition and input field against its seal."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha


def main():
    cloud=ROOT/'outputs/forks/c57-full-masked-validation-v1';target=cloud/'acquisition-fidelity-audit.json';assert not target.exists()
    original_path=ROOT/'outputs/publish/c53-target-holdout-20261112/validation-queries.parquet';freeze=read(ROOT/'outputs/local/c53-target-holdout-20261112/validation-freeze.json')
    assert freeze['files_sha256'][str(original_path.relative_to(ROOT))]==sha(original_path)
    original=pd.read_parquet(original_path);actual=pd.read_parquet(cloud/'c41_queries.parquet')
    assert len(original)==len(actual)==2125 and actual.molecule_id.nunique()==482 and actual.spectrum_id.is_unique
    assert set(original.columns)<=set(actual.columns),set(original.columns)-set(actual.columns)
    assert not {'smiles','normalized_smiles','key','formula','inchikey14'}&set(actual.columns)
    actual=actual.set_index('spectrum_id').loc[original.spectrum_id].reset_index()
    assert actual.molecule_id.tolist()==original.molecule_id.tolist()
    for name in original.columns:
        for a,b in zip(original[name],actual[name]):
            if isinstance(a,(np.ndarray,list,tuple)) or isinstance(b,(np.ndarray,list,tuple)):np.testing.assert_array_equal(a,b)
            else:assert a==b,(name,a,b)
    modes=actual.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).value_counts().to_dict()
    assert modes=={'positive':307,'mixed':141,'negative':34}
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),queries=482,acquisitions=2125,
        original_query_sha256=sha(original_path),cloud_query_sha256=sha(cloud/'c41_queries.parquet'),all_original_acquisitions_retained=True,
        all_original_metadata_and_peak_arrays_exact=True,actual_query_modes=modes,quality_labels_read=False,C13_read=False,
        limitation='Controlled in-corpus target-domain holdout, not independent-laboratory calibration. Historic public parent/encoder exposure persists.')
    write(target,result);print('C57 FULL TARGET ACQUISITION FIDELITY',json.dumps(result))


if __name__=='__main__':main()
