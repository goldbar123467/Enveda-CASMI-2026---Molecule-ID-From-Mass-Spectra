"""Verify all original120C13acquisitions, retaining unknown/trailing tokens."""
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp


def main():
    cloud=ROOT/'outputs/forks/c59-c13-main-compatibility-v1';target=cloud/'acquisition-fidelity-audit.json';assert not target.exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    source=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet'
    pin=read(ROOT/'kaggle/forks/c59-c13-main-compatibility/compatibility-pin.json');assert pin['query_sha256']==sha(source)
    original=pd.read_parquet(source);actual=pd.read_parquet(cloud/'c13_queries.parquet')
    assert len(original)==len(actual)==120 and original.molecule_id.is_unique and actual.molecule_id.is_unique
    assert set(original.columns)<=set(actual.columns) and not {'smiles','normalized_smiles','key','inchikey14','formula','mass','source_number'}&set(actual.columns)
    actual=actual.set_index('molecule_id').loc[original.molecule_id].reset_index()
    for name in original.columns:
        for a,b in zip(original[name],actual[name]):
            if isinstance(a,(np.ndarray,list,tuple)) or isinstance(b,(np.ndarray,list,tuple)):np.testing.assert_array_equal(a,b)
            elif pd.isna(a):assert pd.isna(b)
            else:assert a==b,(name,a,b)
    assert actual.ionization_mode.value_counts().to_dict()=={'positive':60,'negative':60}
    assert actual.adduct.value_counts().to_dict()=={'[M+H]+':34,'[M-H]-':29,'[M+FA-H]-':21,'[M+NH4]+':16,'[M+Na]+':10,'[M+Cl]-':9,'[M+FA-H]- ':1}
    write(target,dict(completed_utc=stamp(),source_sha256=sha(__file__),queries=120,acquisitions=120,
        original_query_sha256=sha(source),cloud_query_sha256=sha(cloud/'c13_queries.parquet'),all_original_acquisitions_retained=True,
        all_original_metadata_peaks_and_unknown_CE_exact=True,original_trailing_space_token_preserved=True,
        actual_query_modes={'positive':60,'negative':60},quality_outcomes_read=False,C13_consumed=False,
        limitation='Original FT/HCD external confirmation; not target timsTOF instrument calibration.'))
    print('C59 ALL120ORIGINAL ACQUISITIONS EXACT',flush=True)


if __name__=='__main__':main()
