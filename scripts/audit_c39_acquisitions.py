"""Check acquisition/mass integrity without matching predictions to labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write
from legacy_formate_compat import scalar_namespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud',type=Path);a=p.parse_args()
    base=ROOT/'outputs/local/c39-source-supported-validation-20261111'
    original=ROOT/'outputs/publish/c39-source-supported-validation-20261111/validation-queries.parquet'
    q=pd.read_parquet(original);lab=pd.read_csv(base/'labels.csv').set_index('molecule_id')
    freeze=json.loads((base/'validation-freeze.json').read_text())
    assert sha(original)==freeze['files_sha256'][str(original.relative_to(ROOT))]
    assert sha(base/'labels.csv')==freeze['files_sha256'][str((base/'labels.csv').relative_to(ROOT))]
    assert len(q)==1292 and q.molecule_id.nunique()==352 and q.spectrum_id.is_unique
    assert not {'smiles','key','inchikey14','formula','mass'}&set(q.columns)
    chem=ROOT/'outputs/local/c41-masked-pipeline-validation-20261111/static-compat-fixture/patched_code/casmi/chem.py'
    ns=scalar_namespace(chem.read_text(encoding='utf8'))
    neutral=ns['neutral_mass'](q.precursor_mz.to_numpy(float),q.adduct.to_numpy())
    expected=q.molecule_id.map(lab.mass).to_numpy(float)
    ppm=np.abs(neutral-expected)/expected*1e6
    assert np.isfinite(ppm).all() and (ppm<=30).all()
    for row in q.itertuples(index=False):
        mz,it=np.asarray(row.ms2_mzs,float),np.asarray(row.ms2_normalized_intensities,float)
        assert len(mz)==len(it)>=3 and np.isfinite(mz).all() and np.isfinite(it).all()
        assert (mz>0).all() and (it>0).all() and (np.diff(mz)>=0).all()
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda x:'mixed' if x.nunique()>1 else x.iloc[0])
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        query_sha256=sha(original),validation_freeze_sha256=sha(base/'validation-freeze.json'),queries=352,acquisitions=1292,
        actual_query_modes=modes.value_counts().to_dict(),maximum_neutral_mass_error_ppm=float(ppm.max()),
        acquisitions_over30ppm=int((ppm>30).sum()),per_source={},quality_metrics_read=False,predictions_read=False,C13_read=False)
    for source in sorted(set(lab.source)):
        mask=q.molecule_id.map(lab.source).eq(source).to_numpy()
        result['per_source'][source]=dict(queries=int(q.loc[mask,'molecule_id'].nunique()),acquisitions=int(mask.sum()),maximum_mass_error_ppm=float(ppm[mask].max()))
    if a.cloud:
        cloud=pd.read_parquet(a.cloud).set_index('spectrum_id');source=q.set_index('spectrum_id')
        assert cloud.index.is_unique and set(cloud.index)==set(source.index) and len(cloud)==1292
        cloud=cloud.loc[source.index]
        for name in ['molecule_id','adduct','ionization_mode']:assert cloud[name].astype(str).equals(source[name].astype(str)),name
        diffs={}
        for name in ['precursor_mz','ms2_mzs','ms2_normalized_intensities']:
            maximum=0.
            for before,after in zip(source[name],cloud[name]):
                before,after=np.asarray(before,float),np.asarray(after,float)
                assert before.shape==after.shape and np.allclose(before,after,atol=1e-6,rtol=5e-7),name
                maximum=max(maximum,float(np.max(np.abs(before-after))))
            diffs[name]=maximum
        result.update(cloud_query_sha256=sha(a.cloud),all_original_acquisitions_retained=True,cloud_scalar_array_max_abs_differences=diffs)
    target=(a.cloud.parent/'acquisition-fidelity-audit.json') if a.cloud else base/'acquisition-integrity-audit.json'
    assert not target.exists();write(target,result);print('C39 ACQUISITION AUDIT PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
