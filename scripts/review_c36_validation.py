"""Independent OOF/source review and corrected offline mass diagnostics only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from prepare_c36_validation import ROOT,OUT,CONFIG,sha,write,verify
from build_external_cache import OFFSETS,normalize_adduct
from validation_admission import check_validation_report


def neutral_mass(precursor,adduct,mode):
    adduct=normalize_adduct(adduct)
    if adduct not in OFFSETS:return None
    if (adduct.endswith('+'))!=(mode=='positive'):return None
    charge,offset=OFFSETS[adduct]
    if not np.isfinite(precursor) or precursor<=0:return None
    return float(precursor)*charge-offset


def main():
    contract,frozen=verify();cfg=json.loads(CONFIG.read_text())
    assert not (OUT/'independent-review.json').exists()
    report=json.loads((OUT/'ranking-results.json').read_text());fit=json.loads((OUT/'fit-audits.json').read_text())
    prediction=json.loads((OUT/'prediction-freeze.json').read_text())
    for name,pin in prediction['files_sha256'].items():assert sha(OUT/name)==pin,name
    assert sha(OUT/'fit-audits.json')==prediction['fit_audits_sha256']
    assert report['prediction_freeze_sha256']==sha(OUT/'prediction-freeze.json')
    assert report['source_sha256']==sha(ROOT/'scripts/evaluate_c36_isomers.py')
    for variant,folds in fit['outer'].items():
        all_test=[]
        for r in folds:
            assert not (set(r['train_ids'])&set(r['test_ids']))
            assert not (set(r['train_groups'])&set(r['test_groups']))
            all_test.extend(r['test_ids'])
            for inner in r['selection']['audits']:
                assert not (set(inner['train_ids'])&set(inner['test_ids']))
                assert not (set(inner['train_groups'])&set(inner['test_groups']))
                assert set(inner['train_ids']+inner['test_ids'])==set(r['train_ids'])
        assert len(all_test)==len(set(all_test))==240
    metadata=pd.read_csv(OUT/'queries.csv',dtype={'group':str});primary=metadata[metadata.cohort.eq('primary')]
    stress=metadata[metadata.cohort.eq('source_stress')]
    for variant,r in fit['source_stress'].items():
        assert not(set(r['train_groups'])&set(stress.group))
        assert not(set(r['train_ids'])&set(stress.molecule_id))
    # The cached nm field is absent on25 formate queries. Recompute only this
    # validation diagnostic from labelled-free precursor/adduct inputs; do not
    # alter the frozen ranking features, predictions or original pipeline files.
    raw=pd.read_parquet(ROOT/contract['inputs']['queries']['path']).set_index('molecule_id')
    rows=[]
    for r in primary.itertuples(index=False):
        q=raw.loc[r.molecule_id];mass=neutral_mass(float(q.precursor_mz),str(q.adduct),str(q.ionization_mode))
        rows.append(dict(molecule_id=r.molecule_id,mode=r.mode,adduct=q.adduct,cached_nm_available=np.isfinite(r.query_neutral_mass),
            derived_nm=mass,truth_mass=r.truth_mass,error_ppm=None if mass is None else abs(mass-r.truth_mass)/r.truth_mass*1e6))
    derived=pd.DataFrame(rows);derived.to_csv(OUT/'derived-neutral-mass-diagnostics.csv',index=False)
    mass_report=dict(queries=240,cached_mass_available=int(derived.cached_nm_available.sum()),cached_mass_missing=int((~derived.cached_nm_available).sum()),
      missing_cached_by_adduct={str(k):int(v) for k,v in derived.loc[~derived.cached_nm_available,'adduct'].value_counts().items()},
      derived_mass_available=int(derived.derived_nm.notna().sum()),derived_errors_over30ppm=int(derived.error_ppm.gt(30).sum()),
      maximum_derived_error_ppm=float(derived.error_ppm.max()),diagnostic_only=True,ranking_inputs_unchanged=True)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),reviewer_source_sha256=sha(__file__),admission_source_sha256=sha(ROOT/'scripts/validation_admission.py'),
      ranking_report_sha256=sha(OUT/'ranking-results.json'),prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),feature_freeze_sha256=sha(OUT/'feature-freeze.json'),
      derived_mass_witness_sha256=sha(OUT/'derived-neutral-mass-diagnostics.csv'),group_fit_exclusions_passed=True,inner_selection_exclusions_passed=True,
      source_stress_chemistry_exclusions_passed=True,predictions_immutable_after_metrics=True,mass_diagnostic=mass_report,
      admission=check_validation_report(report,cfg,independent_confirmation=False,pretraining_audited=False),
      source_gate_scope='Independent review; no recipe or parameter was changed after source-stress results. Explicit source-regression blocking supplements the original evaluator gate.',
      reserved_labels_read=False,gpu_work=False,cloud_runs=False,submission_qualified=False)
    write(OUT/'independent-review.json',result);print('DONE C36 INDEPENDENT VALIDATION REVIEW',json.dumps(result),flush=True)


if __name__=='__main__':main()
