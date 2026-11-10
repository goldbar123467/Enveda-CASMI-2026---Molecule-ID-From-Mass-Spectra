"""Actual GPU deployment parity and real-input fallback fixtures; no labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from precision_cache import ROOT,sha,write
from prepare_c25_reserved_inputs import OUT as INPUT,verify
from followup_models import serving_bundle
from negative_tail import merge
from negative_deployment import run


def main():
    verify();out=ROOT/'outputs/local/c26-deployment-audit-20261110';out.mkdir(exist_ok=True)
    assert not (out/'audit.json').exists() and not (out/'source-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    bundle,record,_=serving_bundle('p15-parent',out/'bundle')
    parent=pd.read_csv(INPUT/'P01.csv',dtype=str,keep_default_na=False);queries=pd.read_parquet(INPUT/'queries.parquet')
    index=pd.read_parquet(INPUT/'fingerprint-smiles.parquet');fp=np.load(INPUT/'fingerprints.npy',mmap_mode='r')
    shared={s:np.asarray(fp[i]) for i,s in enumerate(index.smiles)}
    modes=dict(zip(queries.molecule_id,queries.ionization_mode))
    control=pd.read_parquet(INPUT/'P15-scores.parquet');negative=control[control.molecule_id.map(modes).eq('negative')]
    expected,_=merge(parent,negative,modes,.2)
    source=queries[queries.ionization_mode.eq('negative')].iloc[0]
    extra_parent=[];extra_queries=[]
    for kind in ['absent','unknown','mixed','short']:
        mid='C26_fixture_'+kind
        extra_parent.append(dict(molecule_id=mid,smiles=parent.smiles.iloc[0]))
        if kind=='absent':continue
        row=source.copy();row.molecule_id=mid
        if kind=='unknown':row.ionization_mode='unknown'
        if kind=='short':
            row.ms2_mzs=np.asarray(row.ms2_mzs)[:2];row.ms2_normalized_intensities=np.asarray(row.ms2_normalized_intensities)[:2];row.num_peaks=2
        extra_queries.append(row)
        if kind=='mixed':
            other=row.copy();other.ionization_mode='positive';extra_queries.append(other)
    all_parent=pd.concat([parent,pd.DataFrame(extra_parent)],ignore_index=True)
    all_queries=pd.concat([queries,pd.DataFrame(extra_queries)],ignore_index=True)
    write(out/'source-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in ['negative_deployment.py','audit_negative_deployment.py','negative_tail.py','compatible_tail.py']},
        input_preparation_sha256=sha(INPUT/'preparation.json'),P15_scores_sha256=sha(INPUT/'P15-scores.parquet'),
        model_manifest_sha256=sha(bundle/'c08-manifest.json'),real_queries=120,synthetic_fallback_cases=4,
        labels_read=False,quality_metrics_computed=False,model_selected=False))
    final,receipt,mode_audit=run(all_parent,all_queries,bundle,record['arm'],out/'scores.parquet',shared_fps=shared)
    assert final.iloc[:120].reset_index(drop=True).equals(expected)
    assert final.iloc[120:].reset_index(drop=True).equals(pd.DataFrame(extra_parent))
    assert receipt['valid_rows']==124 and receipt['neural_scored_molecules']==60 and receipt['eligible_negative_queries']==61
    assert receipt['mode_counts']==dict(negative=61,positive=60,absent=1,unknown=1,mixed=1)
    assert receipt['unsupported_negative_molecules']==['C26_fixture_short'] and receipt['positive_parent_rows_exact']==60
    actual=pd.read_parquet(out/'scores.parquet');common=actual[actual.molecule_id.isin(modes)].set_index(['molecule_id','smiles'])
    reference=negative.set_index(['molecule_id','smiles']).loc[common.index]
    errors={name:float(np.max(np.abs(common[name]-reference[name]))) for name in ['c08_raw_fold0','c08_raw_fold1','c08_score']}
    assert max(errors.values())<1e-6,errors
    final.to_csv(out/'fixture-output.csv',index=False);mode_audit.to_csv(out/'mode-audit.csv',index=False)
    write(out/'audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),real_queries=120,synthetic_fallback_cases=4,
        fixed_original_C08_control_only=True,all120_outputs_equal_existing_negative_tail_composition=True,
        all60_positive_rows_exact_parent=True,all4_fallback_rows_exact_parent=True,known_negative_supported=60,
        raw_and_normalized_score_max_abs_error=errors,deployment_receipt=receipt,
        source_freeze_sha256=sha(out/'source-freeze.json'),fixture_csv_sha256=sha(out/'fixture-output.csv'),
        C13_confirmation_metrics_read=False,labels_read=False,quality_metrics_computed=False,
        selected_group_changed=False,submission_qualified=False))
    print('DONE C26 NEGATIVE DEPLOYMENT AUDIT',json.dumps(json.loads((out/'audit.json').read_text())),flush=True)


if __name__=='__main__':main()
