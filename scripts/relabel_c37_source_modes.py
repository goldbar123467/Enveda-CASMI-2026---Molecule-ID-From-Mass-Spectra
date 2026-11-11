"""Correct C36's first-record mode labels without changing its frozen predictions."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from trace_c37_selection import ROOT, OUT, C36, Identity, sha, write, verify, load_csv
from validation_repair import clustered_summary


def main():
    assert not (OUT/'corrected-source-polarity.json').exists()
    contract,_=verify();identity=Identity(contract)
    frozen=json.loads((C36/'prediction-freeze.json').read_text())
    name='source_stress-pairwise_ridge_full.csv'
    assert sha(C36/name)==frozen['files_sha256'][name]
    cfg=json.loads((ROOT/'configs/c36-cpu-validation-20261111.json').read_text())
    q=pd.read_parquet(ROOT/contract['inputs']['stress_queries']['path'])
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda s: 'mixed' if s.nunique()>1 else s.iloc[0])
    meta=pd.read_csv(C36/'queries.csv',dtype={'group':str});meta=meta[meta.cohort.eq('source_stress')].copy()
    meta['acquisition_mode']=meta.molecule_id.map(modes);assert meta.acquisition_mode.notna().all()
    parent=load_csv(ROOT/contract['inputs']['stress_parent']['path']);candidate=load_csv(C36/name)
    before=[];after=[]
    for r in meta.itertuples(index=False):
        def rr(items):
            keys=[identity(s)[0] for s in items]
            assert len(keys)==len(set(keys))<=25
            return 1./(keys.index(r.truth)+1) if r.truth in keys else 0.
        before.append(rr(parent[r.molecule_id]));after.append(rr(candidate[r.molecule_id]))
    before=np.asarray(before);after=np.asarray(after)
    def summarize(mask):
        return clustered_summary(before[mask],after[mask],meta.group.to_numpy()[mask],cfg['bootstrap_draws'],cfg['bootstrap_seed'])
    prior=json.loads((C36/'ranking-results.json').read_text())['source_stress']['pairwise_ridge_full']
    all_result=summarize(np.ones(len(meta),bool));clean_result=summarize(~meta.raw_alias_flag.to_numpy(bool))
    assert abs(all_result['delta']-prior['all']['delta'])<1e-14
    assert abs(clean_result['delta']-prior['raw_alias_free']['delta'])<1e-14
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
                frozen_prediction_sha256=sha(C36/name),query_metadata_sha256=sha(ROOT/contract['inputs']['stress_queries']['path']),
                corrected_mode_counts=meta.acquisition_mode.value_counts().to_dict(),
                first_record_label_crosstab=pd.crosstab(meta['mode'],meta.acquisition_mode).to_dict(),
                primary_variant_only='Previously frozen C36 pairwise_ridge_full; no new fitting, parameter choice, or predictions',
                all80=all_result,predeclared_clean60=clean_result,
                acquisition_mode_results={m:summarize(meta.acquisition_mode.eq(m).to_numpy()) for m in ['positive','negative','mixed']},
                interpretation='C36 negative31 means first-record-negative, not31 exclusively negative queries. Report pure-negative14, pure-positive30 and mixed36 separately. Overall80 and predeclared clean60 results are unchanged; C36 remains rejected.',
                old_c36_artifacts_unchanged=True,fresh_holdout_opened=False,model_or_rule_selected=False,new_gpu_work=False)
    write(OUT/'corrected-source-polarity.json',result);print('DONE C37 SOURCE MODES',json.dumps(result),flush=True)


if __name__=='__main__':main()
