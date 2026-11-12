"""Post-rejection diagnosis of the entire now-USED C53 panel. No fitting."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'2')
import json,types
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha,stamp
from legacy_formate_compat import patch_chem_source
OUT=ROOT/'outputs/local/c63-source-loss-diagnosis-20261112'
BASE=ROOT/'outputs/local/c58-complete-target-20261112'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'diagnosis.json').exists()
    result=read(BASE/'source-results.json');audit=read(BASE/'independent-quality-audit.json')
    assert not result['passed_pair'] and not audit['passed_pair']
    assert audit['quality_results_sha256']==sha(BASE/'source-results.json')
    assert read(ROOT/'.local/c38-active-goal.json')['submissions_used']==0
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    ranks=np.load(BASE/'source-ranks.npz');frames={n:pd.read_csv(BASE/(n+'.csv'),dtype=str,keep_default_na=False) for n in ['P01','P16']}
    for trial in ['cpu','gpu']:frames[trial]=pd.read_csv(BASE/'head-predictions'/(trial+'.csv'),dtype=str,keep_default_na=False)
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==482
    labels=pd.read_csv(ROOT/'outputs/local/c53-target-holdout-20261112/labels.csv').set_index('molecule_id').loc[mids]
    q=pd.read_parquet(BASE/'queries.parquet');modes=q.groupby('molecule_id').ionization_mode.agg(lambda a:'mixed' if a.nunique()>1 else a.iloc[0]).loc[mids]
    identity=pd.read_parquet(BASE/'quality-identity-map.parquet').set_index('smiles').key.to_dict()
    scores=pd.read_parquet(BASE/'head-predictions/candidate-scores.parquet')
    evidence=pd.read_parquet(ROOT/'outputs/forks/c57-full-masked-validation-v1/frozen_candidate_evidence.parquet').set_index(['molecule_id','metric_key'])
    cases=[]
    for trial in ['cpu','gpu']:
        for i in np.flatnonzero(ranks[trial]!=ranks['P16']):
            mid=mids[i];before=frames['P16'].smiles.iloc[i].split(';');after=frames[trial].smiles.iloc[i].split(';')
            old=[identity[s] for s in before];new=[identity[s] for s in after];truth=labels.key.iloc[i]
            assert truth in old and truth in new
            overtakers=[key for key in new[:new.index(truth)] if old.index(key)>old.index(truth)]
            assert overtakers
            group=scores[scores.trial.eq(trial)&scores.molecule_id.eq(mid)].copy();group['key']=group.smiles.map(identity)
            values=group.set_index('key')
            for wrong in overtakers:
                def ev(key):
                    if (mid,key) not in evidence.index:return {}
                    row=evidence.loc[(mid,key)]
                    if isinstance(row,pd.DataFrame):row=row.iloc[0]
                    return {k:None if pd.isna(row[k]) else float(row[k]) for k in ['ice','gl','proxy_gl','main_rank','engine2_rank','pubchem_rank']}
                cases.append(dict(trial=trial,molecule_id=mid,source=labels.source.iloc[i],mode=modes.iloc[i],formula=labels.formula.iloc[i],
                    truth_key=truth,overtaking_key=wrong,truth_smiles=before[old.index(truth)],overtaking_smiles=before[old.index(wrong)],
                    parent_truth_rank=int(ranks['P16'][i]),new_truth_rank=int(ranks[trial][i]),
                    truth_utility=float(values.loc[truth,'score']),wrong_utility=float(values.loc[wrong,'score']),
                    truth_reference_utility=float(values.loc[truth,'reference_score']),wrong_reference_utility=float(values.loc[wrong,'reference_score']),
                    truth_evidence=json.dumps(ev(truth)),wrong_evidence=json.dumps(ev(wrong))))
    pd.DataFrame(cases).to_csv(OUT/'head-loss-cases.csv',index=False)
    matches=pd.read_parquet(BASE/'target-parent/c09e_matches.parquet');reference=pd.read_parquet(BASE/'target-masked-reference/ext_meta.parquet')
    legacy=ROOT/'outputs/local/c37-selection-trace-20261111/pinned-public-source/chem.py';chem=types.ModuleType('c63_exact_mass')
    exec(compile(patch_chem_source(legacy.read_text(encoding='utf8')),str(legacy),'exec'),chem.__dict__)
    promotions=[]
    for i in np.flatnonzero(ranks['P16']!=ranks['P01']):
        mid=mids[i];hit=matches[matches.molecule_id.eq(mid)&matches.view.eq('all_reference')].iloc[0];ref=reference.iloc[int(hit.reference_row)]
        g=q[q.molecule_id.eq(mid)];nm=chem.neutral_mass(g.precursor_mz.to_numpy(),g.adduct.to_numpy())
        # Equivalent spelling at this diagnostic lookup only. Original query and
        # reference tokens are preserved; no prediction or cohort is changed.
        token={'[M+H-H2O]+':'[M-H2O+H]+'}.get(ref.adduct,ref.adduct)
        reference_nm=float(chem.neutral_mass(np.array([ref.precursor_mz]),np.array([token]))[0])
        query_nm=float(np.median(nm[np.isfinite(nm)]));ppm=abs(reference_nm-query_nm)/query_nm*1e6
        promotions.append(dict(molecule_id=mid,source=labels.source.iloc[i],mode=modes.iloc[i],original_rank=int(ranks['P01'][i]),promoted_parent_rank=int(ranks['P16'][i]),
            query_adducts=g.adduct.unique().tolist(),reference_adduct=ref.adduct,query_neutral_mass=query_nm,reference_neutral_mass=reference_nm,
            neutral_mass_error_ppm=ppm,cosine=float(hit.cosine),matched_peaks=int(hit.matched_peaks),same_adduct=bool(hit.same_adduct),
            reference_formula=ref.formula,truth_formula=labels.formula.iloc[i],all_acquisitions_retained=True))
    write(OUT/'parent-promotion-cases.json',promotions)
    # Maximum possible gain under the OLD frozen ranks3..10 restriction; this
    # describes this USED panel only, never the hidden leaderboard population.
    ceiling=sum((Fraction(1,3)-Fraction(1,int(r))) for r in ranks['P16'] if 3<=r<=10)/482
    report=dict(completed_utc=stamp(),source_sha256=sha(__file__),scope='All482now-USED C53queries/2125acquisitions; diagnosis only, no fresh confirmation or fitting',
        queries=482,acquisitions=2125,source_results_sha256=sha(BASE/'source-results.json'),independent_audit_sha256=sha(BASE/'independent-quality-audit.json'),
        frozen_pair_rejected=True,source_panel_now_used=True,rank_distribution={n:{str(int(a)):int(b) for a,b in zip(*np.unique(ranks[n],return_counts=True))} for n in ranks.files},
        complete_deltas={t:result['results'][t]['complete']['all']['point']['delta'] for t in ['cpu','gpu']},
        head_only_deltas={t:result['results'][t]['incremental']['all']['point']['delta'] for t in ['cpu','gpu']},
        head_improved_truths={t:result['results'][t]['incremental']['all']['improved'] for t in ['cpu','gpu']},
        head_harmed_truths={t:result['results'][t]['incremental']['all']['harmed'] for t in ['cpu','gpu']},
        parent_overlay_delta=result['parent_overlay']['all']['point']['delta'],parent_wrong_promotions=promotions,
        old_tail_only_truth_opportunities=int(((ranks['P16']>=3)&(ranks['P16']<=10)).sum()),old_protected_top2_truths=int(((ranks['P16']>=1)&(ranks['P16']<=2)).sum()),
        old_tail_perfect_order_maximum_incremental_delta=float(ceiling),old_tail_perfect_order_maximum_incremental_fraction=str(ceiling),
        proposed_next_work=['Test adduct-implied neutral-mass consistency on all USED P16 paths; raw precursor agreement is insufficient for dehydrated ions.',
            'Assemble all869USED queries for chemistry-group out-of-fold direct isomer-ranking tests with frozen core evidence; test CPU before new GPU fitting.',
            'Freeze any new candidate and an additional untouched source panel before prospective outcomes. Never call these482fresh again. Keep C13unconsumed until a defensible pair passes.'],
        C13_consumed=False,scoring_entries=0,new_GPU_fitting=False,
        limitation='Parent MRR0.926on this in-corpus panel is not leaderboard calibration; historic public parent/encoder exposure persists. CPU/GPU agreement is correlated because the GPU head shares its CPU anchor and training data.')
    write(OUT/'diagnosis.json',report);print(json.dumps({k:report[k] for k in ['head_only_deltas','head_improved_truths','head_harmed_truths','old_tail_only_truth_opportunities','old_protected_top2_truths','old_tail_perfect_order_maximum_incremental_delta']},indent=2))


if __name__=='__main__':main()
