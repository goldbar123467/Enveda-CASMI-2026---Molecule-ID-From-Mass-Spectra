"""One prospectively recorded bounded rule on USED panels, no new outcomes."""
import numpy as np
import pandas as pd
from pathlib import Path
from c45_complete_parent import ROOT,read,write,sha,stamp
from c56_bounded_consensus_tail import order
from evaluate_c45_complete import point
OUT=ROOT/'outputs/local/c56-bounded-semantic-proxy-20261112'
C54=ROOT/'outputs/local/c48-clean-residual-heads-20261112'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'used-development-rule-results.json').exists()
    assert not (ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert not read(ROOT/'outputs/local/c54-isomer-fingerprint-heads-20261112/used-development-results.json')['passed_pair']
    commitment=dict(frozen_utc=stamp(),source_sha256=sha(__file__),tail_sha256=sha(ROOT/'scripts/c56_bounded_consensus_tail.py'),
        original_C54_family_sha256=sha(C54/'complete-candidate-freeze.json'),old352_sha256=sha(C54/'used-development-results.json'),
        old35_sha256=sha(ROOT/'outputs/local/c51-complete-auxiliary-20261112/source-results.json'),C54_fingerprint_recipe_rejected=True,proxy_only=True,new_C53_overlap_in_old_models=True,clean_refit_required_before_real_candidate=True,
        rationale='Protect the first2ranks against high-MRR damage and freeze low-reward ranks11to25. Only ranks3to10 are reordered by unchanged same-formula0.20/offset3RRF; GPU inversions additionally require both semantic heads. Reuse old semantic scores as a hypothesis proxy only; those models include newC53chemistry and CANNOT qualify. New clean fitting is required if this proxy passes. One global rule, no source/identity exception, threshold grid or new fit.',
        model_weights_unchanged=True,candidate_count=dict(cpu=1,gpu=1),original352_and_used35_gates_unchanged=True,
        all_original_queries_retained=True,new_C53_outcomes_read=False,C13_read=False)
    assert not (OUT/'used-development-rule-commitment.json').exists();write(OUT/'used-development-rule-commitment.json',commitment)
    reports={}
    for panel,base,label_path in [('C39',ROOT/'outputs/local/c45-complete-parent-validation-20261112',ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv'),
        ('C51',ROOT/'outputs/local/c51-complete-auxiliary-20261112',ROOT/'outputs/local/c48-auxiliary-holdout-20261112/labels.csv')]:
        scores=C54/'C39-predictions' if panel=='C39' else base/'cpu-predictions';receipt=read(scores/'score-receipt.json')
        for name,pin in receipt['files_sha256'].items():assert sha(scores/name)==pin
        parent=pd.read_csv(base/'P16.csv',dtype=str,keep_default_na=False);mids=parent.molecule_id.tolist();labels=pd.read_csv(label_path,dtype=str).set_index('molecule_id').loc[mids]
        ids=pd.read_parquet(scores/'candidate-identities.parquet').set_index('smiles');raw=pd.read_parquet(scores/'candidate-scores.parquet')
        support=pd.read_csv(scores/'query-coverage.csv').set_index('molecule_id').supported.to_dict()
        truth=labels.key.tolist();oldkeys=[ids.loc[text.split(';'),'key'].tolist() for text in parent.smiles]
        baseline=np.asarray([row.index(key)+1 if key in row else 0 for row,key in zip(oldkeys,truth)],np.int16)
        q=pd.read_parquet(base/'queries.parquet');modes=q.groupby('molecule_id').ionization_mode.agg(lambda s:'mixed' if s.nunique()>1 else s.iloc[0]).loc[mids].to_numpy()
        if panel=='C51':raw=pd.concat([raw,pd.read_parquet(base/'gpu-predictions/candidate-scores.parquet')],ignore_index=True)
        reference={mid:g.sort_values('parent_rank') for mid,g in raw[raw.trial.eq('cpu')].groupby('molecule_id')}
        results={};rank_arrays=dict(parent=baseline)
        for trial in ['cpu','gpu']:
            values={mid:g.sort_values('parent_rank') for mid,g in raw[raw.trial.eq(trial)].groupby('molecule_id')};rows=[];ranks=[]
            for old in parent.itertuples(index=False):
                strings=old.smiles.split(';');table=values[old.molecule_id];assert table.smiles.tolist()==strings
                perm=order(ids.loc[strings,'formula'].tolist(),table.score.to_numpy(float),table.eligible.to_numpy(bool),bool(support[old.molecule_id]),reference_scores=reference[old.molecule_id].score.to_numpy(float) if trial=='gpu' else None)
                chosen=[strings[i] for i in perm];rows.append(dict(molecule_id=old.molecule_id,smiles=';'.join(chosen)))
                keys=ids.loc[chosen,'key'].tolist();key=labels.loc[old.molecule_id,'key'];ranks.append(keys.index(key)+1 if key in keys else 0)
            pd.DataFrame(rows).to_csv(OUT/(panel+'-'+trial+'.csv'),index=False);ranks=np.asarray(ranks,np.int16);rank_arrays[trial]=ranks
            n=len(parent);record=dict(all=point(baseline,ranks,np.ones(n,bool)),
                source={s:point(baseline,ranks,labels.source.eq(s).to_numpy()) for s in sorted(set(labels.source))},
                query_mode={m:point(baseline,ranks,modes==m) for m in sorted(set(modes))})
            if panel=='C39':record['original_development192']=point(baseline,ranks,labels.partition.eq('development').to_numpy())
            passed=record['all']['nonnegative'] and all(v['nonnegative'] for f in ['source','query_mode'] for v in record[f].values())
            if panel=='C39':passed=passed and record['all']['strictly_positive'] and record['original_development192']['strictly_positive']
            record['passed']=passed;record['changed_lists']=sum(a!=b['smiles'] for a,b in zip(parent.smiles,rows));results[trial]=record
        np.savez(OUT/(panel+'-used-ranks.npz'),**rank_arrays);reports[panel]=results
    passed=all(v['passed'] for r in reports.values() for v in r.values())
    write(OUT/'used-development-rule-results.json',dict(completed_utc=stamp(),source_sha256=sha(__file__),commitment_sha256=sha(OUT/'used-development-rule-commitment.json'),
        results=reports,passed_pair=passed,new_C53_quality_read=False,C13_read=False,
        files_sha256={p.name:sha(p) for p in OUT.iterdir() if p.suffix in {'.csv','.npz'}},statistical_promotion=False))
    print('C56 ONE BOUNDED RULE RESULTS',reports,'PAIR',passed,flush=True)


if __name__=='__main__':main()
