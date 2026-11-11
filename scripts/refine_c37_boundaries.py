"""Refine the 26 consumed-panel witnesses without proposing a new ranker."""
import ast
import json
from collections import Counter
from datetime import datetime, timezone
import pandas as pd
from rdkit import RDLogger
from trace_c37_selection import ROOT, OUT, C36, Identity, sha, write, verify, extract_executed


def main():
    RDLogger.DisableLog('rdApp.*'); contract, _ = verify(); identity = Identity(contract)
    assert not (OUT/'boundary-refinement.json').exists()
    trace = json.loads((OUT/'selection-trace.json').read_text())
    assert sha(OUT/'selection-boundary-witnesses.csv') == trace['witnesses_sha256']
    f = pd.read_csv(OUT/'selection-boundary-witnesses.csv')
    metadata = pd.read_csv(C36/'queries.csv').set_index('molecule_id')
    sources = {}; datasets = {}; audits = {}
    for prefix in ['c10','c20']:
        source, _ = extract_executed(prefix); tree = ast.parse(source)
        assert any(isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE_ICE' for t in n.targets)
                   and ast.literal_eval(n.value) is False for n in tree.body)
        node = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='metric_fusion')
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'executed_pure_fusion','exec'),ns)
        sources[prefix]=ns['metric_fusion']
        d=OUT/(prefix+'-stage-artifacts');datasets[prefix]=json.loads((d/'e004_candidate_inputs.json').read_text())
        audits[prefix]=pd.read_csv(d/'pubchem_promotion_audit.csv').set_index('molecule_id')
    rows=[]
    for r in f[f.historical_dropped26].itertuples(index=False):
        prefix='c10' if r.mode=='positive' else 'c20';d=datasets[prefix];mid=r.molecule_id
        prefusion=d['prefusion'][mid];base=d['base_tail'][mid][:25];pc=d['pubchem'].get(mid,{}).get('pc',[])
        record=dict(molecule_id=mid,mode=r.mode,observed_boundary=r.last_observed_boundary,
                    fusion40_truth_rank=r.metric_fusion40_truth_rank,base_tail_truth_rank=r.base_tail_truth_rank,
                    pubchem_truth_rank=r.pubchem_truth_rank)
        cause=r.last_observed_boundary
        if cause=='pubchem_merge_gate_or_budget':
            assert float(metadata.loc[mid,'library'])<.9 and mid in audits[prefix].index
            # BASE_ICE=False proves base[:25] is not physically reranked here.
            # Observe which raw PC representatives actually occupy the merged
            # list; do not guess unavailable original database keys.
            selected=[s for s in prefusion if s not in set(base)]
            selected_pc_ranks=[pc.index(s)+1 for s in selected]
            assert len(selected)==5 and all(j<r.pubchem_truth_rank for j in selected_pc_ranks)
            cause='pubchem_five_slot_budget_exhausted'
            record.update(selected_pubchem_count=len(selected),selected_pubchem_raw_ranks=selected_pc_ranks,
                          library_protection_fired=False,merge_promoted=bool(audits[prefix].loc[mid,'new_promote']),
                          merge_channel_rel=float(audits[prefix].loc[mid,'channel_rel']))
        elif cause=='main_prefusion_truncation_or_merge':
            if r.base_tail_truth_rank>25:cause='main_top60_to_top25_cutoff'
            else:
                assert r.base_tail_truth_rank==22
                kept=[s for s in prefusion if s in set(base)]
                assert kept==base[:20] and len([s for s in prefusion if s not in set(base)])==5
                cause='pubchem_interleave_displaced_main_rank22'
        elif cause=='reciprocal_fusion_40_boundary':
            first=[s for s in prefusion if s and s!='CCO']
            full,_,_=sources[prefix](first,d['engine2'][mid].get('smiles',[]),identity,limit=1000)
            target=identity(metadata.loc[mid,'truth_smiles']);rank=[identity(s) for s in full].index(target)+1
            assert rank>40;record['untruncated_fusion_truth_rank']=rank
        elif cause.startswith('same_formula_'):
            assert r.metric_fusion40_truth_rank>0 and r.e006_truth_rank==0
            if r.metric_fusion40_truth_rank<=25:
                cause='same_formula_'+r.e006_branch+'_demoted_across_25'
            else:cause='outside_top25_before_and_after_'+r.e006_branch
            record['exact_internal_reranked_rank_retained']=False
        record['refined_observed_decision']=cause;rows.append(record)
    identity.save();w=pd.DataFrame(rows);assert len(w)==26
    w.to_csv(OUT/'refined26-boundary-witnesses.csv',index=False)
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
                trace_report_sha256=sha(OUT/'selection-trace.json'),witness_sha256=sha(OUT/'refined26-boundary-witnesses.csv'),
                queries=26,counts=w.refined_observed_decision.value_counts().to_dict(),
                by_mode={mode:g.refined_observed_decision.value_counts().to_dict() for mode,g in w.groupby('mode')},
                source_BASE_ICE_false_verified=True, all10_pubchem_losses_unprotected_with_five_prior_proposals_selected=True,
                interpretation='Observed cutoffs, fixed source budgets and same-formula order changes are explicit policies. These losses alone establish neither an implementation defect nor a beneficial replacement rule.',
                rank2_overlay_true_evictions=0, postfrozen_true_evictions=0,
                retention_rule_frozen_or_quality_tested=False,
                no_retention_test_reason='No label-independent correctness invariant was violated by the selectors. Choosing a new quota or injecting the known missing truths would be outcome-guided ranking design; no defensible frozen rule has emerged from this trace.',
                physical_limit='Saved lists establish three demotions across the25 boundary and four answers already outside25; internal final ranks26-40 and deterministic negative fragment scores were not saved, so no exact within-tail ranks or numerical physical-score cause are claimed.',
                validation_fallback='Use the separately frozen query-only population support audit and prospective validation protocol.',
                untouched_holdout_opened=False,new_predictions=False,new_gpu_work=False)
    write(OUT/'boundary-refinement.json',result);print('DONE C37 REFINEMENT',json.dumps(result),flush=True)


if __name__=='__main__':main()
