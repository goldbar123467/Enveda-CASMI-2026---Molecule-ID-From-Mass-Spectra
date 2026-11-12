"""Independent full used-development ranks, agreement constraints and gates."""
import ast,copy,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from c45_complete_parent import ROOT,read,write,sha
from audit_c45_quality import check_point
from validation_repair import clustered_summary
OUT=ROOT/'outputs/local/c49-consensus-pair-20261112'
C48=ROOT/'outputs/local/c48-clean-residual-heads-20261112'
C45=ROOT/'outputs/local/c45-complete-parent-validation-20261112'


def assignment_to(node,name):return isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id==name for v in node.targets)


def main():
    target=OUT/'independent-used-development-audit.json';assert not target.exists()
    family=read(OUT/'complete-candidate-freeze.json');assert family['policy_sha256']==sha(ROOT/'configs/c49-consensus-pair-20261112.json')
    bundle=ROOT/family['candidates']['gpu']['bundle'];oldbundle=ROOT/family['candidates']['cpu']['bundle']
    for trial,entry in family['candidates'].items():
        folder=ROOT/entry['path'];assert sha(folder/'notebook.ipynb')==entry['notebook_sha256'] and sha(folder/'production-pin.json')==entry['pin_sha256']
        manifest=read(ROOT/entry['bundle']/'head-manifest.json');assert sha(ROOT/entry['bundle']/'head-manifest.json')==entry['head_manifest_sha256']
        for name,pin in manifest['files'].items():assert sha(ROOT/entry['bundle']/name)==pin
    for name in ['cpu-model.npz','gpu-model.pt','fit-scaling.npz','c48_gpu_model.py','features.py','crossmodal_heads.py']:assert sha(bundle/name)==sha(oldbundle/name)
    # The only learned-forward addition is a separate frozen CPU reference.
    # Removing that addition must recover the original CPU/GPU forward AST.
    def scorer(path):return next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='score')
    old=scorer(oldbundle/'score_c48_crossmodal.py');new=scorer(bundle/'score_c49_consensus.py')
    def trial_loop(fn):return next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='trial')
    a,b=trial_loop(old),trial_loop(new)
    assert ast.dump(ast.Module(body=old.body[:old.body.index(a)],type_ignores=[]))==ast.dump(ast.Module(body=new.body[:new.body.index(b)],type_ignores=[]))
    oldif=next(n for n in a.body if isinstance(n,ast.If));newif=copy.deepcopy(next(n for n in b.body if isinstance(n,ast.If)))
    newif.orelse=[n for n in newif.orelse if not assignment_to(n,'reference_queries')]
    assert ast.dump(oldif)==ast.dump(newif)
    origin=C48/'C39-predictions';receipt=read(origin/'score-receipt.json')
    for name,pin in receipt['files_sha256'].items():assert sha(origin/name)==pin
    assert receipt['head_manifest_sha256']==family['candidates']['cpu']['head_manifest_sha256']
    result_commitment=read(OUT/'used-development-rule-results.json');assert result_commitment['prediction_sha256']==sha(OUT/'development-gpu-consensus.csv')
    assert family['used_development_results_sha256']==sha(OUT/'used-development-rule-results.json')
    assert read(OUT/'used-development-rule-commitment.json')['tail_source_sha256']==sha(bundle/'crossmodal_tail.py')
    frames={'P01':pd.read_csv(C45/'P01.csv',dtype=str),'P16':pd.read_csv(C45/'P16.csv',dtype=str),
        'cpu':pd.read_csv(origin/'cpu.csv',dtype=str),'gpu':pd.read_csv(OUT/'development-gpu-consensus.csv',dtype=str)}
    mids=frames['P01'].molecule_id.tolist();assert len(set(mids))==352
    identities=pd.read_parquet(origin/'candidate-identities.parquet').set_index('smiles');scores=pd.read_parquet(origin/'candidate-scores.parquet')
    support=pd.read_csv(origin/'query-coverage.csv').set_index('molecule_id').supported.to_dict()
    raw={trial:{mid:g.sort_values('parent_rank') for mid,g in scores[scores.trial.eq(trial)].groupby('molecule_id')} for trial in ['cpu','gpu']}
    changed=0
    for parent,actual in zip(frames['P16'].itertuples(index=False),frames['gpu'].itertuples(index=False)):
        strings=parent.smiles.split(';');formula=identities.loc[strings,'formula'].tolist();a,b=raw['cpu'][parent.molecule_id],raw['gpu'][parent.molecule_id]
        assert a.smiles.tolist()==b.smiles.tolist()==strings
        expected=list(range(len(strings)));eligible=a.eligible.to_numpy(bool);assert np.array_equal(eligible,b.eligible.to_numpy(bool))
        if support[parent.molecule_id]:
            for f in set(formula[1:]):
                positions=[i for i in range(1,len(strings)) if formula[i]==f]
                if len(positions)<2 or not eligible[positions].all():continue
                utility=[]
                for table in [a,b]:
                    values=table.score.to_numpy(float);ranks=sorted(positions,key=lambda i:(-values[i],i));lookup={i:j for j,i in enumerate(ranks,1)}
                    utility.append({i:1/(3+i+1)+.2/(3+lookup[i]) for i in positions})
                # Independent insertion normal form, distinct from deployed
                # adjacent bubble passes; every inversion needs both votes.
                chosen=[]
                for i in positions:
                    chosen.append(i);j=len(chosen)-1
                    while j and all(u[chosen[j]]>u[chosen[j-1]] for u in utility):chosen[j],chosen[j-1]=chosen[j-1],chosen[j];j-=1
                loc={i:j for j,i in enumerate(chosen)}
                for i in positions:
                    for j in positions:
                        if i<j and loc[j]<loc[i]:assert all(u[j]>u[i] for u in utility)
                for i,j in zip(positions,chosen):expected[i]=j
        output=actual.smiles.split(';');assert output==[strings[i] for i in expected]
        assert output[0]==strings[0] and set(output)==set(strings) and identities.loc[output,'formula'].tolist()==formula
        assert len(set(identities.loc[output,'key']))==len(output);changed+=output!=strings
    labels=pd.read_csv(ROOT/'outputs/local/c39-source-supported-validation-20261111/labels.csv',dtype=str).set_index('molecule_id').loc[mids]
    ranks={};lists={}
    for name,frame in frames.items():
        assert frame.molecule_id.tolist()==mids;lists[name]=[identities.loc[s.split(';'),'key'].tolist() for s in frame.smiles]
        ranks[name]=np.asarray([{key:i for i,key in enumerate(row,1)}.get(truth,0) for row,truth in zip(lists[name],labels.key)],np.int16)
    stored=np.load(OUT/'development-consensus-ranks.npz');np.testing.assert_array_equal(stored['ranks'],ranks['gpu'])
    np.testing.assert_array_equal(ranks['gpu'],ranks['cpu'])
    q=pd.read_parquet(C45/'queries.parquet');assert len(q)==1292
    modes=q.groupby('molecule_id').ionization_mode.agg(lambda v:'mixed' if v.nunique()>1 else v.iloc[0]).loc[mids].to_numpy();assert pd.Series(modes).value_counts().to_dict()=={'positive':176,'mixed':126,'negative':50}
    record=result_commitment['results'];masks=dict(all=np.ones(352,bool),original_development192=labels.partition.eq('development').to_numpy())
    assert masks['original_development192'].sum()==192
    for name,mask in masks.items():check_point(record[name],ranks['P16'],ranks['gpu'],mask)
    assert set(record['source'])==set(labels.source) and set(record['query_mode'])==set(modes)
    for source,value in record['source'].items():check_point(value,ranks['P16'],ranks['gpu'],labels.source.eq(source).to_numpy())
    for mode,value in record['query_mode'].items():check_point(value,ranks['P16'],ranks['gpu'],modes==mode)
    assert record['all']['strictly_positive'] and record['original_development192']['strictly_positive']
    assert all(v['nonnegative'] for key in ['source','query_mode'] for v in record[key].values())
    old_quality=read(C48/'used-development-results.json');assert old_quality['results']['cpu']['admission']['passed']
    # Equality is on every official truth rank, not merely mean MRR. This
    # justifies reusing the exact same grouped uncertainty/strata summaries.
    stats=copy.deepcopy(old_quality['results']['cpu'])
    for comparison,base in [('complete','P01'),('incremental','P16')]:
        for name,mask in masks.items():
            field='all' if name=='all' else 'original_development192';check_point(stats[comparison][field]['point'],ranks[base],ranks['gpu'],mask)
    write(OUT/'used-development-quality.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        complete_candidate_freeze_sha256=sha(OUT/'complete-candidate-freeze.json'),original_CPU_quality_sha256=sha(C48/'used-development-results.json'),
        results=dict(cpu=copy.deepcopy(stats),gpu=copy.deepcopy(stats)),passed_pair=True,queries=352,acquisitions=1292,
        uncertainty_reuse_justification='All352official truth ranks exactly equal the independently verified CPU vectors; groups/modes/denominators are identical. Candidate lists remain different on211queries.',
        C39_is_used_development=True,new_source_outcomes_read=False,C13_read=False))
    result=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),quality_results_sha256=sha(OUT/'used-development-quality.json'),
        complete_candidate_freeze_sha256=sha(OUT/'complete-candidate-freeze.json'),queries=352,acquisitions=1292,
        learned_forward_AST_and_weights_unchanged=True,independent_insertion_order_exact=True,all_inversions_supported_by_both_heads=True,
        rank1_changes=0,membership_changes=0,formula_slot_changes=0,official_identity_duplicates=0,changed_GPU_lists=int(changed),
        exact_official_truth_vectors=True,exact_fraction_points=True,all_original_source_and_mode_guards_pass=True,
        CPU_GPU_different_candidate_lists=int(frames['cpu'].smiles.ne(frames['gpu'].smiles).sum()),passed_pair=True,
        C39_is_used_development=True,new_source_outcomes_read=False,C13_read=False)
    write(target,result);print('C49 INDEPENDENT USED DEVELOPMENT PASS',json.dumps(result))


if __name__=='__main__':main()
