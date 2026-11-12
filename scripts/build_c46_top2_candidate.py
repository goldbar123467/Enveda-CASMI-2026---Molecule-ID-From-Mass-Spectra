"""One development-motivated GPU rule; freeze complete source before testing it."""
import ast,copy,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from build_c45_parent_ablation import unwrap,scientific
from build_general_anchor import guarded_source
from c45_complete_parent import ROOT,OUT as C45,sha,read,write
OUT=ROOT/'outputs/local/c46-top2-protected-pair-20261112'
BUNDLE=ROOT/'outputs/publish/c46-top2-gpu-head-20261112'
POLICY=ROOT/'configs/c46-top2-protected-pair-20261112.json'


def main():
    dev=read(C45/'development-results.json');assert dev['choices']=={'cpu':'P01','gpu':None}
    assert not (C45/'source-results.json').exists() and not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert not OUT.exists() and not BUNDLE.exists() and not POLICY.exists()
    OUT.mkdir();BUNDLE.mkdir();old=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111'
    manifest=read(old/'head-manifest.json')
    for name,pin in manifest['files'].items():assert sha(old/name)==pin;shutil.copy2(old/name,BUNDLE/name)
    tail=(BUNDLE/'crossmodal_tail.py').read_text(encoding='utf8')
    assert tail.count('if i:groups.setdefault(f,[]).append(i)')==1
    tail=tail.replace('if i:groups.setdefault(f,[]).append(i)','if i>=2:groups.setdefault(f,[]).append(i)')
    tail=tail.replace('assert answer[0]==0 and set(answer)==set(range(n))','assert answer[:min(2,n)]==list(range(min(2,n))) and set(answer)==set(range(n))')
    (BUNDLE/'crossmodal_tail.py').write_text(tail,encoding='utf8',newline='\n')
    manifest.update(run_id='C46',parent_bundle_manifest_sha256=sha(old/'head-manifest.json'),
        ranking_policy='Protect original parent positions1 and2. Original0.20/offset3 same-formula RRF on positions3..25.',
        models_or_weights_changed=False,development_results_known=True,source_outcomes_read=False,C13_read=False)
    manifest['files']['crossmodal_tail.py']=sha(BUNDLE/'crossmodal_tail.py');write(BUNDLE/'head-manifest.json',manifest)
    write(BUNDLE/'dataset-metadata.json',dict(title='CASMI26 C46 fixed top2 protected GPU head',id='clarkkitchen/casmi26-c46-top2-gpu-head-20261112',licenses=[{'name':'CC0-1.0'}]))
    prior=ROOT/'kaggle/forks/p23-c45-gpu-p01-parent';nb=read(prior/'notebook.ipynb');pin=read(prior/'production-pin.json')
    source=unwrap(''.join(nb['cells'][27]['source']))
    assert source.count("C42_HEAD_PIN='"+sha(old/'head-manifest.json')+"'")==1
    source=source.replace("C42_HEAD_PIN='"+sha(old/'head-manifest.json')+"'","C42_HEAD_PIN='"+sha(BUNDLE/'head-manifest.json')+"'")
    source=source.replace("_a[0]==_b[0] and len(_a)==len(_b)","_a[:2]==_b[:2] and len(_a)==len(_b)")
    source=source.replace("parent_variant='P01',","parent_variant='P01',ranking_policy='protect_parent_top2',")
    nb['cells'][27]['source']=guarded_source(source,'c46_required_top2_gpu_stage').splitlines(keepends=True)
    folder=ROOT/'kaggle/forks/p24-c46-gpu-top2-protected';assert not folder.exists();folder.mkdir()
    (folder/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
    metadata=read(prior/'kernel-metadata.json');metadata.update(id='clarkkitchen/casmi26-p24-c46-gpu-top2-protected',title='casmi26 p24 c46 gpu top2 protected')
    metadata['dataset_sources']=[s.replace('clarkkitchen/casmi26-c42r1-crossmodal-heads-20261111','clarkkitchen/casmi26-c46-top2-gpu-head-20261112') for s in metadata['dataset_sources']]
    write(folder/'kernel-metadata.json',metadata)
    pin=copy.deepcopy(pin);pin.update(frozen_utc=datetime.now(timezone.utc).isoformat(),original_production_path=str(prior.relative_to(ROOT)),
        original_production_sha256=sha(prior/'notebook.ipynb'),notebook_sha256=sha(folder/'notebook.ipynb'),scientific_sha256=scientific(nb),
        head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),builder_sha256=sha(__file__),ranking_policy='protect_parent_top2',
        head_math_changed=True,change='Only eligible tail begins at parent position3; .20/offset3 unchanged.',
        development_results_known=True,source_outcomes_read=False,C13_read=False)
    write(folder/'production-pin.json',pin)
    policy=dict(cycle='C46',frozen_utc=datetime.now(timezone.utc).isoformat(),C45_development_sha256=sha(C45/'development-results.json'),
        rationale='GPU-alone development loses three original rank2 truths. Test one top2-protected GPU rule using development only. CPU remains the positive C45 P01 candidate. No additional model fitting or encoder inference.',
        GPU_new_recipe_candidates=1,parameter_or_checkpoint_search=False,weights_or_training_changed=False,
        complete_predictions_required_before_source_outcomes=True,development_gate='Each complete candidate strictly positive exact development MRR versus P01.',
        source_gate='Each selected fixed candidate nonnegative exact MRR versus P01 for every source and actual query mode; report all352 and all source/mode intersections with grouped uncertainty.',
        C13_policy=read(ROOT/'configs/c45-complete-parent-validation-20261112.json')['C13'],
        C13_parent='P01 for both candidates: neither production candidate includes the external-reference overlay.',
        invariants=dict(rank1_changes=0,candidate_membership_changes=0,formula_slot_changes=0,GPU_parent_top2_exact=True),
        all_queries=352,all_acquisitions=1292,absent_truths_zero=True,source_outcomes_read=False,C13_read=False,
        maximum_new_accepted_scored_entries=2,order=['cpu','observe_actual_CPU_score','gpu','observe_actual_GPU_score'],
        champion_retained=.480,paid_compute=False,hourly_campaign=False,
        limitation='Development baseline169/192 truths already rank1,1absent,22tail; this is a small exploratory effect, not an established public-score proxy.')
    write(POLICY,policy)
    family={}
    for trial,path,bundle in [('cpu',ROOT/'kaggle/forks/p22-c45-cpu-p01-parent',old),('gpu',folder,BUNDLE)]:
        p=read(path/'production-pin.json');family[trial]=dict(path=str(path.relative_to(ROOT)),kernel=read(path/'kernel-metadata.json')['id'],version=1,
            notebook_sha256=sha(path/'notebook.ipynb'),pin_sha256=sha(path/'production-pin.json'),head_manifest_sha256=sha(bundle/'head-manifest.json'),
            scientific_sha256=p['scientific_sha256'],bundle=str(bundle.relative_to(ROOT)),parent='P01')
    write(OUT/'candidate-pair-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),policy_sha256=sha(POLICY),
        builder_sha256=sha(__file__),candidates=family,source_outcomes_read=False,C13_read=False,maximum_new_scored_entries=2))
    print('C46 COMPLETE PAIR FROZEN BEFORE NEW DEVELOPMENT OR ANY SOURCE OUTCOME',json.dumps(family))


if __name__=='__main__':main()
