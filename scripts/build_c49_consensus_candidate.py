"""Freeze one agreement GPU recipe; retain the already frozen C48 CPU recipe."""
import copy,json,shutil
from datetime import datetime,timezone
from build_c48_complete_candidates import replace_once
from build_c45_parent_ablation import unwrap,scientific
from build_general_anchor import guarded_source
from c45_complete_parent import ROOT,read,write,sha
OUT=ROOT/'outputs/local/c49-consensus-pair-20261112'
OLD=ROOT/'outputs/publish/c48-clean-residual-heads-20261112'
BUNDLE=ROOT/'outputs/publish/c49-consensus-heads-20261112'
CONFIG=ROOT/'configs/c49-consensus-pair-20261112.json'


def main():
    assert not BUNDLE.exists() and not CONFIG.exists() and not (OUT/'complete-candidate-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    development=read(OUT/'used-development-rule-results.json');assert development['passed']
    commitment=read(OUT/'used-development-rule-commitment.json');assert commitment['tail_source_sha256']==sha(ROOT/'scripts/c49_consensus_tail.py')
    oldfamily=read(ROOT/'outputs/local/c48-clean-residual-heads-20261112/complete-candidate-freeze.json')
    oldmanifest=read(OLD/'head-manifest.json');assert sha(OLD/'head-manifest.json')==oldfamily['head_manifest_sha256']
    BUNDLE.mkdir()
    for name,pin in oldmanifest['files'].items():
        assert sha(OLD/name)==pin
        if name not in ['crossmodal_tail.py','score_c48_crossmodal.py','trial-plan.json']:shutil.copy2(OLD/name,BUNDLE/name)
    shutil.copy2(ROOT/'scripts/c49_consensus_tail.py',BUNDLE/'crossmodal_tail.py')
    source=(OLD/'score_c48_crossmodal.py').read_text(encoding='utf8')
    source=replace_once(source,"    for trial in trials:\n        if trial=='cpu':","    for trial in trials:\n        reference_queries=None\n        if trial=='cpu':")
    source=replace_once(source,"            del model\n        rows=[];changed=0", "            del model\n            reference_queries=normalize_rows(ridge_predict(z,np.load(heads/'cpu-model.npz')['weights']))\n        rows=[];changed=0")
    source=replace_once(source,"            chosen=order(formula,values,valid,bool(supported[i]),protected_prefix=2 if trial=='gpu' else 1)",
        "            reference_values=values if trial=='cpu' else [float(y[loc[structures[s]['canonical']]]@reference_queries[i]) if v else 0. for s,v in zip(ss,valid)]\n            chosen=order(formula,values,valid,bool(supported[i]),reference_scores=reference_values if trial=='gpu' else None)")
    source=replace_once(source,'score=values[j],eligible=valid[j])','score=values[j],reference_score=reference_values[j],eligible=valid[j])')
    source=replace_once(source,"protected_parent_prefix=2 if trial=='gpu' else 1,", "protected_parent_prefix=1,consensus_applied=trial=='gpu',")
    compile(source,'score_c49_consensus.py','exec');(BUNDLE/'score_c49_consensus.py').write_text(source,encoding='utf8',newline='\n')
    policy=dict(cycle='C49',frozen_utc=datetime.now(timezone.utc).isoformat(),C48_policy_sha256=sha(ROOT/'configs/c48-clean-residual-pair-20261112.json'),
        C48_models_and_fit_boundary_unchanged=True,C48_GPU_recipe_rejected=True,new_model_fitting=False,new_encoder_inference=False,
        rationale='Used-development causal ablation isolates one extra residual-model harm independent of top2protection. Test one global agreement rule, never an identity/source/mode-specific exception.',
        cpu='Unchanged C48 clean CPU ridge on complete formate-compatible P16 parent.',
        gpu='Same C48 clean ridge and GPU-fitted residual. Compute each fixed0.20/offset3same-formula tail RRF utility, then allow adjacent inversions only when BOTH utilities strictly prefer them. Rank1/membership/formula slots protected; ties/unsupported/missing group features retain parent.',
        development_roles='All original352queries/1292acquisitions are USED development. Keep strict original192and all352gains, and nonnegative EVERY original source and positive/mixed/negative mode; grouped intervals/intersections/absent zeros required.',
        auxiliary='Unchanged full35auxiliary queries/21chemistry groups:5remaining MassBank ESI+30public experimental APCI. No outcomes yet. Nonnegative complete-vsP01 AND incremental-vsP16 points on all35and EACH source. No post-outcome recipe/cohort revision.',
        C13='Same corrected C45 once-only jointly sealed full/incremental protocol after both prior checks and production audits. Whole231source-identity/input-alias exclusion for P16; C13 cannot establish target timsTOF calibration.',
        candidate_family=dict(cpu=1,gpu=1),source_outcomes_read=False,C13_read=False,maximum_new_accepted_entries=2,maximum_pending=1,
        order=['cpu','actual_numeric_CPU_score','gpu','actual_numeric_GPU_score'],approval_already_granted=True,paid_compute=False,hourly_automation=False,
        champion_retained=.480,statistical_promotion=False,limitation='Small uncertain adaptive-development effect. Auxiliary data are positive-only and mostly APCI, not an ESI mixed-population replacement. Public transfer is uncalibrated.')
    write(CONFIG,policy);shutil.copy2(CONFIG,BUNDLE/'trial-plan.json')
    manifest=copy.deepcopy(oldmanifest);manifest.update(frozen_utc=datetime.now(timezone.utc).isoformat(),run_id='C49',
        original_model_bundle_sha256=sha(OLD/'head-manifest.json'),CPU_and_GPU_fitted_weights_unchanged=True,
        GPU_trial=policy['gpu'],new_source_quality_read=False,C13_read=False,
        serving_policy=dict(weight=.2,offset=3,protected_parent_prefix=1,GPU_consensus=True,unsupported_exact_parent=True))
    manifest['files']={p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file()};write(BUNDLE/'head-manifest.json',manifest)
    write(BUNDLE/'dataset-metadata.json',dict(title='CASMI26 C49 fixed clean GPU agreement head',id='clarkkitchen/casmi26-c49-consensus-heads-20261112',licenses=[{'name':'CC0-1.0'}]))
    for name in ['cpu-model.npz','gpu-model.pt','fit-scaling.npz','c48_gpu_model.py','features.py','crossmodal_heads.py']:assert sha(BUNDLE/name)==sha(OLD/name)
    prior=ROOT/'kaggle/forks/p26-c48-bounded-gpu';nb=read(prior/'notebook.ipynb');pin=read(prior/'production-pin.json');stage=unwrap(''.join(nb['cells'][28]['source']))
    stage=replace_once(stage,"C42_HEAD_PIN='"+sha(OLD/'head-manifest.json')+"'","C42_HEAD_PIN='"+sha(BUNDLE/'head-manifest.json')+"'")
    stage=replace_once(stage,'C48_PREFIX=2','C48_PREFIX=1');stage=stage.replace('score_c48_crossmodal.py','score_c49_consensus.py')
    stage=replace_once(stage,"assert _recipe['protected_parent_prefix']==C48_PREFIX","assert _recipe['protected_parent_prefix']==C48_PREFIX and _recipe['consensus_applied']")
    stage=replace_once(stage,"cycle='C48'","cycle='C49'")
    changed=copy.deepcopy(nb);changed['cells'][28]['source']=guarded_source(stage,'c49_required_GPU_consensus_stage').splitlines(keepends=True)
    assert changed['cells'][:28]==nb['cells'][:28]
    folder=ROOT/'kaggle/forks/p27-c49-gpu-agreement';assert not folder.exists();folder.mkdir()
    (folder/'notebook.ipynb').write_text(json.dumps(changed,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
    metadata=read(prior/'kernel-metadata.json');metadata.update(id='clarkkitchen/casmi26-p27-c49-gpu-agreement',title='casmi26 p27 c49 gpu agreement')
    metadata['dataset_sources']=[v.replace('clarkkitchen/casmi26-c48-clean-residual-heads-20261112','clarkkitchen/casmi26-c49-consensus-heads-20261112') for v in metadata['dataset_sources']]
    write(folder/'kernel-metadata.json',metadata)
    newpin=copy.deepcopy(pin);newpin.update(frozen_utc=datetime.now(timezone.utc).isoformat(),original_production_path=str(prior.relative_to(ROOT)),
        original_production_sha256=sha(prior/'notebook.ipynb'),notebook_sha256=sha(folder/'notebook.ipynb'),scientific_sha256=scientific(changed),
        head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),builder_sha256=sha(__file__),protected_parent_prefix=1,GPU_consensus=True,
        fitted_weights_changed=False,source_quality_read=False,C13_read=False)
    write(folder/'production-pin.json',newpin)
    cpu=copy.deepcopy(oldfamily['candidates']['cpu']);cpu.update(bundle=str(OLD.relative_to(ROOT)),head_manifest_sha256=sha(OLD/'head-manifest.json'))
    gpu=dict(path=str(folder.relative_to(ROOT)),kernel=metadata['id'],version=1,notebook_sha256=newpin['notebook_sha256'],scientific_sha256=newpin['scientific_sha256'],
        pin_sha256=sha(folder/'production-pin.json'),parent='P16',bundle=str(BUNDLE.relative_to(ROOT)),head_manifest_sha256=sha(BUNDLE/'head-manifest.json'))
    write(OUT/'complete-candidate-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),policy_sha256=sha(CONFIG),
        candidates=dict(cpu=cpu,gpu=gpu),used_development_rule_sha256=sha(OUT/'used-development-rule-commitment.json'),
        used_development_results_sha256=sha(OUT/'used-development-rule-results.json'),auxiliary_panel_freeze_sha256=sha(ROOT/'outputs/local/c48-auxiliary-holdout-20261112/validation-freeze.json'),
        source_outcomes_read=False,C13_read=False,maximum_new_accepted_entries=2,CPU_and_GPU_models_same_clean_C48_fit=True))
    print('C49 COMPLETE CPU/GPU PAIR FROZEN',json.dumps(dict(cpu=cpu,gpu=gpu)))


if __name__=='__main__':main()
