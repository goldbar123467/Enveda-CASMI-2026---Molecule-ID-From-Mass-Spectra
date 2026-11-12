"""Freeze fingerprint models and complete unchanged-P16 production candidates."""
import ast,copy,shutil
from pathlib import Path
from c45_complete_parent import ROOT,read,write,sha,stamp
from build_c45_parent_ablation import unwrap,scientific
from build_general_anchor import guarded_source
from build_c48_complete_candidates import replace_once
OUT=ROOT/'outputs/local/c54-isomer-fingerprint-heads-20261112'
BUNDLE=ROOT/'outputs/publish/c54-isomer-fingerprint-heads-20261112'
CONFIG=ROOT/'configs/c54-isomer-fingerprint-pair-20261112.json'
OLD=ROOT/'outputs/publish/c48-clean-residual-heads-20261112'


def main():
    assert not BUNDLE.exists() and not (OUT/'complete-candidate-freeze.json').exists()
    assert not (ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    assert read(OUT/'fit-pair-status.json')['stage']=='both_models_fitted'
    cpu=read(OUT/'cpu-ridge/complete.json');gpu=read(OUT/'gpu-residual/complete.json');boundary=read(OUT/'fit-boundary-freeze.json')
    assert cpu['model_sha256']==sha(OUT/'cpu-ridge/model.npz') and gpu['model_sha256']==sha(OUT/'gpu-residual/final_model.pt')
    assert gpu['steps']==8000 and gpu['CPU_anchor_sha256']==cpu['model_sha256']
    assert cpu['feature_freeze_sha256']==gpu['feature_freeze_sha256']==sha(OUT/'head-feature-freeze.json')
    BUNDLE.mkdir()
    sources={'cpu-model.npz':OUT/'cpu-ridge/model.npz','gpu-model.pt':OUT/'gpu-residual/final_model.pt','fit-scaling.npz':OUT/'fit-scaling.npz',
        'features.py':OLD/'features.py','crossmodal_heads.py':OLD/'crossmodal_heads.py','crossmodal_tail.py':ROOT/'scripts/crossmodal_tail.py',
        'c54_gpu_model.py':ROOT/'scripts/c54_gpu_model.py','score_c54_fingerprints.py':ROOT/'scripts/score_c54_fingerprints.py','trial-plan.json':CONFIG}
    for name,source in sources.items():shutil.copy2(source,BUNDLE/name)
    # Same frozen acquisition preprocessing; the new target representation is
    # the deliberate learned-component change, with no input token rewrites.
    def preprocess(path):return ast.dump(next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='preprocess'))
    assert preprocess(BUNDLE/'score_c54_fingerprints.py')==preprocess(OLD/'score_c48_crossmodal.py')
    manifest=read(OLD/'head-manifest.json')
    write(BUNDLE/'head-manifest.json',dict(frozen_utc=stamp(),run_id='C54',files={p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file()},
        encoder_bundle_manifest_sha256=manifest['encoder_bundle_manifest_sha256'],encoders={'dreams.ts':manifest['encoders']['dreams.ts']},
        spectrum_preprocessing=manifest['spectrum_preprocessing'],fit_boundary_sha256=sha(OUT/'fit-boundary-freeze.json'),
        feature_freeze_sha256=sha(OUT/'head-feature-freeze.json'),fingerprint_cache_provenance_sha256=sha(OUT/'fingerprint-cache-provenance.json'),
        fitting_identities=boundary['new_fit_identities'],fitting_acquisitions=boundary['new_fit_spectrum_rows'],
        CPU_trial='Fresh clean ridge to centered unit Morgan2-2048',GPU_trial='Fresh bounded residual with explicit same-formula training negatives',
        serving_policy=read(CONFIG)['serving'],C39_and_C51_used_development=True,C53_quality_read=False,C13_read=False,qualified_for_scoring=False))
    dataset='clarkkitchen/casmi26-c54-isomer-fingerprint-heads-20261112'
    write(BUNDLE/'dataset-metadata.json',dict(title='CASMI26 C54 fixed isomer fingerprint heads',id=dataset,licenses=[{'name':'CC0-1.0'}]))
    entries={}
    for trial,oldslot,newslot in [('cpu','p25-c48-clean-cpu-ridge','p28-c54-cpu-fingerprints'),('gpu','p25-c48-clean-cpu-ridge','p29-c54-gpu-isomer-negatives')]:
        prior=ROOT/'kaggle/forks'/oldslot;nb=read(prior/'notebook.ipynb');pin=read(prior/'production-pin.json');changed=copy.deepcopy(nb)
        stage=unwrap(''.join(nb['cells'][28]['source']))
        stage=replace_once(stage,"C42_HEAD_PIN='"+sha(OLD/'head-manifest.json')+"'","C42_HEAD_PIN='"+sha(BUNDLE/'head-manifest.json')+"'")
        if trial=='gpu':stage=replace_once(stage,"C42_TRIAL='cpu'","C42_TRIAL='gpu'")
        stage=stage.replace('score_c48_crossmodal.py','score_c54_fingerprints.py').replace("cycle='C48'","cycle='C54'")
        changed['cells'][28]['source']=guarded_source(stage,'c54_required_'+trial+'_fingerprint_stage').splitlines(keepends=True)
        assert changed['cells'][:28]==nb['cells'][:28]
        folder=ROOT/'kaggle/forks'/newslot;assert not folder.exists();folder.mkdir()
        import json
        (folder/'notebook.ipynb').write_text(json.dumps(changed,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
        metadata=read(prior/'kernel-metadata.json');metadata.update(id='clarkkitchen/casmi26-'+newslot,title='casmi26 '+newslot.replace('-',' '))
        metadata['dataset_sources']=[dataset if s=='clarkkitchen/casmi26-c48-clean-residual-heads-20261112' else s for s in metadata['dataset_sources']]
        assert dataset in metadata['dataset_sources'];write(folder/'kernel-metadata.json',metadata)
        newpin=dict(frozen_utc=stamp(),trial=trial,parent_variant='P16',parent_notebook_sha256=pin['parent_notebook_sha256'],
            original_production_path=str(prior.relative_to(ROOT)),original_production_sha256=sha(prior/'notebook.ipynb'),
            notebook_sha256=sha(folder/'notebook.ipynb'),scientific_sha256=scientific(changed),head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),
            builder_sha256=sha(__file__),new_candidate_stage_cell=28,complete_parent_all_cells_exact=True,protected_parent_prefix=1,
            C53_quality_read=False,C13_read=False,cloud_run_released=False,qualified_for_submission=False)
        write(folder/'production-pin.json',newpin)
        entries[trial]=dict(path=str(folder.relative_to(ROOT)),kernel=metadata['id'],version=1,notebook_sha256=newpin['notebook_sha256'],
            scientific_sha256=newpin['scientific_sha256'],pin_sha256=sha(folder/'production-pin.json'),bundle=str(BUNDLE.relative_to(ROOT)),
            head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),parent='P16')
    write(OUT/'complete-candidate-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(CONFIG),head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),candidates=entries,
        C39_and_C51_used_development=True,C53_panel_freeze_sha256=sha(ROOT/'outputs/local/c53-target-holdout-20261112/validation-freeze.json'),
        C53_quality_read=False,C13_read=False,maximum_scored_entries=2))
    print('C54 COMPLETE PRODUCTION PAIR SEALED',entries,flush=True)


if __name__=='__main__':main()
