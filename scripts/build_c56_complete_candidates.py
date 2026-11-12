"""Seal clean refitted bounded semantic candidates; proxy weights never ship."""
import copy,shutil,json
from c45_complete_parent import ROOT,read,write,sha,stamp
from build_c45_parent_ablation import unwrap,scientific
from build_general_anchor import guarded_source
from build_c48_complete_candidates import replace_once
OUT=ROOT/'outputs/local/c56-bounded-semantic-heads-20261112'
BUNDLE=ROOT/'outputs/publish/c56-bounded-semantic-heads-20261112'
OLD=ROOT/'outputs/publish/c49-consensus-heads-20261112'
CONFIG=ROOT/'configs/c56-bounded-semantic-pair-20261112.json'


def main():
    assert not BUNDLE.exists() and not (OUT/'complete-candidate-freeze.json').exists()
    assert read(OUT/'fit-pair-status.json')['stage']=='both_models_fitted'
    assert not (ROOT/'outputs/local/c53-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    cpu=read(OUT/'cpu-ridge/complete.json');gpu=read(OUT/'gpu-residual/complete.json')
    assert cpu['model_sha256']==sha(OUT/'cpu-ridge/model.npz') and gpu['model_sha256']==sha(OUT/'gpu-residual/final_model.pt')
    assert gpu['CPU_anchor_sha256']==cpu['model_sha256'] and gpu['steps']==8000
    assert cpu['model_sha256']!=sha(OLD/'cpu-model.npz') and gpu['model_sha256']!=sha(OLD/'gpu-model.pt')
    BUNDLE.mkdir()
    for name,source in {'cpu-model.npz':OUT/'cpu-ridge/model.npz','gpu-model.pt':OUT/'gpu-residual/final_model.pt','fit-scaling.npz':OUT/'fit-scaling.npz',
        'crossmodal_tail.py':ROOT/'scripts/c56_bounded_consensus_tail.py','trial-plan.json':CONFIG,
        **{n:OLD/n for n in ['features.py','crossmodal_heads.py','c48_gpu_model.py']}}.items():shutil.copy2(source,BUNDLE/name)
    source=(OLD/'score_c49_consensus.py').read_text(encoding='utf8')
    source=replace_once(source,"assert prior['source_sha256']=='59156c2a05005e1ef6e0b179525a24e235ef61482f5f0f03f3413925bee37cb4'", "assert prior['source_sha256'] in {'59156c2a05005e1ef6e0b179525a24e235ef61482f5f0f03f3413925bee37cb4','60707e1a7406bbe7884cae81eaa3e981214982fc63fdb66b0ade0c490d91c652','040cf0a124709de562cc394b90b452d499362af62715c4c6e39e528f6afd86e7'}")
    source=replace_once(source,"assert prior['head_manifest_sha256']=='a534d8654ce8d407b6cee9b1c9b6fa7c2e8231dde84e6a232957d097fee8bed6'", "assert prior['head_manifest_sha256'] in {'a534d8654ce8d407b6cee9b1c9b6fa7c2e8231dde84e6a232957d097fee8bed6','7095842c42cdfb81149f3bc448a65bd62a0877ebb9bbdb56a3f41a18747e4f78','7afc1bb6378b18da9683b94511af5a1b1a7468d452fab6c28bd7a189efc01782'}")
    source=replace_once(source,"    unknown=[s for s in canonical if eligibility[s] and s not in vectors]", """    if feature_cache is not None and (feature_cache/'encoded-molecule-vectors.npy').exists():
        assert 'encoded-molecule-vectors.npy' in prior['files_sha256'] and 'encoded-molecule-names.parquet' in prior['files_sha256']
        cached_names=pd.read_parquet(feature_cache/'encoded-molecule-names.parquet').canonical.tolist()
        cached_vectors=np.load(feature_cache/'encoded-molecule-vectors.npy',mmap_mode='r');assert cached_vectors.shape==(len(cached_names),768)
        cached_lookup={s:i for i,s in enumerate(cached_names)}
        for s in canonical:
            if not eligibility[s] or s not in cached_lookup:continue
            value=np.asarray(cached_vectors[cached_lookup[s]],np.float32);assert np.isfinite(value).all()
            if s in vectors:np.testing.assert_array_equal(vectors[s],value)
            vectors[s]=value
    unknown=[s for s in canonical if eligibility[s] and s not in vectors]""")
    source=replace_once(source,"protected_parent_prefix=1,consensus_applied=trial=='gpu'", "protected_parent_prefix=2,last_reorderable_parent_rank=10,consensus_applied=trial=='gpu'")
    compile(source,'score_c56_bounded_semantic.py','exec');(BUNDLE/'score_c56_bounded_semantic.py').write_text(source,encoding='utf8',newline='\n')
    original=read(OLD/'head-manifest.json');boundary=read(OUT/'fit-boundary-freeze.json')
    manifest=dict(frozen_utc=stamp(),run_id='C56',files={p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file()},
        encoder_bundle_manifest_sha256=original['encoder_bundle_manifest_sha256'],encoders=original['encoders'],spectrum_preprocessing=original['spectrum_preprocessing'],
        fit_boundary_sha256=sha(OUT/'fit-boundary-freeze.json'),feature_freeze_sha256=sha(OUT/'head-feature-freeze.json'),
        fitting_identities=boundary['new_fit_identities'],fitting_acquisitions=boundary['new_fit_spectrum_rows'],
        serving_policy=read(CONFIG)['serving'],C39_C51_used_development=True,C53_quality_read=False,C13_read=False,
        fresh_CPU_and_GPU_models=True,old_proxy_weights_not_used=True,qualified_for_scoring=False)
    write(BUNDLE/'head-manifest.json',manifest);dataset='clarkkitchen/casmi26-c56-bounded-semantic-heads-20261112'
    write(BUNDLE/'dataset-metadata.json',dict(title='CASMI26 C56 clean bounded semantic heads',id=dataset,licenses=[{'name':'CC0-1.0'}]))
    entries={}
    for trial,slot,newslot in [('cpu','p25-c48-clean-cpu-ridge','p30-c56-bounded-cpu'),('gpu','p27-c49-gpu-agreement','p31-c56-bounded-gpu')]:
        folder=ROOT/'kaggle/forks'/slot;nb=read(folder/'notebook.ipynb');pin=read(folder/'production-pin.json');stage=unwrap(''.join(nb['cells'][28]['source']))
        stage=replace_once(stage,"C42_HEAD_PIN='"+pin['head_manifest_sha256']+"'","C42_HEAD_PIN='"+sha(BUNDLE/'head-manifest.json')+"'")
        stage=stage.replace('score_c48_crossmodal.py','score_c56_bounded_semantic.py').replace('score_c49_consensus.py','score_c56_bounded_semantic.py')
        stage=replace_once(stage,'C48_PREFIX=1','C48_PREFIX=2');stage=stage.replace("cycle='C48'","cycle='C56'").replace("cycle='C49'","cycle='C56'")
        stage=replace_once(stage,'_a[:C48_PREFIX]==_b[:C48_PREFIX] and len(_a)==len(_b)','_a[:C48_PREFIX]==_b[:C48_PREFIX] and _a[10:]==_b[10:] and len(_a)==len(_b)')
        stage=replace_once(stage,"_recipe=_r['trials'][C42_TRIAL]","_recipe=_r['trials'][C42_TRIAL]\nassert _recipe['last_reorderable_parent_rank']==10")
        changed=copy.deepcopy(nb);changed['cells'][28]['source']=guarded_source(stage,'c56_required_'+trial+'_bounded_semantic').splitlines(keepends=True)
        assert changed['cells'][:28]==nb['cells'][:28]
        dest=ROOT/'kaggle/forks'/newslot;assert not dest.exists();dest.mkdir();(dest/'notebook.ipynb').write_text(json.dumps(changed,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
        metadata=read(folder/'kernel-metadata.json');metadata.update(id='clarkkitchen/casmi26-'+newslot,title='casmi26 '+newslot.replace('-',' '))
        metadata['dataset_sources']=[dataset if v in {'clarkkitchen/casmi26-c48-clean-residual-heads-20261112','clarkkitchen/casmi26-c49-consensus-heads-20261112'} else v for v in metadata['dataset_sources']]
        assert dataset in metadata['dataset_sources'];write(dest/'kernel-metadata.json',metadata)
        newpin=dict(frozen_utc=stamp(),trial=trial,parent_variant='P16',parent_notebook_sha256=pin['parent_notebook_sha256'],
            original_production_path=str(folder.relative_to(ROOT)),original_production_sha256=sha(folder/'notebook.ipynb'),
            notebook_sha256=sha(dest/'notebook.ipynb'),scientific_sha256=scientific(changed),head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),
            builder_sha256=sha(__file__),new_candidate_stage_cell=28,complete_parent_all_cells_exact=True,protected_parent_prefix=2,last_reorderable_parent_rank=10,
            C53_quality_read=False,C13_read=False,cloud_run_released=False,qualified_for_submission=False)
        write(dest/'production-pin.json',newpin)
        entries[trial]=dict(path=str(dest.relative_to(ROOT)),kernel=metadata['id'],version=1,notebook_sha256=newpin['notebook_sha256'],
            scientific_sha256=newpin['scientific_sha256'],pin_sha256=sha(dest/'production-pin.json'),bundle=str(BUNDLE.relative_to(ROOT)),head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),parent='P16')
    write(OUT/'complete-candidate-freeze.json',dict(frozen_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(CONFIG),
        head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),candidates=entries,C39_and_C51_used_development=True,
        C53_panel_freeze_sha256=sha(ROOT/'outputs/local/c53-target-holdout-20261112/validation-freeze.json'),C53_quality_read=False,C13_read=False,maximum_scored_entries=2))
    print('C56 CLEAN COMPLETE PAIR SEALED',entries,flush=True)


if __name__=='__main__':main()
