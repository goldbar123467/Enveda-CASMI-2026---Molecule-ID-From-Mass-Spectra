"""Freeze actual trained heads AND both full P16 production notebooks."""
import copy,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from build_c45_parent_ablation import unwrap,scientific
from build_general_anchor import guarded_source
from c45_complete_parent import ROOT,read,write,sha
OUT=ROOT/'outputs/local/c48-clean-residual-heads-20261112'
BUNDLE=ROOT/'outputs/publish/c48-clean-residual-heads-20261112'
CONFIG=ROOT/'configs/c48-clean-residual-pair-20261112.json'
OLD=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111'


def replace_once(source,old,new):
    assert source.count(old)==1,(old,source.count(old));return source.replace(old,new)


def main():
    assert not BUNDLE.exists() and not (OUT/'complete-candidate-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    worker=read(OUT/'fixed-fit-worker-contract.json');assert worker['config_sha256']==sha(CONFIG)
    for name,pin in worker['source_sha256'].items():assert sha(ROOT/'scripts'/name)==pin
    cpu=read(OUT/'cpu-ridge/complete.json');gpu=read(OUT/'gpu-residual/complete.json');boundary=read(OUT/'fit-boundary-freeze.json')
    assert cpu['model_sha256']==sha(OUT/'cpu-ridge/model.npz') and gpu['model_sha256']==sha(OUT/'gpu-residual/final_model.pt')
    assert gpu['steps']==8000 and gpu['CPU_anchor_sha256']==cpu['model_sha256']
    BUNDLE.mkdir()
    for dest,source in [('cpu-model.npz',OUT/'cpu-ridge/model.npz'),('gpu-model.pt',OUT/'gpu-residual/final_model.pt'),('fit-scaling.npz',OUT/'fit-scaling.npz'),
        ('features.py',OLD/'features.py'),('crossmodal_heads.py',OLD/'crossmodal_heads.py'),('crossmodal_tail.py',ROOT/'scripts/c48_tail.py'),
        ('c48_gpu_model.py',ROOT/'scripts/c48_gpu_model.py'),('trial-plan.json',CONFIG)]:shutil.copy2(source,BUNDLE/dest)
    original=(OLD/'score_c42_crossmodal.py').read_text(encoding='utf8');assert sha(OLD/'score_c42_crossmodal.py')=='59156c2a05005e1ef6e0b179525a24e235ef61482f5f0f03f3413925bee37cb4'
    source=replace_once(original,'from c40_gpu_model import CrossModalHead','from c48_gpu_model import BoundedResidualHead')
    source=source.replace('CrossModalHead(state[\'cfg\'])','BoundedResidualHead(state[\'cfg\'])')
    source=replace_once(source,"def score(heads,encoders,baseline,queries,output,device='cuda',only='both'):","def score(heads,encoders,baseline,queries,output,device='cuda',only='both',reuse_feature_cache=None):")
    # Cached values are descriptor/encoder outputs only, never ranking labels.
    source=replace_once(source,'    structures={}\n', '''    feature_cache=None;cached_structures={};cache_origin=None
    if reuse_feature_cache is not None:
        feature_cache=Path(reuse_feature_cache);prior=json.loads((feature_cache/'score-receipt.json').read_text())
        assert prior['source_sha256']=='59156c2a05005e1ef6e0b179525a24e235ef61482f5f0f03f3413925bee37cb4'
        assert prior['head_manifest_sha256']=='a534d8654ce8d407b6cee9b1c9b6fa7c2e8231dde84e6a232957d097fee8bed6'
        assert sha(heads/'features.py')=='a97ad9b3ab3711517864d56e497e1672cb29bb05683c72b52db9f9b0a17b1455'
        for name,pin in prior['files_sha256'].items():assert sha(feature_cache/name)==pin,name
        assert prior['query_sha256']==sha(queries) and not prior['quality_labels_read']
        cached=pd.read_parquet(feature_cache/'candidate-identities.parquet');assert cached.smiles.is_unique
        cached_structures={row.smiles:dict(key=row.key,canonical=row.canonical,formula=row.formula) for row in cached.itertuples(index=False)}
        cache_origin=dict(receipt_sha256=sha(feature_cache/'score-receipt.json'),query_features_sha256=sha(feature_cache/'query-features.npz'),
            identities_sha256=sha(feature_cache/'candidate-identities.parquet'),query_sha256=sha(queries),no_ranking_or_quality_outcome_reused=True)
        write(output/'feature-cache-origin.json',cache_origin)
    structures={}
''')
    source=replace_once(source,"        key,canonical,formula,*_=features.structure(s)\n        structures[s]=dict(key=key,canonical=canonical,formula=formula)","        if s in cached_structures:structures[s]=cached_structures[s].copy()\n        else:\n            key,canonical,formula,*_=features.structure(s)\n            structures[s]=dict(key=key,canonical=canonical,formula=formula)")
    begin=source.index('    x=np.zeros(');end=source.index('    scale=np.load(')
    raw_block=source[begin:end]
    replacement='    if feature_cache is None:\n'+''.join('    '+line if line.strip() else line for line in raw_block.splitlines(keepends=True))
    replacement+='''    else:
        cached_queries=np.load(feature_cache/'query-features.npz',allow_pickle=False)
        x=cached_queries['features'];supported=cached_queries['supported']
        coverage=pd.read_csv(feature_cache/'query-coverage.csv')
        assert coverage.molecule_id.tolist()==mids and x.shape==(len(mids),1039) and supported.shape==(len(mids),)
        assert np.array_equal(coverage.supported.to_numpy(bool),supported) and np.isfinite(x).all()
        np.savez(output/'query-features.npz',features=x,supported=supported);coverage.to_csv(output/'query-coverage.csv',index=False)
'''
    source=source[:begin]+replacement+source[end:]
    source=replace_once(source,"    y=normalize_rows(np.stack([vectors[s] for s in names])-scale['molecule_center'])", "    encoded=np.stack([vectors[s] for s in names]);np.save(output/'encoded-molecule-vectors.npy',encoded)\n    pd.DataFrame(dict(canonical=names)).to_parquet(output/'encoded-molecule-names.parquet',index=False)\n    y=normalize_rows(encoded-scale['molecule_center'])")
    source=replace_once(source,'chosen=order(formula,values,valid,bool(supported[i]))',"chosen=order(formula,values,valid,bool(supported[i]),protected_prefix=2 if trial=='gpu' else 1)")
    source=replace_once(source,"audits[trial]=dict(changed_lists=int(changed),", "audits[trial]=dict(protected_parent_prefix=2 if trial=='gpu' else 1,changed_lists=int(changed),")
    source=replace_once(source,'quality_labels_read=False,\n        files_sha256=', 'quality_labels_read=False,feature_cache_used=feature_cache is not None,feature_cache_origin=cache_origin,\n        files_sha256=')
    source=replace_once(source,"p.add_argument('--device',default='cuda');", "p.add_argument('--reuse-feature-cache');p.add_argument('--device',default='cuda');")
    source=replace_once(source,'a.output,a.device,a.only)', 'a.output,a.device,a.only,a.reuse_feature_cache)')
    compile(source,'score_c48_crossmodal.py','exec');(BUNDLE/'score_c48_crossmodal.py').write_text(source,encoding='utf8',newline='\n')
    old_manifest=read(OLD/'head-manifest.json')
    manifest=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),run_id='C48',
        files={p.name:sha(p) for p in BUNDLE.iterdir() if p.is_file()},encoder_bundle_manifest_sha256=old_manifest['encoder_bundle_manifest_sha256'],
        encoders=old_manifest['encoders'],spectrum_preprocessing=old_manifest['spectrum_preprocessing'],
        fit_boundary_sha256=sha(OUT/'fit-boundary-freeze.json'),feature_freeze_sha256=sha(OUT/'head-feature-freeze.json'),
        fitting_identities=boundary['new_fit_identities'],fitting_acquisitions=boundary['new_fit_spectrum_rows'],
        C39_quality_known_before_design=True,new_source_quality_read=False,C13_read=False,qualified_for_scoring=False,
        CPU_trial='Freshly clean-refitted ridge; complete formate-compatible P16 parent; frozen encoders.',
        GPU_trial='Fresh8000-update bounded spectral residual around same-cycle ridge; unchanged molecular geometry; complete P16 parent; top2protected.',
        serving_policy=read(CONFIG)['serving'],pretraining_exposure='New-component holdouts only; historical frozen encoder/parent exposure is not erased.')
    write(BUNDLE/'head-manifest.json',manifest)
    write(BUNDLE/'dataset-metadata.json',dict(title='CASMI26 C48 clean CPU and bounded GPU heads',id='clarkkitchen/casmi26-c48-clean-residual-heads-20261112',licenses=[{'name':'CC0-1.0'}]))
    entries={}
    for trial,old_slot,new_slot in [('cpu','p20-c42-cpu-alignment','p25-c48-clean-cpu-ridge'),('gpu','p21-c42-gpu-alignment','p26-c48-bounded-gpu')]:
        prior=ROOT/'kaggle/forks'/old_slot;nb=read(prior/'notebook.ipynb');pin=read(prior/'production-pin.json');changed=copy.deepcopy(nb)
        stage=unwrap(''.join(nb['cells'][28]['source']))
        stage=replace_once(stage,"C42_HEAD_PIN='"+sha(OLD/'head-manifest.json')+"'","C42_HEAD_PIN='"+sha(BUNDLE/'head-manifest.json')+"'")
        stage=stage.replace('score_c42_crossmodal.py','score_c48_crossmodal.py')
        prefix=1 if trial=='cpu' else 2
        stage=replace_once(stage,"C42_TRIAL='"+trial+"'","C42_TRIAL='"+trial+"'\nC48_PREFIX="+str(prefix))
        stage=replace_once(stage,"_recipe=_r['trials'][C42_TRIAL]","_recipe=_r['trials'][C42_TRIAL]\nassert _recipe['protected_parent_prefix']==C48_PREFIX")
        stage=replace_once(stage,'_a[0]==_b[0] and len(_a)==len(_b)','_a[:C48_PREFIX]==_b[:C48_PREFIX] and len(_a)==len(_b)')
        stage=replace_once(stage,"training_performed=False)","training_performed=False,cycle='C48',protected_parent_prefix=C48_PREFIX)")
        changed['cells'][28]['source']=guarded_source(stage,'c48_required_'+trial+'_stage').splitlines(keepends=True)
        assert changed['cells'][:28]==nb['cells'][:28]
        folder=ROOT/'kaggle/forks'/new_slot;assert not folder.exists();folder.mkdir()
        (folder/'notebook.ipynb').write_text(json.dumps(changed,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')
        meta=read(prior/'kernel-metadata.json');meta.update(id='clarkkitchen/casmi26-'+new_slot,title='casmi26 '+new_slot.replace('-',' '))
        meta['dataset_sources']=[s.replace('clarkkitchen/casmi26-c42r1-crossmodal-heads-20261111','clarkkitchen/casmi26-c48-clean-residual-heads-20261112') for s in meta['dataset_sources']]
        write(folder/'kernel-metadata.json',meta)
        newpin=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),trial=trial,parent_variant='P16',
            parent_notebook_sha256=pin['parent_notebook_sha256'],original_production_path=str(prior.relative_to(ROOT)),original_production_sha256=sha(prior/'notebook.ipynb'),
            notebook_sha256=sha(folder/'notebook.ipynb'),scientific_sha256=scientific(changed),head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),
            builder_sha256=sha(__file__),new_candidate_stage_cell=28,complete_parent_all_cells_exact=True,protected_parent_prefix=prefix,
            new_source_quality_read=False,C13_read=False,cloud_run_released=False,qualified_for_submission=False)
        write(folder/'production-pin.json',newpin)
        entries[trial]=dict(path=str(folder.relative_to(ROOT)),kernel=meta['id'],version=1,notebook_sha256=newpin['notebook_sha256'],scientific_sha256=newpin['scientific_sha256'],pin_sha256=sha(folder/'production-pin.json'),parent='P16')
    write(OUT/'complete-candidate-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),config_sha256=sha(CONFIG),
        candidates=entries,head_manifest_sha256=sha(BUNDLE/'head-manifest.json'),new_source_quality_read=False,C13_read=False,
        new_candidate_C39_quality_read=False,C39_old_quality_known_before_design=True,maximum_scored_entries=2))
    print('C48 COMPLETE PRODUCTION PAIR FROZEN',json.dumps(entries))


if __name__=='__main__':main()
