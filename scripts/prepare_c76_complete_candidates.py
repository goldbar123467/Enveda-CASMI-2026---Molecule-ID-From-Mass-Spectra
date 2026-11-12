"""Prepare the fixed C76 pair and an explicit R3 production port, without release."""
import ast
import base64
import copy
import json
import shutil
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value, replace_value
from build_general_anchor import guarded_source

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
BUNDLE = ROOT / 'outputs/publish/c76-complete-direct-pair-20261112'
PAIR = ROOT / 'outputs/local/c71-final-clean-direct-pair-20261112'
CODE = ROOT / 'outputs/local/c45-complete-parent-validation-20261112/frozen-p16-code'


def eng_files(source):
    return value(source, 'ENG_FILES')


def production_stage(trial, manifest_pin, payload_pin):
    return f'''C76_TRIAL={trial!r}
C76_PAIR_PIN={manifest_pin!r}
C76_PAYLOAD_PIN={payload_pin!r}
import os,subprocess
assert FROZEN_SHA256==C76_PAYLOAD_PIN
assert MERGE_REPORT.get('status')=='applied'
assert not list(INPUT.rglob('validation-contract.json')),'Validation inputs cannot ship in production'
assert not os.environ.get('CASMI_C41_SID_MASK') and not os.environ.get('CASMI_C41_EXCLUSION_FILE')
shutil.copy2(ROOT/'submission.csv',ROOT/'c76_p16.csv')
(ROOT/'submission.csv').unlink()
_devices=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
assert len(_devices)==2 and all('T4' in d['name'] for d in _devices)
_hits=[p for p in INPUT.rglob('pair-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest()==C76_PAIR_PIN]
assert len(_hits)==1,'Exactly one frozen C76 pair bundle required'
_bundle=_hits[0].parent;_manifest=json.loads(_hits[0].read_text())
for _name,_pin in _manifest['files_sha256'].items():assert hashlib.sha256((_bundle/_name).read_bytes()).hexdigest()==_pin,_name
_refs=list(INPUT.rglob('ext_meta.parquet'));assert len(_refs)==1
with (ROOT/'c76_pair_inference.log').open('w') as _log:
    _run=subprocess.run([sys.executable,str(_bundle/'score_c76_complete_pair.py'),'--work',str(ROOT),
        '--bundle',str(_bundle),'--reference',str(_refs[0].parent),'--output',str(ROOT/'c76_scores'),
        '--trial',C76_TRIAL],stdout=_log,stderr=subprocess.STDOUT,
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
assert _run.returncode==0,'C76 required head stage failed; inspect c76_pair_inference.log'
_receipt=json.loads((ROOT/'c76_scores/score-receipt.json').read_text())
assert _receipt['pair_manifest_sha256']==C76_PAIR_PIN
assert not _receipt['quality_labels_read'] and not _receipt['training_performed']
_active=set(pd.read_parquet(ROOT/'effective_query_spectra.parquet').molecule_id)
assert _receipt['queries']==len(_active)
_recipe=_receipt['trials'][C76_TRIAL]
assert _recipe['independent_scores_and_lists_exact']
assert _recipe['candidate_membership_changes']==_recipe['formula_position_changes']==_recipe['official_identity_duplicates']==0
_answer=pd.read_csv(ROOT/'c76_scores'/(C76_TRIAL+'.csv'),dtype=str,keep_default_na=False)
assert _answer.molecule_id.tolist()==SAMPLE_IDS and _answer.molecule_id.is_unique
if not IS_RERUN:
    assert len(_active)==12 and _recipe['changed_lists']>0,'Real preview must exercise learned ranking changes'
_receipt.update(actual_devices=_devices,is_hidden_rerun=IS_RERUN,scientific_payload_sha256=FROZEN_SHA256,
    P16_execution_receipt_sha256=hashlib.sha256((ROOT/'p16_execution_receipt.json').read_bytes()).hexdigest(),
    validation_reference_masks_mounted=False,query_labels_mounted=False)
(ROOT/'c76_execution.json').write_text(json.dumps(_receipt,indent=2))
shutil.copy2(ROOT/'c76_scores'/(C76_TRIAL+'.csv'),ROOT/'submission.csv')
assert hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()==_receipt['files_sha256'][C76_TRIAL+'.csv']
print('C76 REQUIRED PRODUCTION COMPLETE',json.dumps(_receipt),flush=True)
'''


def main():
    assert not BUNDLE.exists() and not (BASE / 'complete-candidates-preparation.json').exists()
    assert read(BASE / 'fixture-resource-port/independent-run-audit.json')['C76_engineering_fixture_passed']
    assert read(BASE / 'cloud-validation-preparation/independent-source-audit.json')['passed']
    assert not (ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    for filename in ['complete-candidate-freeze.json', 'complete-pipeline-binding.json']:
        for name, pin in read(PAIR / filename)['files_sha256'].items():
            assert sha(ROOT / name.replace('\\', '/')) == pin, name
    BUNDLE.mkdir()
    sources = {'score_c76_complete_pair.py': ROOT / 'scripts/score_c76_complete_pair.py',
        'c67_ranker.py': ROOT / 'scripts/c67_ranker.py',
        'legacy_formate_compat.py': ROOT / 'scripts/legacy_formate_compat.py',
        'legacy_chem_source.py': ROOT / 'outputs/local/c37-selection-trace-20261111/pinned-public-source/chem.py',
        'identity_features.py': CODE / 'c09e_features.py',
        'cpu-model.npz': PAIR / 'cpu-model.npz', 'gpu-model.npz': PAIR / 'gpu-model.npz',
        'feature-names.json': ROOT / 'outputs/local/c69-clean-parent-evidence-oof-20261112/feature-names.json',
        'pool_popularity.csv': ROOT / 'outputs/research/pop-prior/pool_popularity.csv'}
    for name, source in sources.items():
        shutil.copyfile(source, BUNDLE / name)
        assert sha(BUNDLE / name) == sha(source)
    contract = read(CODE / 'deployment.json')
    write(BUNDLE / 'pair-manifest.json', dict(frozen_utc=stamp(), builder_sha256=sha(__file__),
        files_sha256={name: sha(BUNDLE / name) for name in sources},
        source_files_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources.values()},
        prior_C71_complete_binding_sha256=sha(PAIR / 'complete-pipeline-binding.json'),
        P16_source_files=contract['source_files'], P16_policy_sha256=contract['policy_sha256'],
        reference_policy='Original frozen production reference; controlled validation uses separately frozen C74 mask',
        ranking='Fixed C71 CPU direct and GPU bounded0.25tanh heads, exact same-formula slots and membership; no model refit',
        features=101, quality_read=False, C13_read=False, qualified_for_scoring=False))
    write(BUNDLE / 'dataset-metadata.json', dict(title='CASMI26 C76 complete direct pair',
        id='clarkkitchen/casmi26-c76-complete-direct-pair-20261112', licenses=[{'name': 'CC0-1.0'}]))
    folder = ROOT / 'kaggle/forks/c76-full-masked-validation'
    source_nb = read(folder / 'notebook.ipynb')
    source_outer = unwrap(''.join(source_nb['cells'][12]['source']))
    source_bytes = base64.b64decode(value(source_outer, 'FROZEN_B64'))
    source_inner = json.loads(source_bytes)
    old_folder = ROOT / 'kaggle/forks/p30-c56-bounded-cpu'
    old_nb = read(old_folder / 'notebook.ipynb')
    old_outer = unwrap(''.join(old_nb['cells'][12]['source']))
    old_inner = json.loads(base64.b64decode(value(old_outer, 'FROZEN_B64')))
    # Restore only production input/reference wiring. Keep the new actual R3
    # empty-pool recovery (cell43) and C76 numerical integration (cell41).
    candidate = copy.deepcopy(source_inner)
    restored_cells = [26, 27, 28, 37]
    old_eng = eng_files(''.join(old_inner['cells'][32]['source']))
    new_eng = eng_files(''.join(source_inner['cells'][32]['source']))
    # This exact helper performs source-text edits only. Extract its unchanged
    # function AST so stdlib packaging does not import the ML runtime on Windows.
    guard_tree = ast.parse((ROOT / 'scripts/c41_reference_guard.py').read_text(encoding='utf8'))
    guard_function = next(n for n in guard_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'patch_engine2_source')
    namespace = {}
    exec(compile(ast.Module(body=[guard_function], type_ignores=[]), 'pinned_guard_text_function', 'exec'), namespace)
    patch_engine2_source = namespace['patch_engine2_source']
    assert set(new_eng) == set(old_eng) | {'c41_reference_guard.py'}
    assert new_eng['casmi_engine.py'] == patch_engine2_source(old_eng['casmi_engine.py'])
    assert set(old_eng) == {'pv.py', 'pv_fp.py', 'casmi_engine.py'}
    for name in ['pv.py', 'pv_fp.py']:
        assert new_eng[name] != old_eng[name] and 'def normalize_pv_adduct(value):' in new_eng[name]
    for i in restored_cells:
        candidate['cells'][i] = copy.deepcopy(old_inner['cells'][i])
    production_eng = dict(new_eng)
    del production_eng['c41_reference_guard.py']
    production_eng['casmi_engine.py'] = old_eng['casmi_engine.py']
    engine_source = replace_value(''.join(source_inner['cells'][32]['source']), 'ENG_FILES', production_eng)
    candidate['cells'][32]['source'] = engine_source.splitlines(True)
    assert eng_files(engine_source)['pv.py'] == new_eng['pv.py']
    assert eng_files(engine_source)['pv_fp.py'] == new_eng['pv_fp.py']
    assert candidate['cells'][43] == source_inner['cells'][43] and candidate['cells'][43] != old_inner['cells'][43]
    assert candidate['cells'][41] == source_inner['cells'][41]
    raw = json.dumps(candidate, ensure_ascii=False, indent=1).encode()
    payload_pin = __import__('hashlib').sha256(raw).hexdigest()
    archive = BASE / 'production-preparation'; assert not archive.exists(); archive.mkdir()
    (archive / 'production-inner.ipynb').write_bytes(raw)
    entries = {}
    for trial in ['cpu', 'gpu']:
        notebook = copy.deepcopy(old_nb)
        notebook['cells'] = notebook['cells'][:28]
        # Early allocation check; production has no validation-only child process.
        allocation = "import torch\nassert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),'Actual dual-T4 production allocation required'\n"
        notebook['cells'][0] = dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=allocation.splitlines(True))
        outer = replace_value(old_outer, 'FROZEN_B64', base64.b64encode(raw).decode())
        outer = replace_value(outer, 'FROZEN_SHA256', payload_pin)
        notebook['cells'][12]['source'] = guarded_source(outer, 'c76_required_sources').splitlines(True)
        stage = production_stage(trial, sha(BUNDLE / 'pair-manifest.json'), payload_pin)
        notebook['cells'].append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
            source=guarded_source(stage, 'c76_required_' + trial + '_head').splitlines(True)))
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'c76_production_' + str(i), 'exec')
                cell['outputs'] = []; cell['execution_count'] = None
        dest = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
        assert not dest.exists(); dest.mkdir()
        write(dest / 'notebook.ipynb', notebook)
        meta = read(old_folder / 'kernel-metadata.json')
        meta.update(id='clarkkitchen/casmi26-c76-' + trial + '-production', title='casmi26 C76 ' + trial + ' production')
        old_dataset = 'clarkkitchen/casmi26-c56-bounded-semantic-heads-20261112'
        assert meta['dataset_sources'].count(old_dataset) == 1
        meta['dataset_sources'] = [read(BUNDLE / 'dataset-metadata.json')['id'] if d == old_dataset else d for d in meta['dataset_sources']]
        write(dest / 'kernel-metadata.json', meta)
        entry = dict(prepared_utc=stamp(), builder_sha256=sha(__file__), trial=trial,
            notebook_sha256=sha(dest / 'notebook.ipynb'), metadata_sha256=sha(dest / 'kernel-metadata.json'),
            production_payload_sha256=payload_pin, source_payload_sha256=value(source_outer, 'FROZEN_SHA256'),
            original_C71_R3_payload_sha256='3f973abbb1593e356db4f6dec149c8ad99f5afa5ded8da8bab59152bd49e60e7',
            production_input_and_reference_wiring_cells=restored_cells + [32],
            R3_Engine2_formate_normalization_retained_exact=True,
            R3_Engine2_fingerprint_adduct_normalization_retained_exact=True,
            original_R3_empty_pool_recovery_retained=True, explicit_C76_GL_integration_retained=True,
            original_P16_stage_exact=True, shared_head_runtime=True,
            pair_manifest_sha256=sha(BUNDLE / 'pair-manifest.json'),
            cloud_run_released=False, qualified_for_scoring=False, quality_read=False, C13_read=False)
        write(dest / 'production-pin.json', entry)
        entries[trial] = dict(folder=dest.relative_to(ROOT).as_posix(), **entry)
    write(BASE / 'complete-candidates-preparation.json', dict(prepared_utc=stamp(), source_sha256=sha(__file__),
        bundle=BUNDLE.relative_to(ROOT).as_posix(), pair_manifest_sha256=sha(BUNDLE / 'pair-manifest.json'),
        candidates=entries, actual_T4_source_and_production_evidence_required=True,
        independent_production_source_audit_required=True, complete_binding_still_required=True,
        new_model_fits=0, quality_read=False, C13_read=False, scoring_entries=0))
    print('C76 COMPLETE PAIR BUNDLE AND EXPLICIT R3 PRODUCTION PORT PREPARED; NOT RELEASED')


if __name__ == '__main__':
    main()
