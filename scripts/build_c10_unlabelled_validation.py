"""Build fresh C10 P01/P14 inference without mounting or reading query labels."""
import hashlib
import json
from pathlib import Path

from build_general_anchor import ROOT, guarded_source
from build_c09_controls import unwrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = ROOT / 'kaggle/forks/c10-casmi2022-validation'
    assert not (folder / 'validation-pin.json').exists(), 'Existing validation is sealed'
    package = ROOT / 'outputs/publish/c10-casmi2022-20261110'
    contract = package / 'validation-contract.json'
    spec = json.loads(contract.read_text()); count = spec['queries']
    assert count == 160 and spec['development'] == 40 and spec['confirmation'] == 120
    assert not (package / 'labels.csv').exists() and not spec['query_labels_visible_to_pipeline']
    template = (ROOT / 'scripts/build_compatible_validation.py').read_text(encoding='utf-8')
    for before, after in {
        'c08-validation-20261108': 'c10-casmi2022-20261110',
        'c08-external-validation': 'c10-casmi2022-validation',
        'casmi26-c08-novel-external-validation': 'casmi26-c10-casmi2022-validation',
        'casmi26 C08 novel external validation': 'casmi26 C10 prospective CASMI2022 validation',
        '==128': '==160', 'queries=128': 'queries=160'
    }.items():
        template = template.replace(before, after)
    exec(compile(template, 'c10_trusted_input_adapter_builder', 'exec'), dict(__name__='__main__', __file__=str(ROOT / 'scripts/build_compatible_validation.py')))
    notebook = json.loads((folder / 'notebook.ipynb').read_text(encoding='utf-8'))
    kept, removed = [], 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'DONE C08 FULL PIPELINE VALIDATION' in unwrap(''.join(cell['source'])):
            removed += 1
        else:
            kept.append(cell)
    assert removed == 1, 'Remove the historical label-reading evaluation cell exactly once'
    notebook['cells'] = kept
    p01 = json.loads((ROOT / 'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    replay = [''.join(p01['cells'][i]['source']) for i in [15, 17, 18]]
    current = json.loads((ROOT / 'kaggle/forks/c09-clean-validation/notebook.ipynb').read_text(encoding='utf-8'))
    preamble = unwrap(''.join(current['cells'][0]['source']))
    assert '_c09t4_child=subprocess.Popen' in preamble
    preamble = preamble.replace('c09', 'c10').replace('C09', 'C10').replace('clean80', 'unlabelled160')
    notebook['cells'].insert(0, dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
        source=guarded_source(preamble, 'c10_actual_t4_lanes').splitlines(keepends=True), id='c10-t4-lanes'))
    manifest = ROOT / 'outputs/publish/frozen-p14-20261107/frozen-manifest.json'
    driver = (ROOT / 'scripts/score_frozen_channel.py').read_text(encoding='utf-8')
    ending = '''# No query labels are mounted; this run exports predictions only.
assert IS_RERUN and MERGE_REPORT.get('status')=='applied'
assert json.loads((ROOT/'execution_receipt.json').read_text())['processed_molecules']==160
_c10_baseline=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
assert len(_c10_baseline)==160 and _c10_baseline.molecule_id.is_unique
assert set(_c10_baseline.molecule_id)==set(SAMPLE_IDS)
assert not (C08_VALIDATION_SOURCE/'labels.csv').exists()
for _file in ['neural_public.parquet','neural_ours.parquet','neural_candidate_features.parquet']:
    _frame=pd.read_parquet(ROOT/_file)
    assert _frame.neural_ok.astype(bool).all(),'Full-query neural scoring fell back'
    assert _frame.molecule_id.nunique()==160
    assert np.isfinite(_frame.neural_score.to_numpy()).all()
shutil.copy2(ROOT/'submission.csv',ROOT/'c10_p01.csv')
shutil.copy2(ROOT/'neural_candidate_features.parquet',ROOT/'c10_neural_p01.parquet')
_c10_replay_sources=REPLAY_SOURCES
def _c10_replay():
    global CONSENSUS_OK,REPORT,MOVES,SUBMITTED
    CONSENSUS_OK,REPORT,MOVES=build_consensus();assert CONSENSUS_OK
    SUBMITTED=choose_and_submit();assert SUBMITTED==VARIANT
    shutil.copy2(ROOT/'submission.csv',ROOT/'submission_control_V2.csv')
    for _i,_source in enumerate(_c10_replay_sources):exec(compile(_source,'c10_post_'+str(_i),'exec'),globals())
    assert MERGE_REPORT.get('status')=='applied'
    return pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
assert _c10_replay().equals(_c10_baseline),'Unchanged complete P01 replay differs'
_hits=list(INPUT.rglob('frozen-manifest.json'));assert len(_hits)==1
assert hashlib.sha256(_hits[0].read_bytes()).hexdigest()==FROZEN_PIN
(CODE/'score_frozen_channel.py').write_text(P14_DRIVER,encoding='utf-8')
assert _c10t4_child.wait(timeout=300)==0,'GPU1 encoder preflight failed'
_c10t4_log.close()
os.environ['CUDA_VISIBLE_DEVICES']='1'
try:
    run_stage([sys.executable,CODE/'score_frozen_channel.py',_hits[0].parent,PUB_BUNDLE,ROOT/'c10_neural_p01.parquet',ROOT/'effective_query_spectra.parquet',ROOT/'neural_candidate_features.parquet','cuda'],ROOT/'c10_p14.log','C10 unlabelled P14 negative control')
finally:
    os.environ['CUDA_VISIBLE_DEVICES']='0'
_c10_receipt=json.loads((ROOT/'neural_candidate_features.parquet.frozen.json').read_text())
assert _c10_receipt['input_errors']==0
_c10_replay().to_csv(ROOT/'c10_p14.csv',index=False)
shutil.copy2(ROOT/'effective_query_spectra.parquet',ROOT/'c10_queries.parquet')
_c10_report=dict(kind='prospective_unlabelled_full_pipeline_inference',queries=160,baseline_replay_exact=True,
    query_labels_mounted=False,metrics_computed=False,recipe_selection=False,scoring_submission=False,
    validation_contract_sha256=CONTRACT_PIN,p14_receipt=_c10_receipt,
    baseline_sha256=hashlib.sha256((ROOT/'c10_p01.csv').read_bytes()).hexdigest(),
    limitation='Predictions only. Local development-only recipe freeze must precede reserved evaluation; no hidden-score or universal-pretraining-exclusion claim.')
(ROOT/'c10_report.json').write_text(json.dumps(_c10_report,indent=2))
(ROOT/'submission.csv').unlink()
print('DONE C10 UNLABELLED INFERENCE',json.dumps(_c10_report),flush=True)
'''.replace('REPLAY_SOURCES', repr(replay)).replace('P14_DRIVER', repr(driver)).replace('FROZEN_PIN', repr(sha(manifest))).replace('CONTRACT_PIN', repr(sha(contract)))
    assert 'labels.csv' not in ending.replace("(C08_VALIDATION_SOURCE/'labels.csv').exists()", '')
    compile(ending, 'c10_unlabelled_ending', 'exec')
    notebook['cells'].append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
        source=guarded_source(ending, 'c10_unlabelled_exports').splitlines(keepends=True), id='c10-unlabelled-ending'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), 'c10_cell', 'exec')
    path = folder / 'notebook.ipynb'; path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    metadata = json.loads((ROOT / 'kaggle/forks/c09-clean-validation/kernel-metadata.json').read_text())
    metadata.update(id='clarkkitchen/casmi26-c10-unlabelled-casmi2022-validation', title='casmi26 C10 unlabelled CASMI2022 validation', is_private=True)
    metadata['dataset_sources'] = [source.replace('clarkkitchen/casmi26-c09-validation-20261109', 'clarkkitchen/casmi26-c10-casmi2022-20261110') for source in metadata['dataset_sources']]
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    pin = dict(campaign='C10', notebook_sha256=sha(path), validation_contract_sha256=sha(contract),
        original_P01_sha256=sha(ROOT / 'kaggle/forks/p01-public480/notebook.ipynb'),
        input_adapter_builder_sha256=sha(ROOT / 'scripts/build_compatible_validation.py'),
        builder_sha256=sha(__file__), frozen_manifest_sha256=sha(manifest), p14_driver_sha256=sha(ROOT / 'scripts/score_frozen_channel.py'),
        queries=160, local_development=40, local_confirmation=120, query_labels_mounted=False,
        removed_historical_label_evaluation_cells=1, private=True, internet=False, scoring_submission=False)
    (folder / 'validation-pin.json').write_text(json.dumps(pin, indent=2), encoding='utf-8')
    print('BUILT C10 UNLABELLED VALIDATION', json.dumps(pin), flush=True)


if __name__ == '__main__':
    main()
