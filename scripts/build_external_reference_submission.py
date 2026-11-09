"""P16 smoke preparation; never submits. Pin P01 and the fixed C09E overlay."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from build_general_anchor import ROOT, guarded_source, replace_function
from build_compatible_submission import substitute


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recover-failed-smoke', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'kaggle/forks/p16-external-reference-c09'
    if (folder / 'external-pin.json').exists():
        assert args.recover_failed_smoke, 'Built version is sealed; never mutate an active smoke'
        from kaggle.api.kaggle_api_extended import KaggleApi
        from finish_compatible_campaign import status_name
        api = KaggleApi(); api.authenticate()
        assert status_name(api.kernels_status('clarkkitchen/casmi26-p16-external-reference-c09')) == 'ERROR'
        old = json.loads((folder / 'external-pin.json').read_text())
        assert sha(folder / 'notebook.ipynb') == old['notebook_sha256']
        archive = ROOT / 'outputs/research' / ('p16-failed-' + old['notebook_sha256'][:12]); archive.mkdir(parents=True, exist_ok=True)
        for name in ['notebook.ipynb', 'external-pin.json', 'kernel-metadata.json']:
            if not (archive / name).exists():
                shutil.copy2(folder / name, archive / name)
    out = ROOT / 'outputs/local/c09-external-reference-20261109'
    development = json.loads((out / 'development.json').read_text())
    assert not development['development_rejects'] and not development['confirmation_metrics_used']
    assert development['development']['all_reference']['paired']['ci95'][0] > 0
    policy = ROOT / 'configs/c09-external-reference-20261109.json'
    assert sha(policy) == development['policy_sha256']
    files = {name: (ROOT / 'scripts' / name).read_text(encoding='utf-8') for name in
             ['apply_external_reference.py', 'external_reference_overlay.py', 'c09_spectrum_audit.py',
              'reference_smoke_fixtures.py', 'reference_model_audit.py']}
    files['c09e_features.py'] = (ROOT / 'outputs/local/kit/features.py').read_text(encoding='utf-8')
    # Text literals materialize with LF on Linux. The checkout's scientific
    # source files are LF-pinned and must match this exact serving contract.
    fixture_path = ROOT / 'outputs/publish/c09e-smoke-fixtures-20261109/c09e-fixture-contract.json'
    assert fixture_path.exists()
    contract = dict(policy_sha256=sha(policy), reference_files=development['reference_sha256'],
        source_files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
        development_evidence_sha256=sha(out / 'development.json'), fixture_contract_sha256=sha(fixture_path), automatic_submission=False)
    notebook = json.loads((ROOT / 'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    swap = '''def _swap_to_ours(base_bundle):
    _pin='c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46'
    _hits=[p for p in Path('/kaggle/input').rglob('final_model.pt') if 'casmi_spectrum_model_ours_v80' in str(p) and hashlib.sha256(p.read_bytes()).hexdigest()==_pin]
    assert len(_hits)==1,'Exactly one pinned P01 v80 checkpoint required'
    _dst=ROOT/'ours_bundle';_dst.mkdir(exist_ok=True)
    for _name in ['model.py','features.py','config.json']:shutil.copy2(Path(base_bundle)/_name,_dst/_name)
    shutil.copy2(_hits[0],_dst/'final_model.pt')
    return str(_dst)'''
    replace_function(notebook, '_swap_to_ours', swap)
    ending = '''# Required final stage; fail closed instead of leaving P01's scoring filename.
assert MERGE_REPORT.get('status')=='applied'
_files=SOURCE_FILES
_code=ROOT/'c09e_code';_code.mkdir(exist_ok=True)
for _name,_text in _files.items():(_code/_name).write_text(_text,encoding='utf-8')
(_code/'c09e_policy.json').write_bytes(POLICY_BYTES)
(_code/'deployment.json').write_text(json.dumps(DEPLOYMENT,indent=2))
sys.path.insert(0,str(_code))
from reference_model_audit import verify as _verify_p01_models
_model_audit=_verify_p01_models(PUB_BUNDLE,OURS_BUNDLE,MODEL_PINS,'c08d2101154d057890de088be63834eb622468892a3b842512fdc5a65e02ff46')
_devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]
run_stage([sys.executable,_code/'apply_external_reference.py','--work',ROOT,'--input-root',INPUT,'--contract',_code/'deployment.json'],ROOT/'c09e_external.log','Required C09E measured-reference overlay')
_receipt=json.loads((ROOT/'c09e_external_receipt.json').read_text())
assert _receipt['required_stage_completed'] and _receipt['official_identity_duplicates']==0
assert _receipt['valid_rows']==len(SAMPLE_IDS) and _receipt['official_rdkit']=='2026.03.3'
if not IS_RERUN:
    assert _receipt['search']['queries']==12
    _before_fixture=hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()
    run_stage([sys.executable,_code/'reference_smoke_fixtures.py','--work',ROOT,'--input-root',INPUT,'--code',_code,'--fixture-pin',DEPLOYMENT['fixture_contract_sha256']],ROOT/'c09e_fixture.log','Required real-input C09E changed-path fixtures')
    assert hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()==_before_fixture
    _fixture=json.loads((ROOT/'p16_fixture_receipt.json').read_text())
    assert _fixture['real_queries']==30 and _fixture['changed_rows']==7 and _fixture['inserted_identities']==2
else:_fixture=None
_receipt.update(pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),merge_status=MERGE_REPORT['status'],is_hidden_rerun=IS_RERUN,devices=_devices,baseline_model_source_audit=_model_audit,changed_path_fixture=_fixture)
(ROOT/'p16_execution_receipt.json').write_text(json.dumps(_receipt,indent=2))
print('P16 REAL EXECUTION COMPLETE',json.dumps(_receipt),flush=True)
'''
    ending = substitute(ending, SOURCE_FILES=files, POLICY_BYTES=policy.read_bytes(), DEPLOYMENT=contract)
    # torch is already used in subprocesses; import explicitly in this cell.
    ending = 'import torch\n' + ending
    notebook['cells'].append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=ending.splitlines(keepends=True)))
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']); compile(source, 'p16_source', 'exec')
            cell['source'] = guarded_source(source, f'p16_{i}').splitlines(keepends=True)
            cell['outputs'] = []; cell['execution_count'] = None
        cell['id'] = hashlib.sha256((str(i)+''.join(cell['source'])).encode()).hexdigest()[:12]
    notebook['cells'].insert(0, dict(cell_type='markdown', id='p16-header', metadata={}, source=[
        '# P16: P01 with measured external-reference promotions\n',
        'Fixed public thresholds; official metric dedup; source/data hashes; complete P01 retained.\n',
        'Attribution: alexchilton external-library idea, Takumu Hata library, and upstream P01 credits below.\n',
        'One development diagnostic; new confirmation and independent real smoke audit remain required.\n']))
    notebook['nbformat_minor'] = 5; folder.mkdir(parents=True, exist_ok=True)
    (folder / 'notebook.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    metadata = json.loads((ROOT / 'kaggle/forks/p01-public480/kernel-metadata.json').read_text())
    metadata.update(id='clarkkitchen/casmi26-p16-external-reference-c09', title='casmi26 p16 external reference c09', is_private=True, machine_shape='NvidiaTeslaT4')
    metadata['dataset_sources'].append('takumuhata/casmi26-extlib')
    metadata['dataset_sources'].append('clarkkitchen/casmi26-c09e-smoke-fixtures-20261109')
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    pin = dict(campaign='C09E', notebook_sha256=sha(folder / 'notebook.ipynb'), parent_P01_sha256=sha(ROOT / 'kaggle/forks/p01-public480/notebook.ipynb'),
               **contract, scoring_submission=False, mandatory_real_smoke=True, scientific_confirmation_pending=True)
    (folder / 'external-pin.json').write_text(json.dumps(pin, indent=2), encoding='utf-8')
    print('BUILT P16; NOT SUBMITTED', json.dumps(pin), flush=True)


if __name__ == '__main__':
    main()
