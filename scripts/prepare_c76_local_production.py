"""Port only hardware/resource wiring for a real bounded production preview."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
import torch
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value, replace_value

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'
OUT = BASE / 'local-production-preview'
NATIVE = Path('/home/imsa/.local/share/enveda-c76/production-preview')


def once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def head_stage(trial, payload_pin):
    path = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production/notebook.ipynb')
    source = unwrap(''.join(read(path)['cells'][28]['source']))
    source = replace_value(source, 'C76_PAYLOAD_PIN', payload_pin)
    source = once(source, "assert len(_devices)==2 and all('T4' in d['name'] for d in _devices)",
        "assert len(_devices)==1 and '2070' in _devices[0]['name']")
    if trial == 'gpu':
        source = source.replace('c76_scores', 'c76_gpu_scores').replace('c76_pair_inference.log', 'c76_gpu_pair_inference.log').replace('c76_execution.json', 'c76_gpu_execution.json')
    compile(source, 'c76_local_' + trial + '_head', 'exec')
    return source


def main():
    assert not OUT.exists() and not NATIVE.exists()
    binding = BASE / 'complete-pipeline-binding.json'
    for name, pin in read(binding)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    assert read(BASE / 'cpu-head-parity/independent-comparison.json')['passed']
    assert read(BASE / 'production-inputs/input-audit.json')['official_API_download_verified']
    assert not (ROOT / 'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT / 'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    inputs = BASE / 'production-inputs'
    for name, pin in read(inputs / 'download-complete.json')['files_sha256'].items(): assert sha(inputs / name) == pin
    full = pd.read_parquet(inputs / 'test.parquet')
    assert full.molecule_id.nunique() == 400 and len(full) == 1213
    ids = sorted(full.molecule_id.unique())[:12]
    selected = full[full.molecule_id.isin(ids)]
    assert selected.molecule_id.nunique() == 12
    fixture = ROOT / 'outputs/publish/c09e-smoke-fixtures-20261109'
    fixture_contract = fixture / 'c09e-fixture-contract.json'
    assert sha(fixture_contract) == '828a4003299ffb475ef0d40c668afa4e87edddc83a37a8d835b5ddc21b4c9b61'
    for name, pin in read(fixture_contract)['files'].items(): assert sha(fixture / name) == pin
    reference = ROOT / 'outputs/research/2026-11-09-silver/extlib'
    for name, pin in read(fixture_contract)['reference_files'].items(): assert sha(reference / name) == pin
    source_path = ROOT / 'kaggle/forks/c76-cpu-production/notebook.ipynb'
    original = read(source_path); nb = copy.deepcopy(original)
    outer = unwrap(''.join(nb['cells'][12]['source']))
    original_payload = base64.b64decode(value(outer, 'FROZEN_B64'))
    inner = json.loads(original_payload); prior_inner = copy.deepcopy(inner)
    assert hashlib.sha256(original_payload).hexdigest() == value(outer, 'FROZEN_SHA256')
    text = ''.join(inner['cells'][28]['source'])
    text = once(text, "assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}",
        "assert comp == os.path.realpath('/kaggle/input/competitions/'+slug)")
    inner['cells'][28]['source'] = text.splitlines(True)
    text = ''.join(inner['cells'][41]['source'])
    text = once(text, "site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])",
        "site='/kaggle/working/ice_site', extra_args=['--any-rdkit','--bs','8'])")
    text = once(text, '    ICE_SCORES = ice_fuse.run_ice(',
        '    gc.collect(); torch.cuda.empty_cache()  # C76 local production resource adapter\n    ICE_SCORES = ice_fuse.run_ice(')
    inner['cells'][41]['source'] = text.splitlines(True)
    text = once(''.join(inner['cells'][43]['source']), "'CUDA_VISIBLE_DEVICES':'1'", "'CUDA_VISIBLE_DEVICES':'0'")
    inner['cells'][43]['source'] = text.splitlines(True)
    assert [i for i, (a, b) in enumerate(zip(prior_inner['cells'], inner['cells'])) if a != b] == [28, 41, 43]
    raw = json.dumps(inner, ensure_ascii=False, indent=1).encode(); payload_pin = hashlib.sha256(raw).hexdigest()
    outer = replace_value(outer, 'FROZEN_B64', base64.b64encode(raw).decode())
    outer = replace_value(outer, 'FROZEN_SHA256', payload_pin)
    nb['cells'][12]['source'] = outer.splitlines(True)
    allocation = "import torch,json\nfrom pathlib import Path\n_devices=[dict(index=i,name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]\nassert len(_devices)==1 and '2070' in _devices[0]['name'],'Actual local RTX2070 required'\nPath('/kaggle/working/c76_local_allocation.json').write_text(json.dumps(dict(devices=_devices,actual_single_2070=True,actual_two_T4s=False),indent=2))\n"
    nb['cells'][0]['source'] = allocation.splitlines(True)
    cpu_stage, gpu_stage = head_stage('cpu', payload_pin), head_stage('gpu', payload_pin)
    nb['cells'][28]['source'] = cpu_stage.splitlines(True)
    assert [i for i, (a, b) in enumerate(zip(original['cells'], nb['cells'])) if a != b] == [0, 12, 28]
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), 'c76_local_production_' + str(i), 'exec')
    # Check visible-test detection with the original unmodified signature rule.
    config = unwrap(''.join(nb['cells'][6]['source']))
    function = next(n for n in ast.parse(config).body if isinstance(n, ast.FunctionDef) and n.name == '_content_signature')
    import numpy as np
    namespace = dict(pd=pd, np=np, json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'original_content_signature', 'exec'), namespace)
    signature = hashlib.md5(','.join(sorted(full.spectrum_id.astype(str))).encode()).hexdigest()
    content = namespace['_content_signature'](inputs / 'test.parquet')
    assert signature == '654e030e4173780e945252b4a5adf70b' and content == 'ab75c9eebd6f3148c810f643310584e3'
    native_input = NATIVE / 'input'; working = NATIVE / 'working'
    source_input = Path('/home/imsa/.local/share/enveda-c75/fixture/input')
    assert set(p.name for p in source_input.iterdir()) == {'c75-query-only', 'competitions',
        'casmi-spectrum-model-v1', 'casmi26-attributed-v4g-reproduction-assets', 'casmi26-fold-safe-fpnet',
        'casmi26-frozen-p14-20261107', 'casmi26-natural-product-knowledge-table', 'casmi26-natural-product-panel-benchmark',
        'casmi26-pubchem-popularity-prior', 'catalogue-experiment-saner', 'v80-source'}
    train = ROOT / 'outputs/competition-raw/train.parquet'; assert train.stat().st_size == 3033286496
    train_pin = sha(train)
    OUT.mkdir(); native_input.mkdir(parents=True); working.mkdir()
    for item in source_input.iterdir():
        if item.name in ['c75-query-only', 'competitions']: continue
        (native_input / item.name).symlink_to(item.resolve(), target_is_directory=True)
    competition = native_input / 'competitions/enveda-CASMI26-molecule-id-mass-spectra'
    competition.mkdir(parents=True)
    for name, source in [('train.parquet', train), ('test.parquet', inputs / 'test.parquet'), ('sample_submission.csv', inputs / 'sample_submission.csv')]:
        (competition / name).symlink_to(source.resolve())
    for name, source in [('c76-complete-pair', ROOT / 'outputs/publish/c76-complete-direct-pair-20261112'),
        ('c09e-smoke-fixtures', fixture), ('casmi26-extlib', reference)]:
        (native_input / name).symlink_to(source.resolve(), target_is_directory=True)
    assert not list(native_input.rglob('validation-contract.json'))
    write(OUT / 'notebook.ipynb', nb)
    (OUT / 'original-production-inner.ipynb').write_bytes(original_payload)
    (OUT / 'local-production-inner.ipynb').write_bytes(raw)
    (OUT / 'gpu-head-stage.py').write_text(gpu_stage, encoding='utf8')
    selected.to_parquet(OUT / 'expected-preview-queries.parquet', index=False)
    write(OUT / 'preview-freeze.json', dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        complete_pipeline_binding_sha256=sha(binding), original_notebook_sha256=sha(source_path),
        original_production_payload_sha256=hashlib.sha256(original_payload).hexdigest(),
        local_payload_sha256=payload_pin, local_notebook_sha256=sha(OUT / 'notebook.ipynb'),
        gpu_head_stage_sha256=sha(OUT / 'gpu-head-stage.py'), changed_outer_cells=[0, 12, 28], changed_inner_cells=[28, 41, 43],
        resource_policy_sha256=sha(ROOT / 'configs/c76-local-production-preview-v1.json'),
        actual_production_input_audit_sha256=sha(inputs / 'input-audit.json'),
        test_sha256=sha(inputs / 'test.parquet'), sample_sha256=sha(inputs / 'sample_submission.csv'), train_sha256=train_pin,
        input=str(native_input), working=str(working), original_input_target=str(source_input),
        original_working_target='/home/imsa/.local/share/enveda-c76/fixture-resource/working',
        all_public_queries=400, all_public_acquisitions=1213, preview_queries=12, preview_acquisitions=len(selected),
        preview_ids=ids, expected_preview_sha256=sha(OUT / 'expected-preview-queries.parquet'),
        original_visible_preview_selection_and_all_acquisitions_exact=True,
        original_P16_changed_path_fixture_sha256=sha(fixture_contract),
        same_frozen_parent_for_both_separate_heads=True, actual_T4_evidence_replaced=False,
        exact_ICE_cross_batch_parity_assumed=False, no_new_head_fitting=True,
        quality_read=False, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 LOCAL PRODUCTION PORT FROZEN', len(selected), 'acquisitions/12 active;400 output rows; NOT T4 evidence', flush=True)


if __name__ == '__main__': main()
