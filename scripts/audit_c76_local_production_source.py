"""Reverse the exact local production wiring and verify all required inputs."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value, replace_value


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    out = base / 'local-production-preview'; pin = read(out / 'preview-freeze.json')
    target = out / 'independent-source-audit.json'; assert not target.exists()
    assert sha(ROOT / 'scripts/prepare_c76_local_production.py') == pin['source_sha256']
    assert sha(out / 'notebook.ipynb') == pin['local_notebook_sha256']
    original = read(ROOT / 'kaggle/forks/c76-cpu-production/notebook.ipynb')
    local = read(out / 'notebook.ipynb')
    assert [i for i, (a, b) in enumerate(zip(original['cells'], local['cells'])) if a != b] == [0, 12, 28]
    a = unwrap(''.join(original['cells'][12]['source']))
    b = unwrap(''.join(local['cells'][12]['source']))
    raw_original, raw_local = base64.b64decode(value(a, 'FROZEN_B64')), base64.b64decode(value(b, 'FROZEN_B64'))
    assert hashlib.sha256(raw_original).hexdigest() == pin['original_production_payload_sha256']
    assert hashlib.sha256(raw_local).hexdigest() == value(b, 'FROZEN_SHA256') == pin['local_payload_sha256']
    a_inner, b_inner = json.loads(raw_original), json.loads(raw_local)
    assert [i for i, (x, y) in enumerate(zip(a_inner['cells'], b_inner['cells'])) if x != y] == [28, 41, 43]
    restored = copy.deepcopy(b_inner)
    text = ''.join(b_inner['cells'][28]['source'])
    text = text.replace("assert comp == os.path.realpath('/kaggle/input/competitions/'+slug)",
        "assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}")
    assert text == ''.join(a_inner['cells'][28]['source'])
    restored['cells'][28] = copy.deepcopy(a_inner['cells'][28])
    text = ''.join(b_inner['cells'][41]['source'])
    assert text.count('extra_args=[\'--any-rdkit\',\'--bs\',\'8\']') == 1
    text = text.replace("extra_args=['--any-rdkit','--bs','8']", "extra_args=['--any-rdkit']")
    addition = '    gc.collect(); torch.cuda.empty_cache()  # C76 local production resource adapter\n'
    assert text.count(addition) == 1
    assert text.replace(addition, '') == ''.join(a_inner['cells'][41]['source'])
    restored['cells'][41] = copy.deepcopy(a_inner['cells'][41])
    text = ''.join(b_inner['cells'][43]['source'])
    assert text.count("'CUDA_VISIBLE_DEVICES':'0'") == 1
    assert text.replace("'CUDA_VISIBLE_DEVICES':'0'", "'CUDA_VISIBLE_DEVICES':'1'") == ''.join(a_inner['cells'][43]['source'])
    restored['cells'][43] = copy.deepcopy(a_inner['cells'][43])
    assert restored == a_inner
    b = replace_value(b, 'FROZEN_B64', value(a, 'FROZEN_B64'))
    assert replace_value(b, 'FROZEN_SHA256', value(a, 'FROZEN_SHA256')) == a
    for trial in ['cpu', 'gpu']:
        old = unwrap(''.join(read(ROOT / 'kaggle/forks' / ('c76-' + trial + '-production/notebook.ipynb'))['cells'][28]['source']))
        new = ''.join(local['cells'][28]['source']) if trial == 'cpu' else (out / 'gpu-head-stage.py').read_text(encoding='utf8')
        assert value(new, 'C76_TRIAL') == trial and value(new, 'C76_PAIR_PIN') == value(old, 'C76_PAIR_PIN')
        assert value(new, 'C76_PAYLOAD_PIN') == pin['local_payload_sha256']
        if trial == 'gpu':
            new = new.replace('c76_gpu_scores', 'c76_scores').replace('c76_gpu_pair_inference.log', 'c76_pair_inference.log').replace('c76_gpu_execution.json', 'c76_execution.json')
        new = new.replace("assert len(_devices)==1 and '2070' in _devices[0]['name']",
            "assert len(_devices)==2 and all('T4' in d['name'] for d in _devices)")
        new = replace_value(new, 'C76_PAYLOAD_PIN', value(old, 'C76_PAYLOAD_PIN'))
        def assignment(source):
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == 'C76_PAYLOAD_PIN' for t in n.targets))
            return ast.get_source_segment(source, node)
        # replace_value uses canonical assignment spacing. Restore the original
        # spelling too, so every byte outside this literal must still match.
        assert new.replace(assignment(new), assignment(old)) == old
    layout = read(out / 'input-layout-amendment.json')
    assert layout['original_preview_freeze_sha256'] == sha(out / 'preview-freeze.json')
    assert layout['source_sha256'] == sha(ROOT / 'scripts/materialize_c76_production_inputs.py')
    assert layout['file_links_sha256'] == sha(out / 'input-file-links.json')
    input_root = Path(pin['input']); links = read(out / 'input-file-links.json')
    for name, destination in links.items():
        path = input_root / name
        assert path.is_symlink() and str(path.resolve(strict=True)) == destination, name
    assert len(links) == layout['linked_files']
    assert not list(input_root.rglob('validation-contract.json'))
    bundle = ROOT / 'outputs/publish/c76-complete-direct-pair-20261112'
    found = [p for p in input_root.rglob('pair-manifest.json') if sha(p) == sha(bundle / 'pair-manifest.json')]
    assert len(found) == 1
    for name, digest in read(bundle / 'pair-manifest.json')['files_sha256'].items(): assert sha(found[0].parent / name) == digest, name
    assert list(input_root.rglob('final_model.pt'))
    competition = input_root / 'competitions/enveda-CASMI26-molecule-id-mass-spectra'
    for name, field in [('test.parquet', 'test_sha256'), ('sample_submission.csv', 'sample_sha256'), ('train.parquet', 'train_sha256')]:
        assert sha(competition / name) == pin[field], name
    fixture = next(p for p in input_root.rglob('c09e-fixture-contract.json') if sha(p) == pin['original_P16_changed_path_fixture_sha256'])
    for name, digest in read(fixture)['files'].items(): assert sha(fixture.parent / name) == digest, name
    reference = list(input_root.rglob('ext_meta.parquet')); assert len(reference) == 1
    for name, digest in read(fixture)['reference_files'].items(): assert sha(reference[0].parent / name) == digest, name
    for i, cell in enumerate(local['cells']):
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), 'audited_local_production_' + str(i), 'exec')
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        preview_freeze_sha256=sha(out / 'preview-freeze.json'),
        input_layout_amendment_sha256=sha(out / 'input-layout-amendment.json'),
        exact_original_production_source_after_reversing_hardware_resource_wiring=True,
        all_bound_head_weights_features_and_rules_exact=True, both_separate_head_stages_audited=True,
        original_preview_selection_retained=True, all_public_queries=400, preview_queries=12,
        preview_acquisitions=pin['preview_acquisitions'], required_inputs_discoverable=True,
        actual_unlabeled_test_sample_and_training_bytes_pinned=True,
        original_P16_real_changed_path_fixtures_present=True,
        actual_T4_source_and_production_evidence_replaced=False,
        quality_read=False, C13_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 LOCAL PRODUCTION SOURCE/ASSET/PREVIEW AUDIT PASSED; NOT T4 EVIDENCE', flush=True)


if __name__ == '__main__': main()
