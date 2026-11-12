"""Reverse and audit production wiring against the frozen C76/R3 science."""
import ast
import base64
import copy
import json
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value, replace_value

BASE = ROOT / 'outputs/local/c76-numerical-portability-20261112'


def modules(source):
    return {ast.literal_eval(t.slice): ast.literal_eval(n.value)
        for n in ast.parse(source).body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Subscript) and getattr(t.value, 'id', '') == 'MODULES'}


def same_ast(a, b):
    assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def main():
    source_nb = read(ROOT / 'kaggle/forks/c76-full-masked-validation/notebook.ipynb')
    source_outer = unwrap(''.join(source_nb['cells'][12]['source']))
    source = json.loads(base64.b64decode(value(source_outer, 'FROZEN_B64')))
    old_nb = read(ROOT / 'kaggle/forks/p30-c56-bounded-cpu/notebook.ipynb')
    old_outer = unwrap(''.join(old_nb['cells'][12]['source']))
    old = json.loads(base64.b64decode(value(old_outer, 'FROZEN_B64')))
    manifests = {}
    for trial in ['cpu', 'gpu']:
        folder = ROOT / 'kaggle/forks' / ('c76-' + trial + '-production')
        pin = read(folder / 'production-pin.json')
        assert sha(folder / 'notebook.ipynb') == pin['notebook_sha256']
        assert sha(folder / 'kernel-metadata.json') == pin['metadata_sha256']
        nb = read(folder / 'notebook.ipynb')
        assert len(nb['cells']) == 29
        assert [i for i in range(28) if nb['cells'][i] != old_nb['cells'][i]] == [0, 12]
        outer = unwrap(''.join(nb['cells'][12]['source']))
        payload = base64.b64decode(value(outer, 'FROZEN_B64'))
        assert __import__('hashlib').sha256(payload).hexdigest() == value(outer, 'FROZEN_SHA256') == pin['production_payload_sha256']
        inner = json.loads(payload)
        assert len(inner['cells']) == len(source['cells'])
        assert [i for i, (a, b) in enumerate(zip(inner['cells'], source['cells'])) if a != b] == [26, 27, 28, 32, 37]
        for i in [26, 27, 28, 37]:
            assert inner['cells'][i] == old['cells'][i]
        assert ''.join(source['cells'][26]['source']).replace("COMP = os.environ['CASMI_C08_VALIDATION_COMP']\n", '') == ''.join(inner['cells'][26]['source'])
        a = ''.join(source['cells'][28]['source'])
        b = ''.join(inner['cells'][28]['source'])
        assert a.replace("assert comp == os.path.realpath(os.environ['CASMI_C08_VALIDATION_COMP'])",
            "assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}") == b
        a = ''.join(source['cells'][27]['source'])
        for before, after in [('c41_reference_guard', 'legacy_runtime_compat'),
            ('_c41_patch_legacy', '_c42_patch_legacy'), ('_c41_compat_v3', '_c42_v3'),
            ('_c41_compat_v4', '_c42_v4'), ('c41_legacy_', 'c42_legacy_')]:
            a = a.replace(before, after)
        same_ast(a, ''.join(inner['cells'][27]['source']))
        guard = "from c41_reference_guard import install_library_guard as _c41_install_library_guard, install_engine_guard as _c41_install_engine_guard\n_c41_sid_mask_path = os.environ.get('CASMI_C41_SID_MASK')\nassert _c41_sid_mask_path, 'C41 reference mask is mandatory'\n_c41_install_library_guard(Library, np.load(_c41_sid_mask_path,allow_pickle=False))\n_c41_install_engine_guard(Engine)\n\n"
        a = ''.join(source['cells'][37]['source']); assert a.count(guard) == 1
        assert a.replace(guard, '') == ''.join(inner['cells'][37]['source'])
        new_eng = value(''.join(source['cells'][32]['source']), 'ENG_FILES')
        prod_eng = value(''.join(inner['cells'][32]['source']), 'ENG_FILES')
        old_eng = value(''.join(old['cells'][32]['source']), 'ENG_FILES')
        assert set(prod_eng) == {'pv.py', 'pv_fp.py', 'casmi_engine.py'}
        for name in ['pv.py', 'pv_fp.py']:
            assert prod_eng[name] == new_eng[name] and prod_eng[name] != old_eng[name]
        assert prod_eng['casmi_engine.py'] == old_eng['casmi_engine.py']
        guard_tree = ast.parse((ROOT / 'scripts/c41_reference_guard.py').read_text(encoding='utf8'))
        function = next(n for n in guard_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'patch_engine2_source')
        namespace = {}; exec(compile(ast.Module(body=[function], type_ignores=[]), 'unchanged_text_guard', 'exec'), namespace)
        assert namespace['patch_engine2_source'](prod_eng['casmi_engine.py']) == new_eng['casmi_engine.py']
        assert replace_value(''.join(inner['cells'][32]['source']), 'ENG_FILES', {}) == replace_value(''.join(source['cells'][32]['source']), 'ENG_FILES', {})
        # Crucially retain both new numerical GL paths and the genuine R3 recovery.
        assert inner['cells'][41] == source['cells'][41]
        assert inner['cells'][43] == source['cells'][43] and inner['cells'][43] != old['cells'][43]
        prod_modules, val_modules = modules(outer), modules(source_outer)
        assert set(prod_modules) - set(val_modules) == {'legacy_runtime_compat.py'}
        assert set(val_modules) - set(prod_modules) == {'c41_reference_guard.py'}
        for name in set(prod_modules) & set(val_modules) - {'runner.py'}:
            assert prod_modules[name] == val_modules[name], name
        runner = val_modules['runner.py'].replace("mode='external_validation',source_adaptations=['fixed_external_test_path','external_input_preflight']",
            "mode='submission',source_adaptations=[]")
        assert runner == prod_modules['runner.py']
        legacy_fn = next(n for n in ast.parse(prod_modules['legacy_runtime_compat.py']).body if isinstance(n, ast.FunctionDef))
        guard_fn = next(n for n in ast.parse(val_modules['c41_reference_guard.py']).body if isinstance(n, ast.FunctionDef) and n.name == 'patch_legacy_tree')
        assert ast.dump(legacy_fn) == ast.dump(guard_fn)
        for i in [5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 21, 26]:
            same_ast(unwrap(''.join(nb['cells'][i]['source'])), unwrap(''.join(source_nb['cells'][i]['source'])))
        assert nb['cells'][27] == old_nb['cells'][27]
        stage = unwrap(''.join(nb['cells'][28]['source']))
        assert value(stage, 'C76_TRIAL') == trial
        assert value(stage, 'C76_PAIR_PIN') == pin['pair_manifest_sha256']
        assert value(stage, 'C76_PAYLOAD_PIN') == pin['production_payload_sha256']
        assert 'score_c76_complete_pair.py' in stage and "(ROOT/'submission.csv').unlink()" in stage
        meta = read(folder / 'kernel-metadata.json')
        assert meta['is_private'] and not meta['enable_internet'] and meta['enable_gpu']
        assert meta['docker_image'] == read(ROOT / 'kaggle/forks/c76-full-masked-validation/kernel-metadata.json')['docker_image']
        assert not any('validation' in d for d in meta['dataset_sources'])
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code': compile(''.join(cell['source']), 'production_' + str(i), 'exec')
        manifests[trial] = dict(notebook_sha256=sha(folder / 'notebook.ipynb'),
            production_pin_sha256=sha(folder / 'production-pin.json'))
    write(BASE / 'production-preparation/independent-source-audit.json', dict(completed_utc=stamp(),
        source_sha256=sha(__file__), passed=True, candidates=manifests,
        exact_R3_empty_pool_recovery_retained=True,
        exact_R3_Engine2_scalar_and_fingerprint_formate_normalization_retained=True,
        exact_C76_classic_and_N1_GL_integration_retained=True,
        only_production_input_reference_wiring_and_receipt_mode_adapted=True,
        original_P16_stage_exact=True, shared_head_runtime=True,
        scoring_file_removed_before_required_head_stage=True,
        actual_production_smoke_still_required=True, quality_read=False, C13_read=False,
        scoring_entries=0, qualified_for_scoring=False))
    print('C76 BOTH PRODUCTION SOURCE PORTS AUDITED; ACTUAL SMOKE STILL REQUIRED')


if __name__ == '__main__': main()
