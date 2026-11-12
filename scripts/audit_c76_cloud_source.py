"""Independently reverse the explicit C76 source integration; read no outcomes."""
import ast
import base64
import copy
import hashlib
import json
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value, replace_value


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    folder = ROOT / 'kaggle/forks/c76-full-masked-validation'
    prior_folder = ROOT / 'kaggle/forks/c73-full-masked-validation'
    pin = read(folder / 'validation-pin.json')
    for name, digest in pin['files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    old = read(prior_folder / 'notebook.ipynb')
    new = read(folder / 'notebook.ipynb')
    assert len(old['cells']) == len(new['cells'])
    assert [i for i, (a, b) in enumerate(zip(old['cells'], new['cells'])) if a != b] == [12]
    old_outer = unwrap(''.join(old['cells'][12]['source']))
    new_outer = unwrap(''.join(new['cells'][12]['source']))
    old_bytes = base64.b64decode(value(old_outer, 'FROZEN_B64'))
    new_bytes = base64.b64decode(value(new_outer, 'FROZEN_B64'))
    assert hashlib.sha256(old_bytes).hexdigest() == pin['original_payload_sha256']
    assert hashlib.sha256(new_bytes).hexdigest() == pin['candidate_payload_sha256']
    old_inner, new_inner = json.loads(old_bytes), json.loads(new_bytes)
    assert [i for i, (a, b) in enumerate(zip(old_inner['cells'], new_inner['cells'])) if a != b] == [41]
    old_source = ''.join(old_inner['cells'][41]['source'])
    new_source = ''.join(new_inner['cells'][41]['source'])
    addition = read(base / 'cloud-validation-preparation/clone-addition.txt')
    assert new_source.count(addition) == 1
    restored = new_source.replace(addition, '')
    restored = replace_value(restored, 'N1_GL_RUNNER_SHA256', value(old_source, 'N1_GL_RUNNER_SHA256'))
    assert restored == old_source
    restored_inner = copy.deepcopy(new_inner)
    restored_inner['cells'][41] = copy.deepcopy(old_inner['cells'][41])
    assert restored_inner == old_inner
    restored_outer = replace_value(new_outer, 'FROZEN_B64', value(old_outer, 'FROZEN_B64'))
    restored_outer = replace_value(restored_outer, 'FROZEN_SHA256', value(old_outer, 'FROZEN_SHA256'))
    assert restored_outer == old_outer
    # Parse the materialization calls rather than invoking the generating builder.
    calls = [node for node in ast.walk(ast.parse(new_source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_bytes'
             and isinstance(node.func.value, ast.BinOp)]
    embedded = {}
    for call in calls:
        name = ast.literal_eval(call.func.value.right)
        if name not in ['gl_runner.py', 'c76_gl_portability.py']:
            continue
        data = base64.b64decode(ast.literal_eval(call.args[0].args[0]))
        compile(data, name, 'exec')
        embedded[name] = hashlib.sha256(data).hexdigest()
    assert embedded == {'gl_runner.py': pin['candidate_GL_runner_sha256'],
                        'c76_gl_portability.py': pin['helper_sha256']}
    assert embedded['c76_gl_portability.py'] == sha(ROOT / 'scripts/c76_gl_portability.py')
    assert embedded['gl_runner.py'] == sha(base / 'cloud-validation-preparation/pinned-gl_runner.py')
    a = read(prior_folder / 'kernel-metadata.json')
    b = read(folder / 'kernel-metadata.json')
    assert {key for key in set(a) | set(b) if a.get(key) != b.get(key)} == {'id', 'title'}
    assert b['is_private'] and b['enable_gpu']
    for i, cell in enumerate(new['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'c76_outer_' + str(i), 'exec')
    result = dict(completed_utc=stamp(), source_sha256=sha(__file__),
        validation_pin_sha256=sha(folder / 'validation-pin.json'),
        passed=True, outer_change_only_embedded_payload=True, inner_changed_cells=[41],
        exact_original_source_after_reversing_GL_integration=True,
        embedded_loader_and_helper_sha256=embedded,
        metadata_changes_only=['id', 'title'], original_datasets_image_and_machine_retained=True,
        original_ICE_batch_default_retained=True, no_local_resource_adapter=True,
        original_C73_preserved_as_separate_legacy_control=True,
        queries=512, acquisitions=2215, quality_read=False, C13_read=False,
        scoring_entries=0, qualified_for_scoring=False)
    write(base / 'cloud-validation-preparation/independent-source-audit.json', result)
    print('C76 CLOUD SOURCE AUDIT PASSED; NOT A QUALITY OR SCORING QUALIFICATION')


if __name__ == '__main__':
    main()
