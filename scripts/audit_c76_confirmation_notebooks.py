"""Check each reserved notebook against its original source, without inference."""
import base64
import hashlib
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value
from c76_confirmation_preparation import ARMS, CORE_PINS, PACKAGE, PREPARATION, SLOTS, TEMPLATES, prior
from c76_submission_contract import BINDING


def compare(original, candidate, arm, old_contract, new_contract):
    assert len(original['cells']) == len(candidate['cells'])
    changed = [i for i, (a, b) in enumerate(zip(original['cells'], candidate['cells'])) if a['source'] != b['source']]
    assert changed == [0, 6, 12, 27]
    for index in [0, 6, 27]:
        before = unwrap(''.join(original['cells'][index]['source']))
        after = unwrap(''.join(candidate['cells'][index]['source']))
        # Independently undo only the explicitly allowed count and contract edits.
        after = after.replace(new_contract, old_contract)
        after = after.replace('==120', '==512').replace('queries=120', 'queries=512').replace('unlabelled120', 'unlabelled512')
        assert after == before, index
    old_outer = unwrap(''.join(original['cells'][12]['source']))
    new_outer = unwrap(''.join(candidate['cells'][12]['source']))
    assert new_outer.startswith(old_outer)
    suffix = new_outer[len(old_outer):]
    import ast
    nodes = ast.parse(suffix).body
    assert len(nodes) == 3
    assert isinstance(nodes[0], ast.AugAssign) and isinstance(nodes[1], ast.Import) and isinstance(nodes[2], ast.Assign)
    target = nodes[0].target
    assert isinstance(target, ast.Subscript) and getattr(target.value, 'id', None) == 'MODULES'
    assert ast.literal_eval(target.slice) == 'export_evidence.py'
    export_source = ast.literal_eval(nodes[0].value)
    export_tree = ast.parse(export_source)
    assert [type(node).__name__ for node in export_tree.body] == ['Assign', 'FunctionDef']
    assert export_tree.body[1].name == 'export'
    assert export_source.count('_c76_c13_original_export(scope,root)') == 1
    assert "scope['te']" in export_source and "scope.get('ENG',{})" in export_source and "scope.get('PC',{})" in export_source
    assert 'c76_c13_channel_ids.json' in export_source
    assert 'submission.csv' not in export_source and 'smiles' not in export_source
    for outer in [old_outer, new_outer]:
        payload = base64.b64decode(value(outer, 'FROZEN_B64'), validate=True)
        assert hashlib.sha256(payload).hexdigest() == value(outer, 'FROZEN_SHA256') == CORE_PINS[arm]
    assert value(old_outer, 'FROZEN_B64') == value(new_outer, 'FROZEN_B64')
    return dict(all_other_outer_sources_exact=True, whole_scientific_inner_payload_byteexact=True,
        original_actual_T4_batch_and_R3_adapters_exact=True, independent_count_and_contract_edit_reversal_exact=True)


def main():
    prior(); target = PREPARATION / 'independent-notebook-audit.json'; assert not target.exists()
    old_contract = read(ROOT / 'kaggle/forks/c73-full-masked-validation/validation-pin.json')['query_contract_sha256']
    new_contract = sha(PACKAGE / 'validation-contract.json'); results = {}; files = []
    from c76_confirmation_preparation import DATASET
    for arm in ARMS:
        template = ROOT / 'kaggle/forks' / TEMPLATES[arm]; folder = ROOT / 'kaggle/forks' / SLOTS[arm]
        pin = read(folder / 'validation-pin.json')
        assert pin['builder_sha256'] == sha(ROOT / 'scripts/build_c76_confirmation.py')
        assert pin['notebook_sha256'] == sha(folder / 'notebook.ipynb')
        assert pin['metadata_sha256'] == sha(folder / 'kernel-metadata.json')
        assert pin['complete_pipeline_binding_sha256'] == sha(BINDING)
        results[arm] = compare(read(template / 'notebook.ipynb'), read(folder / 'notebook.ipynb'), arm, old_contract, new_contract)
        old = read(template / 'kernel-metadata.json'); new = read(folder / 'kernel-metadata.json')
        assert new['is_private'] and new['enable_gpu'] and not new['enable_internet']
        assert new['machine_shape'] == 'NvidiaTeslaT4' and new['docker_image'] == old['docker_image']
        assert new['id'] == 'clarkkitchen/casmi26-' + SLOTS[arm]
        for key in old:
            if key not in ['id', 'title', 'dataset_sources']: assert old[key] == new[key], key
        assert new['dataset_sources'] == [DATASET if name == 'clarkkitchen/casmi26-c73-full-masked-validation-20261111' else name for name in old['dataset_sources']]
        files += [folder / name for name in ['notebook.ipynb', 'kernel-metadata.json', 'validation-pin.json']]
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), passed=True,
        complete_pipeline_binding_sha256=sha(BINDING), queries=120, acquisitions=120, arms=results,
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        query_labels_read=False, scoring_entries=0))
    print('C76 RESERVED NOTEBOOKS INDEPENDENT SOURCE AUDIT PASSED', flush=True)


if __name__ == '__main__': main()
