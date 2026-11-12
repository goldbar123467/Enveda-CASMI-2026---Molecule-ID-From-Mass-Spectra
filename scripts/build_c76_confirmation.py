"""Adapt the original full source notebooks to the unchanged reserved inputs.

The scientific inner payload is exact. Only counts, query-contract pins and
after-the-core channel-ID instrumentation differ. This command cannot push.
"""
import argparse
import base64
import copy
import hashlib
import json
import re
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value
from build_general_anchor import guarded_source
from c76_confirmation_preparation import (ARMS, CORE_PINS, DATASET, ORIGINAL,
    PACKAGE, PREPARATION, SLOTS, TEMPLATES, prior)
from c76_submission_contract import BINDING, FAMILY

ORIGINAL_DATASET = 'clarkkitchen/casmi26-c73-full-masked-validation-20261111'
CHANNEL_EXPORT = '''
_c76_c13_original_export=export
def export(scope,root):
    _c76_c13_original_export(scope,root)
    root=Path(root)
    record=dict(input_IDs=sorted(set(scope['te'].molecule_id.astype(str))),
        Engine2_IDs=sorted(map(str,scope.get('ENG',{}))),
        PubChem_IDs=sorted(map(str,scope.get('PC',{}))),
        query_selection_or_token_rewriting=False,quality_labels_read=False)
    (root/'c76_c13_channel_ids.json').write_text(json.dumps(record,indent=2))
'''
INSTRUMENTATION = "\nMODULES['export_evidence.py'] += " + repr(CHANNEL_EXPORT) + \
    "\nimport hashlib as _c76_c13_module_hash\nMODULE_SHA256['export_evidence.py']=_c76_c13_module_hash.sha256(MODULES['export_evidence.py'].encode()).hexdigest()\n"


def adapt_notebook(original, arm, old_contract, new_contract):
    assert arm in ARMS
    assert re.fullmatch('[0-9a-f]{64}', old_contract) and re.fullmatch('[0-9a-f]{64}', new_contract)
    assert old_contract != new_contract
    nb = copy.deepcopy(original)
    outer = unwrap(''.join(original['cells'][12]['source']))
    payload = base64.b64decode(value(outer, 'FROZEN_B64'), validate=True)
    assert hashlib.sha256(payload).hexdigest() == value(outer, 'FROZEN_SHA256') == CORE_PINS[arm]
    expected_counts = {0: (0, 0, 1, 0), 6: (1, 0, 0, 1), 27: (3, 1, 0, 1)}
    for index in [0, 6, 27]:
        source = unwrap(''.join(original['cells'][index]['source']))
        actual = tuple(source.count(token) for token in ['==512', 'queries=512', 'unlabelled512', old_contract])
        assert actual == expected_counts[index], (index, actual)
        source = source.replace('==512', '==120').replace('queries=512', 'queries=120').replace('unlabelled512', 'unlabelled120')
        source = source.replace(old_contract, new_contract)
        nb['cells'][index]['source'] = guarded_source(source, 'c76_c13_' + arm + '_' + str(index)).splitlines(keepends=True)
    nb['cells'][12]['source'] = guarded_source(outer + INSTRUMENTATION, 'c76_c13_' + arm + '_export').splitlines(keepends=True)
    assert [i for i, (a, b) in enumerate(zip(original['cells'], nb['cells'])) if a['source'] != b['source']] == [0, 6, 12, 27]
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'c76_c13_' + arm + '_' + str(index), 'exec')
            cell['outputs'] = []; cell['execution_count'] = None
    return nb


def adapt_metadata(original, arm):
    meta = copy.deepcopy(original)
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert meta['machine_shape'] == 'NvidiaTeslaT4'
    assert meta['dataset_sources'].count(ORIGINAL_DATASET) == 1
    meta['dataset_sources'] = [DATASET if name == ORIGINAL_DATASET else name for name in meta['dataset_sources']]
    meta['id'] = 'clarkkitchen/casmi26-' + SLOTS[arm]
    meta['title'] = 'casmi26 C76 reserved ' + arm + ' r1'
    return meta


def build():
    prior()
    audit = read(PREPARATION / 'independent-reference-audit.json')
    assert audit['passed'] and audit['forbidden_reference_rows'] == 0
    assert audit['queries'] == audit['acquisitions'] == 120
    for name, pin in audit['artifact_sha256'].items(): assert sha(ROOT / name) == pin, name
    contract_path = PACKAGE / 'validation-contract.json'; contract = read(contract_path)
    assert contract['queries'] == contract['spectra'] == 120
    assert contract['reference_mask_required'] and not contract['query_labels_visible_to_pipeline'] and not contract['truth_injection']
    assert set(contract['files']) == {'validation-queries.parquet', 'validation-sample.csv',
        'main-reference-sid-exclusion.npy', 'engine2-raw-key-exclusion.json', 'ATTRIBUTION.txt'}
    for name, pin in contract['files'].items(): assert sha(PACKAGE / name) == pin, name
    for name in ['validation-queries.parquet', 'validation-sample.csv']:
        assert sha(PACKAGE / name) == read(ORIGINAL / 'validation-contract.json')['files'][name]
    pins = {}
    old_pin = read(ROOT / 'kaggle/forks/c73-full-masked-validation/validation-pin.json')['query_contract_sha256']
    for arm in ARMS:
        template = ROOT / 'kaggle/forks' / TEMPLATES[arm]
        folder = ROOT / 'kaggle/forks' / SLOTS[arm]
        assert not folder.exists(), 'Keep an existing or partial build for inspection'
        nb = adapt_notebook(read(template / 'notebook.ipynb'), arm, old_pin, sha(contract_path))
        metadata = adapt_metadata(read(template / 'kernel-metadata.json'), arm)
        folder.mkdir(); write(folder / 'notebook.ipynb', nb); write(folder / 'kernel-metadata.json', metadata)
        pin = dict(prepared_utc=stamp(), builder_sha256=sha(__file__), family=FAMILY, arm=arm,
            complete_pipeline_binding_sha256=sha(BINDING), notebook_sha256=sha(folder / 'notebook.ipynb'),
            metadata_sha256=sha(folder / 'kernel-metadata.json'), template_sha256=sha(template / 'notebook.ipynb'),
            query_contract_sha256=sha(contract_path), frozen_scientific_payload_sha256=CORE_PINS[arm],
            changed_outer_cells=[0, 6, 12, 27], changed_inner_cells=[],
            inner_core_and_original_actual_T4_ICE_batch_exact=True,
            instrumentation_after_scientific_core_only=True, queries=120, acquisitions=120,
            original_query_bytes_and_untrimmed_tokens_exact=True, quality_labels_read=False,
            query_truth_mapping_mounted=False, scoring_entries=0, cloud_run_released=False)
        write(folder / 'validation-pin.json', pin); pins[arm] = pin
    write(PREPARATION / 'notebook-builds.json', dict(completed_utc=stamp(), source_sha256=sha(__file__),
        family=FAMILY, builds=pins, queries=120, acquisitions=120, quality_labels_read=False, scoring_entries=0))
    print('C76 TWO RESERVED CORE NOTEBOOKS PREPARED; NOT RELEASED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.parse_args(); build()
