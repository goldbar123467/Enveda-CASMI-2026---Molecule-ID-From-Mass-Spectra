"""Adapt the completed C10 prediction-only notebook to fresh C13 queries.

Preserves scientific P01/P14 stages. Changes only run/output names, query-count
assertions and the new query-only contract pin; no label metrics or selection.
"""
import json
from pathlib import Path

from build_general_anchor import ROOT, guarded_source
from build_c09_controls import unwrap
from precision_cache import sha


def main():
    old = ROOT / 'kaggle/forks/c10-casmi2022-validation'
    folder = ROOT / 'kaggle/forks/c13-mixed-casmi2022-validation'
    assert not (folder / 'validation-pin.json').exists(), 'Existing validation is sealed'
    assert sha(old / 'notebook.ipynb') == '98a304613ba2e38a9814465e8632ddfd7bca7cc2df4af63c98b0ca09e182ca48'
    package = ROOT / 'outputs/publish/c13-mixed-casmi2022-20261110'
    contract = package / 'validation-contract.json'; spec = json.loads(contract.read_text())
    assert spec['queries'] == spec['confirmation'] == 120 and spec['development'] == 0
    assert spec['modes'] == {'positive':60,'negative':60}
    assert not spec['prediction_metrics_computed'] and not spec['query_labels_visible_to_pipeline']
    assert {p.name for p in package.iterdir()} == {'validation-contract.json','validation-queries.parquet','validation-sample.csv','dataset-metadata.json','README.md'}
    previous_pin = '2d75016513dca38ef4d4bb5f3a011c429f7f81173310e746d2e2359cfa26b463'
    notebook = json.loads((old / 'notebook.ipynb').read_text(encoding='utf-8'))
    changed, pin_replacements = [], 0
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code': continue
        original = unwrap(''.join(cell['source'])); text = original
        # Other cells contain large embedded scientific payloads. A global run
        # name replacement could alter a coincidental substring inside them.
        if i in [0,6,27]:
            pin_replacements += text.count(previous_pin)
            text = text.replace(previous_pin,sha(contract))
            text = text.replace('c10_','c13_').replace('_c10t4','_c13t4').replace('C10 ','C13 ')
            text = text.replace('==160','==120').replace('queries=160','queries=120').replace('unlabelled160','unlabelled120')
        assert previous_pin not in text
        compile(text,'c13_input_adapter_cell_'+str(i),'exec')
        if text != original:
            cell['source'] = guarded_source(text,'c13_unlabelled_'+str(i)).splitlines(keepends=True)
            changed.append(i)
        cell['execution_count'] = None; cell['outputs'] = []
    assert pin_replacements == 2 and changed == [0,6,27]
    # Explicitly ensure chart styling and all unmodified scientific cells remain
    # byte-for-byte source-identical to the completed C10 notebook.
    before = json.loads((old / 'notebook.ipynb').read_text(encoding='utf-8'))
    assert all(notebook['cells'][i]['source'] == before['cells'][i]['source'] for i in range(len(before['cells'])) if i not in changed)
    assert "'savefig.dpi': 160" in unwrap(''.join(notebook['cells'][6]['source']))
    folder.mkdir(exist_ok=True)
    (folder / 'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    metadata = json.loads((old / 'kernel-metadata.json').read_text())
    metadata.update(id='clarkkitchen/casmi26-c13-fresh-mixed-casmi2022-validation',
        title='casmi26 C13 fresh mixed CASMI2022 validation',is_private=True,enable_internet=False)
    metadata['dataset_sources'] = [x.replace('clarkkitchen/casmi26-c10-casmi2022-20261110',
        'clarkkitchen/casmi26-c13-mixed-casmi2022-20261110') for x in metadata['dataset_sources']]
    assert not any('c10-casmi2022' in x for x in metadata['dataset_sources'])
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    pin = dict(campaign='C13',notebook_sha256=sha(folder / 'notebook.ipynb'),
        validation_contract_sha256=sha(contract),builder_sha256=sha(__file__),
        template_notebook_sha256=sha(old / 'notebook.ipynb'),changed_cells=changed,
        unchanged_scientific_cells_source_exact=True,queries=120,local_confirmation=120,
        query_labels_mounted=False,metrics_computed=False,recipe_selection=False,
        private=True,internet=False,scoring_submission=False,
        limitation='Unlabelled full-pipeline inference only; actual allocation/source/privacy must be checked after push. No inference result may be called validation gain before a frozen recipe and prospective local evaluation.')
    (folder / 'validation-pin.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
    print('BUILT C13 UNLABELLED VALIDATION',json.dumps(pin),flush=True)


if __name__ == '__main__': main()
