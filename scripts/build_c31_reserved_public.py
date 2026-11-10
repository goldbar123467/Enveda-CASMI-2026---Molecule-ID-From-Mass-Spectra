"""Prepare reserved120 inference from frozen C29; never reads labels/metrics."""
import json
from datetime import datetime,timezone
import pandas as pd
from precision_cache import ROOT,sha,write
from build_c09_controls import unwrap
from build_general_anchor import guarded_source
from submit_c09_external_once import scientific_sha

PLAN=ROOT/'configs/c31-reserved-public-20261110.json'
OUT=ROOT/'outputs/local/c31-reserved-public-20261110'
FORK=ROOT/'kaggle/forks/c31-reserved-public-replay'


def main():
    OUT.mkdir(exist_ok=True);FORK.mkdir(exist_ok=True)
    assert not (FORK/'reserved-pin.json').exists(),'Existing build sealed'
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    cfg=json.loads(PLAN.read_text());template=ROOT/'kaggle/forks/c29-public-replay/notebook.ipynb'
    inputs=ROOT/'outputs/publish/c13-mixed-casmi2022-20261110';contract=inputs/'validation-contract.json'
    assert sha(template)==cfg['source_notebook_sha256'] and sha(contract)==cfg['input_contract_sha256']
    spec=json.loads(contract.read_text());assert spec['queries']==120 and not spec['truth_injection']
    assert not spec['query_labels_visible_to_pipeline']
    for name,pin in spec['files'].items():assert sha(inputs/name)==pin,name
    assert not (inputs/'labels.csv').exists()
    q=pd.read_parquet(inputs/'validation-queries.parquet')
    assert len(q)==q.molecule_id.nunique()==120 and q.ionization_mode.value_counts().to_dict()=={'positive':60,'negative':60}
    assert not {'smiles','key','inchikey14','formula'}&set(q.columns)
    nb=json.loads(template.read_text(encoding='utf-8'));before=json.loads(template.read_text(encoding='utf-8'))
    old=sha(ROOT/'outputs/publish/c29-used240-casmi2022-20261110/validation-contract.json')
    changed=[];pins=0;count_changes=0
    for i in [0,6,27,29]:
        source=unwrap(''.join(nb['cells'][i]['source']));expected=source
        pins+=source.count(old);count_changes+=source.count('==240')+source.count('queries=240')
        expected=source.replace(old,sha(contract)).replace('==240','==120').replace('queries=240','queries=120').replace('unlabelled240','unlabelled120')
        assert source!=expected,i
        compile(expected,'c31_input_count_adapter_'+str(i),'exec')
        nb['cells'][i]['source']=guarded_source(expected,'c31_reserved_'+str(i)).splitlines(keepends=True);changed.append(i)
    assert pins==2 and count_changes==7
    assert all(nb['cells'][i]['source']==before['cells'][i]['source'] for i in range(len(before['cells'])) if i not in changed)
    assert nb['cells'][12]['source']==before['cells'][12]['source']
    for c in nb['cells']:
        if c['cell_type']=='code':compile(unwrap(''.join(c['source'])),'c31_whole_build','exec');c['execution_count']=None;c['outputs']=[]
    nb['cells'].append(dict(cell_type='markdown',metadata={},source=['C31: reserved120 predictions only. All C29 recipe/source/weights unchanged; only validation contract/count differ. Labels and quality metrics remain inaccessible until C30 selects the passing C29 recipe. Existing C29 artifact names are retained inside this separate job. No public scoring submission.']))
    (FORK/'notebook.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
    meta=json.loads(template.with_name('kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-c31-reserved-public-replay',title='casmi26 c31 reserved public replay',is_private=True,enable_internet=False)
    meta['dataset_sources']=[s.replace('clarkkitchen/casmi26-c29-used240-casmi2022-20261110','clarkkitchen/casmi26-c13-mixed-casmi2022-20261110') for s in meta['dataset_sources']]
    assert not any('c29-used240' in s for s in meta['dataset_sources'])
    write(FORK/'kernel-metadata.json',meta)
    write(FORK/'reserved-pin.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(PLAN),builder_sha256=sha(__file__),
        notebook_sha256=sha(FORK/'notebook.ipynb'),scientific_sha256=scientific_sha(nb),source_notebook_sha256=sha(template),
        input_contract_sha256=sha(contract),changed_cells=changed,changed_contract_literals=pins,changed_count_literals=count_changes,
        original_scientific_payload_exact=True,no_new_recipe_model_or_threshold=True,quarantined_predictions=True,
        query_labels_mounted=False,confirmation_metrics_read=False,automatic_confirmation=False,automatic_submission=False,
        note='C29 artifact names retained inside the separate C31 job. Actual source/metadata and complete120/parent parity must be verified.'))
    print('BUILT C31 QUARANTINED RESERVED120 INFERENCE',sha(FORK/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
