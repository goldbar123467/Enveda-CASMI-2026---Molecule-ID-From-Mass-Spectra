"""Use completed C13 formate-compatible source for private C20 development inputs."""
import json
from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from build_c09_controls import unwrap


def main():
    old=ROOT/'kaggle/forks/c13-formate-recovery';out=ROOT/'kaggle/forks/c20-negative-development'
    assert not (out/'validation-pin.json').exists()
    assert sha(old/'notebook.ipynb')=='cbd4503ef7dc1ba089d33d3f3d8a45615db406b236c1b3b135005a47f7d8088f'
    package=ROOT/'outputs/publish/c20-negative-casmi2022-20261110';contract=package/'validation-contract.json'
    spec=json.loads(contract.read_text());audit=json.loads((ROOT/'outputs/local/c20-negative-development-20261110/independent-input-audit.json').read_text())
    assert spec['queries']==spec['development']==80 and spec['confirmation']==0
    assert spec['modes']=={'positive':0,'negative':80} and not spec['prediction_metrics_computed']
    assert not spec['query_labels_visible_to_pipeline'] and spec['reserved_C13_keys_excluded']
    assert audit['recognized_PV_neutral_masses']==80 and audit['reserved_C13_truth_identity_overlap']==0
    assert {p.name for p in package.iterdir()}=={'validation-contract.json','validation-queries.parquet','validation-sample.csv','dataset-metadata.json','README.md'}
    for name,pin in spec['files'].items():assert sha(package/name)==pin,name
    previous='8ef3d0600cf0dc1ca2e8f2edbe664adffc84e80e75a9e972ea50f5fdaf9554c7'
    notebook=json.loads((old/'notebook.ipynb').read_text(encoding='utf-8'));changed=[];pins=0
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']!='code':continue
        original=unwrap(''.join(cell['source']));text=original
        if i in [0,6,27]:
            pins+=text.count(previous);text=text.replace(previous,sha(contract))
            text=text.replace('c13_','c20_').replace('_c13t4','_c20t4').replace('C13 ','C20 ')
            text=text.replace('==120','==80').replace('queries=120','queries=80').replace('unlabelled120','unlabelled80')
        assert previous not in text
        compile(text,'c20_input_adapter_'+str(i),'exec')
        if text!=original:cell['source']=guarded_source(text,'c20_unlabelled_'+str(i)).splitlines(keepends=True);changed.append(i)
        cell['execution_count']=None;cell['outputs']=[]
    assert pins==2 and changed==[0,6,27]
    before=json.loads((old/'notebook.ipynb').read_text(encoding='utf-8'))
    assert all(notebook['cells'][i]['source']==before['cells'][i]['source'] for i in range(len(before['cells'])) if i not in changed)
    assert notebook['cells'][12]['source']==before['cells'][12]['source'],'Formate-compatible scientific payload must remain exact'
    out.mkdir(exist_ok=True);(out/'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((old/'kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-c20-negative-casmi2022-development',title='casmi26 C20 negative CASMI2022 development',is_private=True,enable_internet=False)
    meta['dataset_sources']=[x.replace('clarkkitchen/casmi26-c13-mixed-casmi2022-20261110','clarkkitchen/casmi26-c20-negative-casmi2022-20261110') for x in meta['dataset_sources']]
    assert not any('c13-mixed-casmi2022' in x for x in meta['dataset_sources'])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    write(out/'validation-pin.json',dict(campaign='C20',notebook_sha256=sha(out/'notebook.ipynb'),builder_sha256=sha(__file__),
        template_notebook_sha256=sha(old/'notebook.ipynb'),validation_contract_sha256=sha(contract),changed_cells=changed,
        queries=80,development=80,confirmation=0,unchanged_scientific_payload_exact=True,formate_compatibility_retained=True,
        query_labels_mounted=False,metrics_computed=False,recipe_selection=False,private=True,internet=False,
        scoring_submission=False,limitation='Private prediction-only source expansion for development; no performance or hidden-score claim. P14 is unsupported for negative spectra and must remain exact baseline fallback.'))
    print('BUILT C20 QUERY-ONLY DEVELOPMENT',sha(out/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
