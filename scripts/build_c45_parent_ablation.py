"""Freeze P01-parent controls before any C39 quality; do not push or submit."""
import ast,copy,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from build_general_anchor import guarded_source

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local/c45-complete-parent-validation-20261112'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific(nb):
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    return hashlib.sha256(json.dumps(cells,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def unwrap(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='compile' and node.args and isinstance(node.args[0],ast.Constant):return node.args[0].value
    raise ValueError('Expected the existing guarded source wrapper')


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1),encoding='utf8',newline='\n')


def main():
    assert not (OUT/'candidate-family-freeze.json').exists()
    assert not (ROOT/'outputs/local/c42-fixed-c39-validation-20261111/quality-results.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    head=ROOT/'outputs/publish/c42r1-crossmodal-heads-20261111/head-manifest.json'
    assert sha(head)=='a534d8654ce8d407b6cee9b1c9b6fa7c2e8231dde84e6a232957d097fee8bed6'
    family={}
    for trial,prior,slot in [('cpu','p20-c42-cpu-alignment','p22-c45-cpu-p01-parent'),('gpu','p21-c42-gpu-alignment','p23-c45-gpu-p01-parent')]:
        original=ROOT/'kaggle/forks'/prior;nb=json.loads((original/'notebook.ipynb').read_text(encoding='utf8'))
        assert len(nb['cells'])==29
        assert 'apply_external_reference.py' in unwrap(''.join(nb['cells'][27]['source']))
        parent=copy.deepcopy(nb);parent['cells']=parent['cells'][:27]
        parent['cells'][0]['source']=['# Frozen P01 core with the tested formate source compatibility repair\n']
        parent_path=OUT/('frozen-P01-parent-'+trial+'.ipynb');write(parent_path,parent)
        changed=copy.deepcopy(parent)
        source=unwrap(''.join(nb['cells'][28]['source']))
        old="assert (ROOT/'p16_execution_receipt.json').is_file()"
        assert source.count(old)==1
        source=source.replace(old,"assert (ROOT/'execution_receipt.json').is_file()")
        parsed=ast.parse(source);assignments=[n for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='C42_PARENT_PIN' for t in n.targets)]
        assert len(assignments)==1
        lines=source.splitlines(keepends=True);node=assignments[0]
        lines[node.lineno-1:node.end_lineno]=['C42_PARENT_PIN='+repr(sha(parent_path))+'\n'];source=''.join(lines)
        old="parent_only_change='Tested legacy scalar formate alias source adapter'"
        assert source.count(old)==1
        source=source.replace(old,"parent_only_change='P01 formate-compatible parent; predeclared external-reference ablation',parent_variant='P01'")
        changed['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(source,'c45_required_head_stage').splitlines(keepends=True)))
        for i,cell in enumerate(changed['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),'c45_'+str(i),'exec')
        assert len(changed['cells'])==28
        assert all(changed['cells'][i]['source']==nb['cells'][i]['source'] for i in range(1,27))
        assert not any('apply_external_reference.py' in ''.join(c['source']) or 'p16_execution_receipt.json' in ''.join(c['source']) for c in changed['cells'])
        folder=ROOT/'kaggle/forks'/slot;folder.mkdir(exist_ok=True)
        assert not (folder/'production-pin.json').exists()
        write(folder/'notebook.ipynb',changed)
        meta=json.loads((original/'kernel-metadata.json').read_text())
        meta.update(id='clarkkitchen/casmi26-'+slot,title='casmi26 '+slot.replace('-',' '))
        meta['dataset_sources']=[s for s in meta['dataset_sources'] if s!='takumuhata/casmi26-extlib']
        write(folder/'kernel-metadata.json',meta)
        pin=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),trial=trial,parent_variant='P01',
            original_production_path=str(original.relative_to(ROOT)),original_production_sha256=sha(original/'notebook.ipynb'),
            parent_notebook_sha256=sha(parent_path),notebook_sha256=sha(folder/'notebook.ipynb'),scientific_sha256=scientific(changed),
            head_manifest_sha256=sha(head),builder_sha256=sha(__file__),new_candidate_stage_cell=27,
            removed_original_stage_cell=27,head_math_changed=False,core_source_changed=False,
            quality_metrics_read=False,C13_read=False,cloud_run_released=False,qualified_for_submission=False)
        write(folder/'production-pin.json',pin)
        family[trial]={}
        for variant,source in [('P01',folder),('P16',original)]:
            family[trial][variant]=dict(kernel='clarkkitchen/casmi26-'+source.name,version=1,path=str(source.relative_to(ROOT)),
                notebook_sha256=sha(source/'notebook.ipynb'),pin_sha256=sha(source/'production-pin.json'),
                scientific_sha256=scientific(json.loads((source/'notebook.ipynb').read_text(encoding='utf8'))))
    result=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),families=family,head_manifest_sha256=sha(head),
        selection='Per fitted head, select the eligible parent with the highest exact development192 complete-candidate MRR; ties prefer P01. Eligibility requires strictly positive development gain versus P01 and versus the candidate own parent. Freeze both choices before source outcomes; never rescue a failing source choice with the other parent.',
        development_queries=192,source_queries=160,quality_metrics_read=False,C13_read=False,models_or_weights_changed=False,
        maximum_new_accepted_scoring_entries=2,additional_cloud_runs_released=0)
    write(OUT/'candidate-family-freeze.json',result);print('C45 FOUR COMPLETE CANDIDATES FROZEN BEFORE QUALITY',json.dumps({t:{v:f['notebook_sha256'] for v,f in rows.items()} for t,rows in family.items()}))


if __name__=='__main__':main()
