"""Recover missing Engine2 catalogues from already-generated Engine1 proposals."""
import ast,base64,hashlib,json
from precision_cache import ROOT,sha,write
from build_c09_controls import unwrap
from build_general_anchor import guarded_source
from build_c13_formate_recovery import value,replace_value


def main():
    old=ROOT/'kaggle/forks/c20-negative-development';out=ROOT/'kaggle/forks/c20-generated-recovery'
    assert not (out/'recovery-pin.json').exists()
    assert sha(old/'notebook.ipynb')=='b46df73b9a897d0c68c04931afbfaa4ef44561aca66f0aaf8c8a828db4127871'
    notebook=json.loads((old/'notebook.ipynb').read_text(encoding='utf-8'));outer=unwrap(''.join(notebook['cells'][12]['source']))
    frozen=json.loads(base64.b64decode(value(outer,'FROZEN_B64')));original=json.loads(json.dumps(frozen))
    cell32=''.join(frozen['cells'][32]['source'])
    runners=[n.args[0].value for n in ast.walk(ast.parse(cell32)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='write' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str) and 'subs, recs, S = E.main' in n.args[0].value]
    assert len(runners)==1;runner=runners[0]
    needle='    T0 = time.time()';assert runner.count(needle)==1
    injected="""    import pandas as pd
    from empty_pool_recovery import augment
    _original_build = E.build_pool
    def _with_generated(L, workers):
        return augment(_original_build(L, workers), pd.read_parquet(test), json.load(open(sys.argv[5])), E, sys.argv[6])
    E.build_pool = _with_generated
    import torch
    assert torch.cuda.is_available()
    print('EMPTY POOL RECOVERY GPU', torch.cuda.get_device_name(0), flush=True)
    T0 = time.time()"""
    runner=runner.replace(needle,injected);compile(runner,'eng_generated_recovery_runner','exec')
    helper=(ROOT/'scripts/empty_pool_recovery.py').read_text(encoding='utf-8');compile(helper,'empty_pool_recovery.py','exec')
    prefix="""# Empty Engine2 catalogue recovery; all original fusion and coverage guards follow unchanged.
_e2r_active = set(te.molecule_id.astype(str))
_e2r_missing = _e2r_active - set(map(str, ENG))
if _e2r_missing:
    import os as _e2r_os, sys as _e2r_sys, subprocess as _e2r_subprocess
    from pathlib import Path as _e2r_Path
    _e2r_before = json.loads(json.dumps(ENG))
    _e2r_proposals = {}
    for _e2r_mid in sorted(_e2r_missing):
        _e2r_base, _e2r_keys, _ = BASE.get(_e2r_mid, ([], [], 0.0))
        assert _e2r_base and len(_e2r_base) == len(_e2r_keys), 'No real Engine1 proposals for missing Engine2 query'
        _e2r_proposals[_e2r_mid] = list(_e2r_base[:60])
    _e2r_root = _e2r_Path('/kaggle/working')
    _e2r_input = _e2r_root/'c20_engine2_recovery_input.parquet'
    _e2r_sample = _e2r_root/'c20_engine2_recovery_sample.csv'
    _e2r_json = _e2r_root/'c20_engine2_recovery_proposals.json'
    _e2r_output = _e2r_root/'c20_engine2_recovered_lists.json'
    _e2r_audit = _e2r_root/'c20_engine2_pool_audit.json'
    _e2r_q = pd.read_parquet(_e2r_os.path.join(COMP, 'test.parquet'))
    _e2r_q = _e2r_q[_e2r_q.molecule_id.astype(str).isin(_e2r_missing)]
    assert set(_e2r_q.molecule_id.astype(str)) == _e2r_missing
    _e2r_q.to_parquet(_e2r_input, index=False)
    pd.DataFrame(dict(molecule_id=sorted(_e2r_missing),smiles='CCO')).to_csv(_e2r_sample,index=False)
    _e2r_json.write_text(json.dumps(_e2r_proposals),encoding='utf-8')
    (_e2r_root/'eng_lists_before_recovery.json').write_text(json.dumps(ENG),encoding='utf-8')
    _e2r_Path(ENG_DIR,'empty_pool_recovery.py').write_text(E2R_HELPER,encoding='utf-8')
    _e2r_Path(ENG_DIR,'eng_generated_recovery_runner.py').write_text(E2R_RUNNER,encoding='utf-8')
    _e2r_args = [_e2r_sys.executable,str(_e2r_Path(ENG_DIR,'eng_generated_recovery_runner.py')),str(_e2r_input),
        _e2r_os.path.join(COMP,'train.parquet'),str(_e2r_sample),str(_e2r_output),str(_e2r_json),str(_e2r_audit)]
    _e2r_run = _e2r_subprocess.run(_e2r_args,capture_output=True,text=True,timeout=4*3600,cwd=ENG_DIR,
        env={**_e2r_os.environ,'CUDA_VISIBLE_DEVICES':'1','PYTHONHASHSEED':'0'})
    (_e2r_root/'c20_engine2_recovery.log').write_text(_e2r_run.stdout+'\\n'+_e2r_run.stderr,encoding='utf-8')
    assert _e2r_run.returncode == 0, _e2r_run.stderr[-2000:]
    _e2r_extra = json.loads(_e2r_output.read_text())
    assert set(_e2r_extra) == _e2r_missing and all(v['smiles'] and len(v['smiles'])==len(v['keys']) for v in _e2r_extra.values())
    ENG.update(_e2r_extra)
    assert all(ENG[k] == v for k,v in _e2r_before.items())
    assert _e2r_active <= set(map(str,ENG))
    (_e2r_root/'eng_lists.json').write_text(json.dumps(ENG),encoding='utf-8')
    (_e2r_root/'c20_engine2_recovery.json').write_text(json.dumps(dict(original_covered=len(_e2r_before),
        recovered_queries=sorted(_e2r_missing),final_covered=len(ENG),old_entries_exact=True,
        query_rows_dropped=0,placeholders_in_engine_lists=0,coverage_guard_retained=True,
        source='Existing Engine1 generated BASE top60; actual Engine2 fingerprints and unchanged rankers on initially empty pools only',
        prediction_label_metrics_read=False),indent=2),encoding='utf-8')
    print('REAL ENGINE2 EMPTY-POOL RECOVERY',len(_e2r_missing),'queries; original covered entries exact',flush=True)
"""
    code='E2R_HELPER = '+repr(helper)+'\nE2R_RUNNER = '+repr(runner)+'\n'+prefix
    original43=''.join(frozen['cells'][43]['source']);frozen['cells'][43]['source']=(code+'\n'+original43).splitlines(keepends=True)
    assert frozen['cells'][48]['source']==original['cells'][48]['source']
    assert "assert _active <= set(map(str, ENG))" in ''.join(frozen['cells'][48]['source'])
    assert all(frozen['cells'][i]['source']==original['cells'][i]['source'] for i in range(len(frozen['cells'])) if i!=43)
    for c in frozen['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'c20_recovery_frozen_cell','exec')
    payload=json.dumps(frozen,ensure_ascii=False,indent=1).encode('utf-8')
    outer=replace_value(outer,'FROZEN_B64',base64.b64encode(payload).decode());outer=replace_value(outer,'FROZEN_SHA256',hashlib.sha256(payload).hexdigest())
    notebook['cells'][12]['source']=guarded_source(outer,'c20_generated_recovery_payload').splitlines(keepends=True)
    before=json.loads((old/'notebook.ipynb').read_text(encoding='utf-8'))
    assert all(notebook['cells'][i]['source']==before['cells'][i]['source'] for i in range(len(before['cells'])) if i!=12)
    out.mkdir(exist_ok=True);(out/'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    (out/'kernel-metadata.json').write_bytes((old/'kernel-metadata.json').read_bytes())
    write(out/'recovery-pin.json',dict(notebook_sha256=sha(out/'notebook.ipynb'),original_notebook_sha256=sha(old/'notebook.ipynb'),
        builder_sha256=sha(__file__),helper_sha256=sha(ROOT/'scripts/empty_pool_recovery.py'),changed_frozen_cell=43,
        strict_coverage_guard_exact=True,existing_fusion_source_exact=True,other_frozen_cells_exact=True,
        query_data_and_contract_unchanged=True,queries_retained=80,query_labels_or_metrics_read=False,
        original_engine2_models_rankers_and_30ppm_rules_unchanged=True,
        adaptation='Real Engine1 generated proposals extend only an initially empty Engine2 candidate pool. Actual fingerprints and rankers execute on GPU1; do not claim an unchanged candidate universe for recovered queries.',
        existing_covered_79_parity_requires_execution_receipt=True,private=True,scoring_submission=False))
    print('BUILT REAL C20 EMPTY-POOL RECOVERY',sha(out/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
