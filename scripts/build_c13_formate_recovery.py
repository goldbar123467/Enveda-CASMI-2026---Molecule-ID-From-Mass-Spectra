"""Recover private C13's equivalent-formate lookup failure; retain all guards."""
import ast
import base64
import hashlib
import inspect
import json

import numpy as np
import pandas as pd

from precision_cache import ROOT,sha,write
from build_general_anchor import guarded_source
from build_c09_controls import unwrap
from adduct_aliases import normalize_pv_adduct


def replace_value(source,name,value):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1,name
    before=ast.get_source_segment(source,nodes[0]);assert source.count(before)==1
    return source.replace(before,name+' = '+repr(value))


def value(source,name):
    return ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))


def neutral_function(source):
    names={'MASS','E','PROTON','H2O','NH4','FORMATE','ACETATE','ADDUCTS'}
    nodes=[n for n in ast.parse(source).body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='neutral_mass')]
    namespace={'np':np,'normalize_pv_adduct':normalize_pv_adduct}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pv_numeric_lookup_only','exec'),namespace)
    return namespace['neutral_mass'],namespace['ADDUCTS']


def main():
    original=ROOT/'kaggle/forks/c13-mixed-casmi2022-validation'
    out=ROOT/'kaggle/forks/c13-formate-recovery';assert not (out/'recovery-pin.json').exists()
    assert sha(original/'notebook.ipynb')=='270a4b1d262cce1b3fe41696c2805b32729d07732adf7db30bb79d7889d7ecfa'
    notebook=json.loads((original/'notebook.ipynb').read_text(encoding='utf-8'))
    source=unwrap(''.join(notebook['cells'][12]['source']))
    frozen=json.loads(base64.b64decode(value(source,'FROZEN_B64')))
    before=json.loads(json.dumps(frozen));cell=''.join(frozen['cells'][32]['source']);files=value(cell,'ENG_FILES');old=dict(files)
    normalizer=inspect.getsource(normalize_pv_adduct)
    needle='ad = np.asarray(adduct, dtype=object)';assert files['pv.py'].count(needle)==1
    files['pv.py']=files['pv.py'].replace(needle,'ad = np.asarray([normalize_pv_adduct(a) for a in adduct], dtype=object)')+'\n\n'+normalizer
    needle="ADDUCT_IX.get(a, ADDUCT_IX['<unk>'])";assert files['pv_fp.py'].count(needle)==1
    files['pv_fp.py']=files['pv_fp.py'].replace(needle,"ADDUCT_IX.get(normalize_pv_adduct(a), ADDUCT_IX['<unk>'])")+'\n\n'+normalizer
    assert files['casmi_engine.py']==old['casmi_engine.py']
    for name,code in files.items():compile(code,name,'exec')
    old_fn,table=neutral_function(old['pv.py']);new_fn,_=neutral_function(files['pv.py'])
    known=np.array(list(table),object);mz=np.full(len(known),500.)
    np.testing.assert_array_equal(old_fn(mz,known),new_fn(mz,known))
    assert np.isnan(new_fn(np.array([500.]),np.array(['unknown'],object)))[0]
    queries=pd.read_parquet(ROOT/'outputs/publish/c13-mixed-casmi2022-20261110/validation-queries.parquet')
    old_mass=old_fn(queries.precursor_mz.to_numpy(),queries.adduct.to_numpy())
    new_mass=new_fn(queries.precursor_mz.to_numpy(),queries.adduct.to_numpy())
    assert int(np.isfinite(old_mass).sum())==98 and np.isfinite(new_mass).all()
    np.testing.assert_array_equal(old_mass[np.isfinite(old_mass)],new_mass[np.isfinite(old_mass)])
    assert int((queries.adduct.map(normalize_pv_adduct)!=queries.adduct).sum())==22
    frozen['cells'][32]['source']=replace_value(cell,'ENG_FILES',files).splitlines(keepends=True)
    assert all(frozen['cells'][i]['source']==before['cells'][i]['source'] for i in range(len(frozen['cells'])) if i!=32)
    assert frozen['cells'][48]['source']==before['cells'][48]['source']
    assert "assert _active <= set(map(str, ENG))" in ''.join(frozen['cells'][48]['source'])
    payload=json.dumps(frozen,ensure_ascii=False,indent=1).encode('utf-8')
    source=replace_value(source,'FROZEN_B64',base64.b64encode(payload).decode())
    source=replace_value(source,'FROZEN_SHA256',hashlib.sha256(payload).hexdigest())
    notebook['cells'][12]['source']=guarded_source(source,'c13_formate_recovery_payload').splitlines(keepends=True)
    for c in notebook['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'c13_v2_cell','exec');c['outputs']=[];c['execution_count']=None
    old_notebook=json.loads((original/'notebook.ipynb').read_text(encoding='utf-8'))
    assert all(notebook['cells'][i]['source']==old_notebook['cells'][i]['source'] for i in range(len(notebook['cells'])) if i!=12)
    out.mkdir(exist_ok=True)
    (out/'notebook.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    (out/'kernel-metadata.json').write_bytes((original/'kernel-metadata.json').read_bytes())
    write(out/'recovery-pin.json',dict(notebook_sha256=sha(out/'notebook.ipynb'),builder_sha256=sha(__file__),
        normalizer_sha256=sha(ROOT/'scripts/adduct_aliases.py'),original_notebook_sha256=sha(original/'notebook.ipynb'),
        source_adaptation='PV neutral-mass and FP-encoder lookup accepts equivalent formate spelling and boundary whitespace; no unknown-adduct default.',
        unchanged_canonical_mass_lookups=len(known),old_finite_query_mass=98,new_finite_query_mass=120,
        old98_query_neutral_masses_exact=True,changed_query_adduct_encodings=22,
        all120_queries_retained=True,query_data_and_contract_unchanged=True,coverage_guard_unchanged=True,
        other_frozen_cells_exact=True,ranking_weights_models_simulators_unchanged=True,
        fresh_query_labels_or_metrics_read=False,private=True,scoring_submission=False,
        engine_files_before={n:hashlib.sha256(v.encode()).hexdigest() for n,v in old.items()},
        engine_files_after={n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()}))
    print('BUILT PRIVATE C13 FORMATE RECOVERY',sha(out/'notebook.ipynb'),flush=True)


if __name__=='__main__':main()
