"""Prospective adapter for the receipt-pinned legacy chem/frag source files.

Normalize only this equivalent formate token at the legacy lookup boundary.
Do not rewrite shared queries or the separate encoder's adduct vocabulary.
"""
import ast


def normalize_legacy_adduct(value):
    return '[M+CH2O2-H]-' if value == '[M+FA-H]-' else value


def patch_chem_source(source):
    import inspect
    replacements = [
        ('ADDUCT_IX.get(a, ADDUCT_IX["<unk>"])', 'ADDUCT_IX.get(normalize_legacy_adduct(a), ADDUCT_IX["<unk>"])', 1),
        ('ad = np.asarray(adduct, dtype=object)', 'ad = np.asarray([normalize_legacy_adduct(a) for a in adduct], dtype=object)', 1),
        ('ADDUCTS[adduct]', 'ADDUCTS[normalize_legacy_adduct(adduct)]', 2),
    ]
    assert 'def normalize_legacy_adduct(' not in source, 'Already patched'
    for old, new, count in replacements:
        assert source.count(old) == count, ('Upstream source changed', old)
        source = source.replace(old, new)
    source += '\n\n'+inspect.getsource(normalize_legacy_adduct)
    compile(source, 'legacy_chem_formate_compat.py', 'exec')
    return source


def patch_frag_source(source):
    assert 'adduct = chem.normalize_legacy_adduct(adduct)' not in source, 'Already patched'
    needle = "    n, z, d, sign = chem.ADDUCTS.get(adduct, chem.ADDUCTS['[M+H]+' if adduct.endswith('+') else '[M-H]-'])"
    assert source.count(needle) == 1, 'Upstream fragment source changed'
    source = source.replace(needle, '    adduct = chem.normalize_legacy_adduct(adduct)\n'+needle)
    compile(source, 'legacy_frag_formate_compat.py', 'exec')
    return source


def scalar_namespace(source):
    """Extract constants and scalar chemistry only; never load models."""
    import numpy as np
    from typing import Dict, Tuple
    names = {'MASS', 'ELECTRON', 'H2O', 'NH3', 'PROTON', '_H', '_C', '_N', '_O',
             'ADDUCTS', 'TEST_ADDUCTS', 'ADDUCT_LIST', 'ADDUCT_IX'}
    functions = {'_ad', 'adduct_index', 'neutral_mass', 'precursor_mz_from_mass',
                 'adduct_delta', 'normalize_legacy_adduct'}
    nodes = []
    for n in ast.parse(source).body:
        if isinstance(n, ast.FunctionDef) and n.name in functions: nodes.append(n)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in names: nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets): nodes.append(n)
    namespace = dict(np=np, Dict=Dict, Tuple=Tuple)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'legacy_numeric_lookups_only', 'exec'), namespace)
    return namespace


def fragment_offsets(source, chem_namespace):
    from types import SimpleNamespace
    from typing import List
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'ion_deltas')
    namespace = dict(chem=SimpleNamespace(**chem_namespace), PROTON=chem_namespace['PROTON'],
                     H_MASS=chem_namespace['MASS']['H'], ELECTRON=chem_namespace['ELECTRON'], List=List)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'fragment_offsets_only', 'exec'), namespace)
    return namespace['ion_deltas']
