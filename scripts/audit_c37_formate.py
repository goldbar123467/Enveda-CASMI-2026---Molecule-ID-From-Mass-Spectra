"""Prove the cached legacy alias defect and stage a CPU-tested source adapter."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from trace_c37_selection import ROOT, OUT, C36, sha, write, verify, extract_executed
from legacy_formate_compat import patch_chem_source, patch_frag_source, scalar_namespace, fragment_offsets


def main():
    assert (ROOT/'.local/compute-paused.json').is_file()
    assert not (OUT/'formate-compat-audit.json').exists()
    contract, _ = verify()
    pinned = OUT/'pinned-public-source'
    code = (pinned/'chem.py').read_text(encoding='utf-8')
    frag = (pinned/'frag.py').read_text(encoding='utf-8')
    # Bind the downloaded scalar source to the exact mounted files of C20.
    inventory = json.loads((OUT/'c20-stage-artifacts/v43_mounted_input_hashes.json').read_text())
    matches = {}
    def walk(value):
        if isinstance(value, dict):
            yield value
            for v in value.values(): yield from walk(v)
        elif isinstance(value, list):
            for v in value: yield from walk(v)
    for name in ['chem.py', 'frag.py']:
        digest = sha(pinned/name)
        matches[name] = [r for r in walk(inventory) if r.get('sha256') == digest]
        assert matches[name], ('Not mounted source', name)
    patched_code = patch_chem_source(code); patched_frag = patch_frag_source(frag)
    old = scalar_namespace(code); new = scalar_namespace(patched_code)
    old_frag = fragment_offsets(frag, old); new_frag = fragment_offsets(patched_frag, new)
    assert old['ADDUCT_LIST'] == new['ADDUCT_LIST'] and old['ADDUCT_IX'] == new['ADDUCT_IX']
    known = np.asarray(list(old['ADDUCTS']), dtype=object)
    # Cover every supported adduct over several precursor values, not just the panel.
    for mass in [125., 500., 1500.]:
        mz = np.full(len(known), mass)
        np.testing.assert_array_equal(old['neutral_mass'](mz, known), new['neutral_mass'](mz, known))
    for adduct in known:
        assert old['adduct_index'](adduct) == new['adduct_index'](adduct)
        assert old['adduct_delta'](adduct) == new['adduct_delta'](adduct)
        assert old['precursor_mz_from_mass'](500., adduct) == new['precursor_mz_from_mass'](500., adduct)
        assert old_frag(adduct) == new_frag(adduct)
    for unknown in ['unknown', '[M+MADEUP]-', None, ' [M+H]+ ']:
        assert np.isnan(new['neutral_mass']([500.], [unknown])[0])
        assert new['adduct_index'](unknown) == old['ADDUCT_IX']['<unk>']
    alias, canonical = '[M+FA-H]-', '[M+CH2O2-H]-'
    assert new['adduct_index'](alias) == new['adduct_index'](canonical)
    assert new['adduct_delta'](alias) == new['adduct_delta'](canonical)
    assert new['precursor_mz_from_mass'](500., alias) == new['precursor_mz_from_mass'](500., canonical)
    assert new_frag(alias) == new_frag(canonical) and old_frag(alias) != old_frag(canonical)
    q = pd.read_parquet(ROOT/contract['inputs']['queries']['path'])
    assert q.molecule_id.is_unique and len(q) == 240
    before = old['neutral_mass'](q.precursor_mz.to_numpy(), q.adduct.to_numpy())
    after = new['neutral_mass'](q.precursor_mz.to_numpy(), q.adduct.to_numpy())
    finite = np.isfinite(before); assert int(finite.sum()) == 215 and np.isfinite(after).all()
    np.testing.assert_array_equal(before[finite], after[finite])
    affected = q.adduct.eq(alias); assert int(affected.sum()) == 25
    pc = json.loads((OUT/'c20-stage-artifacts/e004_candidate_inputs.json').read_text())['pubchem']
    assert all(not pc.get(mid) for mid in q.loc[affected, 'molecule_id'])
    frozen, _ = extract_executed('c20')
    # Inspect the executed embedded worker rather than execute its model code.
    import ast
    runner = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(frozen)) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'RUNNER' for t in n.targets))
    assert "te['nm'] = chem.neutral_mass" in runner
    assert 'if not len(nms):\n            continue' in runner
    output = OUT/'prospective-formate-compat'; output.mkdir(exist_ok=True)
    (output/'chem.py').write_text(patched_code, encoding='utf-8')
    (output/'frag.py').write_text(patched_frag, encoding='utf-8')
    witness = q[['molecule_id', 'ionization_mode', 'adduct']].copy()
    witness['legacy_mass_available'] = finite; witness['patched_mass_available'] = np.isfinite(after)
    witness['legacy_pubchem_present'] = q.molecule_id.map(lambda m: bool(pc.get(m)))
    witness.to_csv(OUT/'formate-compat-witnesses.csv', index=False)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                  auditor_sha256=sha(__file__), adapter_sha256=sha(ROOT/'scripts/legacy_formate_compat.py'),
                  mounted_inventory_sha256=sha(OUT/'c20-stage-artifacts/v43_mounted_input_hashes.json'),
                  source_sha256={n: sha(pinned/n) for n in matches}, mounted_exact_source_matches={n: len(v) for n,v in matches.items()},
                  patched_source_sha256={n: sha(output/n) for n in ['chem.py', 'frag.py']},
                  queries=240, affected_formate_queries=25, old_finite_mass=215, patched_finite_mass=240,
                  affected_queries_missing_pubchem=25, existing215_masses_exact=True,
                  existing_adducts_tested=len(known), supported_mass_vectors_tested=3*len(known),
                  old_vocabulary_and_indices_exact=True, unknown_mass_stays_nan=True,
                  formate_fragment_offsets_equivalent=True, existing_fragment_offsets_exact=True,
                  existing_supported_scalar_lookups_exact=True, shared_query_tokens_unchanged=True,
                  fresh_holdout_opened=False, new_predictions=False, new_gpu_work=False,
                  production_notebook_changed=False, quality_metric_claim=None,
                  disposition='General parser defect confirmed; prospective local adapter only. Candidate eligibility repair is not demonstrated retrieval or MRR improvement. No GPU run authorized.',
                  limitations=['Missing PubChem formate candidates and model logits cannot be reconstructed from saved final outputs.',
                               'Legacy model training exposure and out-of-source benefit remain unqualified.',
                               'Adapter requires explicit future integration into the correct legacy subprocess code directories; it is not installed in production.'])
    write(OUT/'formate-compat-audit.json', report)
    print('DONE C37 FORMATE', json.dumps(report), flush=True)


if __name__ == '__main__': main()
