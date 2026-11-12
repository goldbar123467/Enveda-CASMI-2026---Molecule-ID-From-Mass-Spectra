"""Independently audit complete reserved parents and both fixed model rankings."""
import math
import statistics
import types
from c75_port_utils import ROOT, read, write, sha, stamp
from c76_submission_contract import BINDING, CONFIRM, FAMILY, verify_files
from c76_confirmation_preparation import ARMS, PREPARATION
from c76_confirmation_inference import BUNDLE, CLOUDS, PLAN, collected, verify_plan
from complete_c76_confirmation import PARENTS, PREDICTIONS, PRE_FREEZE, load


def independent_physical(p01, p16, q, matches, references, adducts, normalize):
    """Scalar mass equations and finite-only median, independent of c67 code."""
    result = p16.copy(); accepted = rejected = 0
    aliases = {'[M+H-H2O]+': '[M-H2O+H]+', '[M+H-2H2O]+': '[M-2H2O+H]+'}
    def mass(value, token):
        data = adducts.get(normalize(token))
        if data is None: return float('nan')
        number, charge, delta, _ = data
        return (float(value)*charge-delta)/number
    for index, row in p01.iterrows():
        if row.smiles == p16.smiles.iloc[index]: continue
        values = [mass(r.precursor_mz, r.adduct) for r in q[q.molecule_id.eq(row.molecule_id)].itertuples(index=False)]
        values = [value for value in values if math.isfinite(value)]
        query_mass = statistics.median(values) if values else float('nan')
        hit = matches[matches.molecule_id.eq(row.molecule_id) & matches.view.eq('all_reference')]
        assert len(hit) == 1; reference = references.iloc[int(hit.reference_row.iloc[0])]
        reference_mass = mass(reference.precursor_mz, aliases.get(reference.adduct, reference.adduct))
        keep = math.isfinite(query_mass) and math.isfinite(reference_mass) and abs(query_mass-reference_mass) <= max(.002, query_mass*1e-5)
        if keep: accepted += 1
        else: rejected += 1; result.at[index, 'smiles'] = row.smiles
    return result, dict(kept=accepted, reverted=rejected)


def independent_head(rows, features, model, trial, np):
    assert trial in ['cpu', 'gpu'] and np.isfinite(features).all()
    assert np.array_equal(rows.index.to_numpy(), np.arange(len(rows)))
    z = np.minimum(10., np.maximum(-10., (features.astype(float)-model['mean'])/model['std']))
    scores = -np.log(rows.parent_rank.to_numpy(float)) + z.dot(model['weights'])
    if trial == 'gpu':
        h = z.dot(model['hidden_weight'].T) + model['hidden_bias']
        gelu = h*.5*(1.+np.tanh((2./np.pi)**.5*(h+.044715*h*h*h)))
        scores += .25*np.tanh(gelu.dot(model['output_weight'].T)+model['output_bias']).reshape(-1)
    assert np.isfinite(scores).all()
    lists = {mid: group.sort_values('parent_rank').smiles.tolist() for mid, group in rows.groupby('molecule_id', sort=False)}
    for (mid, _), group in rows.groupby(['molecule_id', 'formula'], sort=False):
        if len(group) < 2 or not group.available.all(): continue
        order = np.lexsort((group.parent_rank.to_numpy(), -scores[group.index]))
        texts = group.smiles.to_numpy()[order]
        for slot, text in zip(group.parent_rank, texts): lists[mid][int(slot)-1] = text
    return lists, scores


def main():
    verify_plan(); target = CONFIRM / 'execution-qualification.json'; assert not target.exists()
    frozen = read(PRE_FREEZE); verify_files(frozen['files_sha256'])
    assert frozen['source_sha256'] == sha(ROOT / 'scripts/complete_c76_confirmation.py')
    assert frozen['family'] == FAMILY and frozen['inference_plan_sha256'] == sha(PLAN)
    assert frozen['all_six_controls_and_both_heads_frozen_before_quality'] and not frozen['quality_labels_read']
    import torch
    import numpy as np
    import pandas as pd
    from functools import lru_cache
    from c74_complete_target import CODE, independent_overlay, module
    reference = PREPARATION / 'masked-P16-reference'; refs = pd.read_parquet(reference / 'ext_meta.parquet')
    reference_audit = read(PREPARATION / 'independent-reference-audit.json')
    assert reference_audit['passed'] and reference_audit['forbidden_reference_rows'] == 0
    assert reference_audit['all_prior_and_global231_chemistry_and_alias_masks_preserved']
    verify_files(reference_audit['artifact_sha256'])
    notebook_audit = read(PREPARATION / 'independent-notebook-audit.json')
    assert notebook_audit['passed']; verify_files(notebook_audit['artifact_sha256'])
    identity = lru_cache(maxsize=50000)(module('c76_c13_execution_identity', 'c09e_features.py').structure)
    compatibility = load('c76_c13_scalar_mass_compat', BUNDLE / 'legacy_formate_compat.py')
    chemistry = types.ModuleType('c76_c13_scalar_mass')
    source = BUNDLE / 'legacy_chem_source.py'
    exec(compile(compatibility.patch_chem_source(source.read_text(encoding='utf8')), str(source), 'exec'), chemistry.__dict__)
    physical_checks = {}; collections = {}
    for arm in ARMS:
        cloud, collection = collected(arm); collections[arm] = sha(cloud / 'collection-audit.json')
        assert collection['complete_pipeline_binding_sha256'] == sha(BINDING)
        parent = PARENTS / arm; work = parent / 'work'; pin = read(parent / 'parent-freeze.json')
        assert pin['source_sha256'] == sha(ROOT / 'scripts/complete_c76_confirmation.py')
        assert pin['collection_audit_sha256'] == sha(cloud / 'collection-audit.json'); verify_files(pin['files_sha256'])
        p01 = pd.read_csv(parent / 'P01.csv', dtype=str, keep_default_na=False)
        p16 = pd.read_csv(parent / 'P16.csv', dtype=str, keep_default_na=False)
        q = pd.read_parquet(parent / 'queries.parquet')
        pd.testing.assert_frame_equal(p01, pd.read_csv(cloud / 'c41_p01.csv', dtype=str, keep_default_na=False), check_exact=True)
        pd.testing.assert_frame_equal(q, pd.read_parquet(cloud / 'c41_queries.parquet'), check_exact=True)
        receipt = read(work / 'c09e_external_receipt.json')
        assert receipt['required_stage_completed'] and receipt['baseline_sha256'] == sha(parent / 'P01.csv')
        assert receipt['csv_sha256'] == sha(parent / 'P16.csv') and receipt['query_sha256'] == sha(work / 'effective_query_spectra.parquet')
        assert receipt['search']['queries'] == receipt['search']['spectra'] == 120
        assert receipt['search']['query_input_alias_reference_rows'] == 0
        matches = pd.read_parquet(work / 'c09e_matches.parquet')
        evidence = pd.read_parquet(work / 'frozen_candidate_evidence.parquet')
        assert evidence.groupby('molecule_id').library_max.nunique(dropna=False).eq(1).all() and np.isfinite(evidence.library_max).all()
        confidence = evidence.groupby('molecule_id').library_max.first().to_dict()
        rebuilt = independent_overlay(p01, matches[matches.view.eq('all_reference')], confidence,
            read(CODE / 'c09e_policy.json'), lambda text: identity(text)[0])
        pd.testing.assert_frame_equal(rebuilt, p16, check_exact=True)
        physical, checks = independent_physical(p01, p16, q, matches, refs, chemistry.ADDUCTS, compatibility.normalize_legacy_adduct)
        pd.testing.assert_frame_equal(physical, pd.read_csv(parent / 'physical-parent.csv', dtype=str, keep_default_na=False), check_exact=True)
        physical_checks[arm] = checks
    score_receipt = read(PREDICTIONS / 'score-receipt.json')
    assert score_receipt['source_sha256'] == sha(BUNDLE / 'score_c76_complete_pair.py')
    assert score_receipt['pair_manifest_sha256'] == sha(BUNDLE / 'pair-manifest.json')
    for name, pin in score_receipt['files_sha256'].items(): assert sha(PREDICTIONS / name) == pin
    parent = pd.read_csv(PARENTS / 'c76/physical-parent.csv', dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(parent, pd.read_csv(PREDICTIONS / 'physical-parent.csv', dtype=str, keep_default_na=False), check_exact=True)
    rows = pd.read_parquet(PREDICTIONS / 'candidates.parquet'); features = np.load(PREDICTIONS / 'features.npy')
    assert features.shape == (len(rows), 101) and np.isfinite(features).all() and rows.molecule_id.nunique() == 120
    assert not rows.duplicated(['molecule_id', 'key']).any()
    ranker = load('c76_c13_rebuilt_features', BUNDLE / 'c67_ranker.py')
    cloud = CLOUDS['c76']; q = pd.read_parquet(cloud / 'c41_queries.parquet')
    pop = pd.read_csv(BUNDLE / 'pool_popularity.csv', usecols=['inchikey14', 'pop', 'np_coconut', 'np_lotus', 'src_natural_product'])
    again_rows, again_features, names = ranker.assemble(parent, q, pd.read_parquet(cloud / 'c41_neural_p01.parquet'),
        pd.read_parquet(cloud / 'neural_public.parquet'), pd.read_parquet(cloud / 'neural_ours.parquet'), pop, identity)
    pd.testing.assert_frame_equal(rows, again_rows, check_exact=True); np.testing.assert_array_equal(features, again_features)
    assert names == read(PREDICTIONS / 'feature-names.json')
    heads = {}
    for trial in ['cpu', 'gpu']:
        model_path = BUNDLE / (trial + '-model.npz')
        assert sha(model_path) == read(BINDING)['candidates'][trial]['model_sha256']
        with np.load(model_path, allow_pickle=False) as model: lists, scores = independent_head(rows, features, model, trial, np)
        np.testing.assert_allclose(scores, np.load(PREDICTIONS / (trial + '-scores.npy')), atol=1e-10, rtol=1e-10)
        final = pd.read_csv(PREDICTIONS / (trial + '.csv'), dtype=str, keep_default_na=False)
        rebuilt = parent.copy(); rebuilt['smiles'] = rebuilt.molecule_id.map(lambda mid: ';'.join(lists[mid]))
        pd.testing.assert_frame_equal(final, rebuilt, check_exact=True)
        rank1 = 0
        for before, after in zip(parent.smiles, final.smiles):
            a, b = before.split(';'), after.split(';')
            assert set(a) == set(b) and [identity(s)[2] for s in a] == [identity(s)[2] for s in b]
            assert len(b) == len({identity(s)[0] for s in b})
            rank1 += identity(a[0])[0] != identity(b[0])[0]
        heads[trial] = dict(candidate_rows=len(rows), scores_independently_rebuilt=True,
            final_formula_slot_lists_independently_rebuilt=True, rank1_changes=rank1)
    files = {**frozen['files_sha256'], PRE_FREEZE.relative_to(ROOT).as_posix(): sha(PRE_FREEZE)}
    files.update(reference_audit['artifact_sha256']); files.update(notebook_audit['artifact_sha256'])
    write(target, dict(completed_utc=stamp(), source_sha256=sha(__file__), family=FAMILY, passed=True,
        inference_plan_sha256=sha(PLAN), complete_pipeline_binding_sha256=sha(BINDING),
        pre_evaluation_artifact_freeze_sha256=sha(PRE_FREEZE), queries=120, acquisitions=120,
        actual_two_T4s_both_controls=True, original_C76_R3_core_and_adapter_exact=True,
        all_original_metadata_peaks_and_unknown_CE_exact=True, original_trailing_space_token_preserved=True,
        exact_PC_supported_IDs=119, exact_Engine2_IDs=120, forbidden_reference_rows=0,
        all_prior_and_global231_chemistry_and_alias_masks_preserved=True,
        both_frozen_heads_independently_rebuilt=True, physical_scalar_checks=physical_checks, heads=heads,
        original_actual_core_collection_audits_sha256=collections, artifact_sha256=files,
        quality_labels_read=False, scoring_entries=0, qualified_for_scoring=False))
    print('C76 COMPLETE RESERVED EXECUTION AND BOTH HEAD RANKINGS INDEPENDENTLY AUDITED', flush=True)


if __name__ == '__main__': main()
