"""Synthetic strict-mask exercise and original-notebook source-preservation tests.

Never opens actual C13 queries/answers or calls a Kaggle API. The full-size
reference fixtures contain synthetic rows only, under one temporary directory.
"""
import copy
import hashlib
import itertools
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from c75_port_utils import ROOT, read, write, sha, unwrap, value
import c76_confirmation_preparation as shared
import build_c76_confirmation as builder
import audit_c76_confirmation_notebooks as notebooks
import prepare_c76_confirmation_references as prepare
import audit_c76_confirmation_references as references


class PreparationTests(unittest.TestCase):
    def test_alias_closure_order_independent_transitive_and_invalid(self):
        edges = [('R1', 'A'), ('R2', 'A'), ('R2', 'B'), ('R3', 'B'), ('BAD', None), ('SAFE', 'C')]
        expected = {'R1', 'R2', 'R3', 'BAD'}, {'A', 'B'}
        for permutation in itertools.permutations(edges[:4]):
            primary = prepare.alias_closure([*permutation, *edges[4:]], {'R1'}, set())
            independent = references.graph_closure([*permutation, *edges[4:]], {'R1'}, set())
            self.assertEqual(primary[:2], expected); self.assertEqual(independent, expected)

    def test_seed_keys_outside_witnesses_and_identity_domain_preserved(self):
        edges = [('A', 'B'), ('B', 'C'), ('SAFE', 'OTHER')]
        result = prepare.alias_closure(edges, {'ABSENT'}, {'C'})
        self.assertEqual(result[:2], ({'ABSENT', 'B'}, {'C'}))
        self.assertEqual(result[:2], references.graph_closure(edges, {'ABSENT'}, {'C'}))

    def test_both_source_payloads_and_all_untouched_cells_exact(self):
        pin = read(ROOT / 'kaggle/forks/c73-full-masked-validation/validation-pin.json')['query_contract_sha256']
        for arm in shared.ARMS:
            folder = ROOT / 'kaggle/forks' / shared.TEMPLATES[arm]
            original = read(folder / 'notebook.ipynb')
            candidate = builder.adapt_notebook(original, arm, pin, '1' * 64)
            audit = notebooks.compare(original, candidate, arm, pin, '1' * 64)
            self.assertTrue(audit['whole_scientific_inner_payload_byteexact'])
            self.assertEqual(value(unwrap(''.join(candidate['cells'][12]['source'])), 'FROZEN_SHA256'), shared.CORE_PINS[arm])
            metadata = builder.adapt_metadata(read(folder / 'kernel-metadata.json'), arm)
            self.assertEqual(metadata['dataset_sources'].count(shared.DATASET), 1)
            self.assertNotIn(builder.ORIGINAL_DATASET, metadata['dataset_sources'])

    def test_source_tamper_or_wrong_template_rejected(self):
        pin = read(ROOT / 'kaggle/forks/c73-full-masked-validation/validation-pin.json')['query_contract_sha256']
        original = read(ROOT / 'kaggle/forks/c73-full-masked-validation/notebook.ipynb')
        with self.assertRaises(AssertionError): builder.adapt_notebook(original, 'c76', pin, '1' * 64)
        candidate = builder.adapt_notebook(original, 'legacy', pin, '1' * 64)
        source = candidate['cells'][20]['source']
        candidate['cells'][20]['source'] = [''.join(source), '\nUNAUTHORIZED_RANKING_CHANGE=True\n']
        with self.assertRaises(AssertionError): notebooks.compare(original, candidate, 'legacy', pin, '1' * 64)
        with self.assertRaises(AssertionError): builder.adapt_notebook(original, 'legacy', pin, 'not-a-hash')

    def test_all_preparation_entrypoints_stop_before_query_access(self):
        class PriorHold(Exception): pass
        functions = [shared.read_queries, prepare.raw_aliases, prepare.main_mask,
            prepare.p16_mask, prepare.package, builder.build, notebooks.main, references.main]
        with patch.object(shared, 'authorization', return_value={'submissions_used': 0}), \
            patch.object(shared, 'source_and_production', side_effect=PriorHold('Synthetic missing prior evidence')):
            for fn in functions:
                with self.subTest(fn=fn.__name__), self.assertRaises(PriorHold): fn()

    def test_strict_masks_and_independent_audit_end_to_end_synthetic(self):
        import numpy as np
        import pandas as pd
        import c74_complete_target as c74
        features = c74.module('c76_synthetic_reference_fixture_identity', 'c09e_features.py')
        acids = 'O=C(O)c1ccc(Cl)cc1'; safe = 'OP(=O)(O)O'
        held_smiles = ['CCO', 'CCCCCCCCCC', 'c1ccc2ccccc2c1']
        descriptors = [features.structure(s) for s in held_smiles]
        held = pd.DataFrame(dict(key=[d[0] for d in descriptors], smiles=[d[1] for d in descriptors], formula=[d[2] for d in descriptors]))
        acid_key = features.structure(acids)[0]; safe_key = features.structure(safe)[0]
        with tempfile.TemporaryDirectory(prefix='c76-synthetic-mask-') as name, ExitStack() as stack:
            root = Path(name); old = root / 'old'; prep = root / 'prepared'; original = root / 'synthetic-original'; reference = root / 'reference'
            for folder in [old, prep, original, reference, root / 'scripts']: folder.mkdir(parents=True)
            (root / 'scripts/prepare_c76_confirmation_references.py').write_bytes(Path(prepare.__file__).read_bytes())
            binding = root / 'binding.json'; write(binding, {'synthetic': True})
            q = pd.DataFrame(dict(molecule_id=['SYNTHETIC_' + str(i) for i in range(120)], spectrum_id=['SYNTHETIC_SPEC_' + str(i) for i in range(120)],
                ionization_mode=['positive'] * 60 + ['negative'] * 60, precursor_mz=[300.] * 120,
                ms2_mzs=[np.array([50., 100., 150.])] * 120, ms2_normalized_intensities=[np.array([1., .5, .2])] * 120))
            q.to_parquet(original / 'validation-queries.parquet', index=False)
            raw = root / 'synthetic-raw.parquet'
            pd.DataFrame(dict(precursor_mz=[300., 300., 900.], inchikey14=['ALIAS', 'REFRAW', 'OUTSIDE'],
                ms2_mzs=[np.array([50., 100., 150.])] * 3, ms2_normalized_intensities=[np.array([1., .5, .2])] * 3)).to_parquet(raw, index=False)
            protocol = root / 'outputs/local/c39-source-supported-validation-20261111/preparation-protocol.json'
            protocol.parent.mkdir(parents=True); write(protocol, {'raw_training_sha256': sha(raw)})
            witness = pd.DataFrame(dict(raw_key=['OLD', 'ALIAS', 'NEW_ALIASED', 'SAFE'],
                official_key=[None, acid_key, acid_key, safe_key], smiles=['invalid', acids, acids, safe], exclude=[True, False, False, False]))
            witness.to_parquet(old / 'original-structure-identity-witnesses.parquet', index=False)
            write(old / 'engine2-raw-key-exclusion.json', {'keys': ['OLD']})
            main_reference = root / 'main-reference.parquet'; n = 275810
            strings = [safe] * n; strings[:3] = ['invalid', acids, acids]
            raw_keys = ['SAFE'] * n; raw_keys[:3] = ['OLD', 'ALIAS', 'NEW_ALIASED']
            pd.DataFrame(dict(sid=np.arange(n), inchikey14=raw_keys, smiles=strings)).to_parquet(main_reference, index=False)
            mask = np.zeros(n, bool); mask[0] = True; np.save(old / 'main-reference-sid-exclusion.npy', mask)
            names = ['original-structure-identity-witnesses.parquet', 'engine2-raw-key-exclusion.json', 'main-reference-sid-exclusion.npy']
            write(old / 'reference-exclusion-freeze.json', dict(reference_structure_sha256=sha(main_reference), files_sha256={p: sha(old / p) for p in names}))
            write(old / 'independent-reference-mask-audit.json', dict(passed=True, reference_exclusion_freeze_sha256=sha(old / 'reference-exclusion-freeze.json'),
                retained_global_excluded_identity_formula_scaffold_or_near_chemistry=0))
            held.to_parquet(old / 'global-exclusion-chemistry.parquet', index=False)
            write(old / 'chemistry-classification-freeze.json', dict(files_sha256={(old / 'global-exclusion-chemistry.parquet').relative_to(root).as_posix(): sha(old / 'global-exclusion-chemistry.parquet')}))
            total = 200724
            molecules = [safe] * total
            molecules[:8] = ['CCO', 'COC', 'CCCCCCCCC', 'Oc1ccc2ccccc2c1', acids, safe, 'CS(=O)(=O)O', safe]
            raw_keys = ['GOOD'] * total; raw_keys[5] = 'REFRAW'
            mass = np.full(total, 900.); mass[:8] = 300.
            pd.DataFrame(dict(normalized_smiles=molecules, inchikey14=raw_keys, precursor_mz=mass)).to_parquet(reference / 'ext_meta.parquet', index=False)
            for filename, values in [('ext_mz.npy', [60., 120., 180.]), ('ext_in.npy', [1., .7, .1])]:
                array = np.lib.format.open_memmap(reference / filename, mode='w+', dtype=np.float16, shape=(total, 128))
                array[:] = 0; array[:, :3] = values
                array[6, :3] = [50., 100., 150.] if filename == 'ext_mz.npy' else [1., .5, .2]
                array.flush(); del array
            reference_files = {p: sha(reference / p) for p in ['ext_meta.parquet', 'ext_mz.npy', 'ext_in.npy']}
            for module in [prepare, references]:
                for attr, value in [('ROOT', root), ('C73', old), ('PREPARATION', prep), ('ORIGINAL', original), ('BINDING', binding)]: stack.enter_context(patch.object(module, attr, value))
                stack.enter_context(patch.object(module, 'prior', return_value=None))
                stack.enter_context(patch.object(module, 'read_queries', return_value=q))
            stack.enter_context(patch.object(prepare, 'RAW', raw)); stack.enter_context(patch.object(prepare, 'MAIN_REFERENCE', main_reference))
            stack.enter_context(patch.object(prepare, 'REFERENCE', reference))
            # The auditor deliberately constructs canonical workspace paths itself.
            (root / 'outputs/competition-raw').mkdir(parents=True)
            (root / 'outputs/research/c08-pipeline/reference').mkdir(parents=True)
            (root / 'outputs/research/2026-11-09-silver').mkdir(parents=True)
            import shutil
            shutil.copyfile(raw, root / 'outputs/competition-raw/train.parquet')
            shutil.copyfile(main_reference, root / 'outputs/research/c08-pipeline/reference/train_structs.parquet')
            (root / 'outputs/research/2026-11-09-silver/extlib').symlink_to(reference, target_is_directory=True)
            stack.enter_context(patch.object(c74, 'frozen_contract', return_value={'reference_files': reference_files}))
            prepare.raw_aliases(); prepare.main_mask(); prepare.p16_mask(); references.main()
            proof = read(prep / 'independent-reference-audit.json'); self.assertTrue(proof['passed'])
            self.assertTrue(proof['all_reachable_structure_and_peak_alias_flags_independently_rebuilt'])
            flags = pd.read_parquet(prep / 'masked-P16-reference/row-exclusion-audit.parquet')
            for row, reason in [(0, 'query_identity'), (1, 'same_formula'), (2, 'near_query_tanimoto'),
                (3, 'same_nonempty_scaffold'), (4, 'known_official_alias'), (5, 'known_raw_alias'), (6, 'query_input_alias'), (8, 'outside_all_frozen_precursor_windows')]:
                self.assertTrue(flags.loc[row, reason], (row, reason))
            self.assertEqual(read(prep / 'masked-P16-reference/mask-manifest.json')['retained_rows'], 1)
            self.assertEqual(np.flatnonzero(~flags.drop(columns='original_row').any(axis=1).to_numpy()).tolist(), [7])
            np.testing.assert_array_equal(np.load(prep / 'main-reference-sid-exclusion.npy')[:4], [True, True, True, False])


if __name__ == '__main__':
    import os
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
    import torch
    unittest.main(verbosity=2)
