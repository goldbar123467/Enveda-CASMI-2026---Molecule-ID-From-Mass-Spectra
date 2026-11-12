"""Synthetic-only arithmetic and once-only label-access boundary tests."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import evaluate_c76_confirmation as evaluator


class ProtocolTests(unittest.TestCase):
    def test_absent_truths_are_in_denominator_and_recovery_displacement(self):
        r = evaluator.exact_point([0, 2, 1], [1, 0, 2], range(3))
        self.assertEqual((r['numerator'], r['denominator']), (0, 1))
        self.assertEqual(r['baseline_MRR'], .5)
        self.assertEqual((r['baseline_absent'], r['candidate_absent'], r['recovered'], r['displaced']), (1, 1, 1, 1))

    def test_mode_loss_cannot_be_hidden_by_whole_panel_tie(self):
        before, after = [0, 2, 1], [1, 0, 2]
        self.assertTrue(evaluator.exact_point(before, after, range(3))['nonnegative'])
        self.assertTrue(evaluator.exact_point(before, after, [0])['nonnegative'])
        self.assertFalse(evaluator.exact_point(before, after, [1, 2])['nonnegative'])

    def test_exact_fraction_cancellation(self):
        r = evaluator.exact_point([3, 6], [2, 0], range(2))
        self.assertEqual((r['numerator'], r['denominator']), (0, 1))
        self.assertTrue(r['nonnegative'])
        self.assertFalse(evaluator.exact_point([23], [24], [0])['nonnegative'])

    def test_empty_stratum_and_accidental_run_rejected(self):
        with self.assertRaises(AssertionError): evaluator.exact_point([], [], [])
        with patch.object(evaluator, 'verify_predictions') as verify:
            with self.assertRaises(AssertionError): evaluator.run(False)
            verify.assert_not_called()

    def test_independent_chemistry_component_algorithms_agree(self):
        import pandas as pd
        from audit_c76_confirmation import component_names
        table = pd.DataFrame(dict(molecule_id=['Q3', 'Q2', 'Q1', 'Q0'],
            key=['ethanol', 'ether', 'ethylamine', 'ethanol'], smiles=['CCO', 'COC', 'CCN', 'OCC'],
            formula=['C2H6O', 'C2H6O', 'C2H7N', 'C2H6O']))
        a = evaluator.chemistry_groups(table); b = component_names(table)
        self.assertEqual(a, b)
        self.assertEqual(a[0], a[1]); self.assertEqual(a[0], a[3])

    def test_global_intent_precedes_first_label_access_and_survives_failure(self):
        import pandas as pd
        import c74_complete_target
        ids = ['SYNTHETIC_' + str(i) for i in range(120)]
        query = pd.DataFrame(dict(molecule_id=ids, spectrum_id=ids, ionization_mode=['positive'] * 60 + ['negative'] * 60))
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name); consumption = folder / 'global-consumption.json'; labels = folder / 'synthetic-never-opened-labels.csv'
            paths = {key: str(folder / (key + '.csv')) for key in ['legacy_P01', 'legacy_P16', 'legacy_physical',
                'c76_P01', 'c76_P16', 'c76_physical', 'cpu', 'gpu']}
            for path in paths.values(): pd.DataFrame(dict(molecule_id=ids, smiles=['CCO;COC'] * 120)).to_csv(path, index=False)
            policy = dict(actual_mode_counts={'positive': 60, 'negative': 60})
            reads = []
            class FirstLabelAccess(Exception): pass
            def digest(path):
                if Path(path) == labels:
                    self.assertTrue(consumption.exists())
                    self.assertEqual(json.loads(consumption.read_text())['state'], 'started')
                    reads.append(str(path)); raise FirstLabelAccess('Stop before reading synthetic label bytes')
                return 'pinned_synthetic_hash'
            stub = SimpleNamespace(structure=lambda text: (text, text, 'C2H6O'))
            with patch.object(evaluator, 'CONFIRM', folder), patch.object(evaluator, 'CONSUMPTION', consumption), \
                patch.object(evaluator, 'LABELS', labels), patch.object(evaluator, 'rooted', side_effect=Path), \
                patch.object(evaluator, 'sha', side_effect=digest), \
                patch.object(evaluator, 'verify_predictions', return_value=({}, {}, policy, paths, {'labels_sha256': 'pinned_synthetic_hash'})), \
                patch.object(pd, 'read_parquet', return_value=query), patch.object(c74_complete_target, 'module', return_value=stub):
                with self.assertRaises(FirstLabelAccess): evaluator.run(True)
                original = consumption.read_bytes()
                with self.assertRaises(FileExistsError): evaluator.run(True)
                self.assertEqual(consumption.read_bytes(), original)
                self.assertEqual(len(reads), 1)
                self.assertFalse(labels.exists())
                self.assertFalse((folder / 'confirmation-results.json').exists())


if __name__ == '__main__':
    import os
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '2'
    import torch
    unittest.main(verbosity=2)
