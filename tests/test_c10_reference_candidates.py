import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from c10_reference_search import search_candidates
from external_reference_overlay import search
from develop_c10_candidates import select_recipe
from precision_cache import features


class C10ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.recipes = [dict(name='P16', reference_pool='parent', adduct_rule='any_positive'),
            dict(name='P16-adduct', reference_pool='parent', adduct_rule='same_normalized'),
            dict(name='expanded', reference_pool='expanded', adduct_rule='any_positive')]
        self.policy = dict(thresholds=dict(precursor_ppm=10, precursor_absolute_da=.002,
            fragment_tolerance_da=.015, minimum_matched_peaks=4))
        self.queries = pd.DataFrame([dict(molecule_id='a', precursor_mz=200., adduct='[M+H]1+',
            ionization_mode='positive', ms2_mzs=np.array([50., 80., 110., 140.]),
            ms2_normalized_intensities=np.array([1., 2., 3., 4.]))])

    def test_adduct_filter_searches_next_eligible_hit_and_parent_is_exact(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            meta = pd.DataFrame(dict(precursor_mz=[200.] * 3, ionization_mode=['positive'] * 3,
                adduct=['[M+Na]+', '[M+H]+', '[M+H]+'], normalized_smiles=['CC', 'CO', 'CN']))
            meta.to_parquet(folder / 'ext_meta.parquet', index=False)
            mz = np.zeros((3, 128), np.float32); intensity = mz.copy()
            mz[:, :4] = [50., 80., 110., 140.]
            intensity[:, :4] = [[1., 2., 3., 4.], [1., 2., 3., 5.], [1., 2., 3., 4.]]
            np.save(folder / 'ext_mz.npy', mz); np.save(folder / 'ext_in.npy', intensity)
            actual, report = search_candidates(self.queries, folder, 2, self.recipes, features, self.policy)
            # Compare parent implementation on precisely the two parent rows.
            parent = folder / 'parent'; parent.mkdir()
            meta.iloc[:2].to_parquet(parent / 'ext_meta.parquet', index=False)
            np.save(parent / 'ext_mz.npy', mz[:2]); np.save(parent / 'ext_in.npy', intensity[:2])
            expected, _ = search(self.queries, parent, features, self.policy)
            columns = ['molecule_id', 'view', 'cosine', 'matched_peaks', 'smiles', 'reference_row']
            pd.testing.assert_frame_equal(actual[actual.recipe.eq('P16')][columns].reset_index(drop=True),
                expected[columns], check_dtype=False)
            best = actual[actual.view.eq('all_reference')].set_index('recipe')
            self.assertEqual(best.loc['P16', 'smiles'], 'CC')
            self.assertEqual(best.loc['P16-adduct', 'smiles'], 'CO')
            self.assertTrue(best.loc['P16-adduct', 'normalized_adduct_match'])
            excluded = actual[actual.view.eq('input_alias_excluded')].set_index('recipe')
            self.assertEqual(excluded.loc['expanded', 'smiles'], 'CO')
            self.assertEqual(report['query_input_alias_reference_rows'], 2)

    def test_labels_are_rejected_at_search_boundary(self):
        with self.assertRaises(AssertionError):
            search_candidates(self.queries.assign(key='secret'), Path('unused'), 1,
                self.recipes, features, self.policy)

    def test_predeclared_ties_keep_parent_including_numerical_noise(self):
        self.assertEqual(select_recipe({'P16': .5, 'P16-adduct': .5, 'expanded': .5 + 1e-14}, self.recipes)['name'], 'P16')
        self.assertEqual(select_recipe({'P16': .5, 'P16-adduct': .6, 'expanded': .6}, self.recipes)['name'], 'P16-adduct')


if __name__ == '__main__': unittest.main()
