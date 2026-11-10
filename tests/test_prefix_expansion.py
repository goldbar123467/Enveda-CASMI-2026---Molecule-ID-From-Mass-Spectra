import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prefix_expansion import expand


class PrefixExpansionTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(protect_library_at=.9, rrf_offset=3., rrf_weight=.2, maximum_insertions=3)
        self.before = ['s' + str(i) for i in range(25)]
        self.parent = pd.DataFrame(dict(molecule_id=['a'], smiles=[';'.join(self.before)]))

    def pool(self, extra=1):
        smiles = self.before + ['extra' + str(i) for i in range(extra)]
        return pd.DataFrame(dict(molecule_id='a', smiles=smiles, metric_key=smiles,
            prior_rank=range(1, len(smiles)+1), is_parent=[True]*25+[False]*extra,
            library_max=.1, neural_ok=True, neural_score=[0.]*25+[100.]*extra))

    def test_insert_preserves_rank1_and_exact_parent_prefix(self):
        result, actions = expand(self.parent, self.pool(), self.config)
        after = result.smiles.iloc[0].split(';')
        self.assertEqual(after, self.before[:11] + ['extra0'] + self.before[11:24])
        self.assertEqual(actions.inserted.iloc[0], 1)

    def test_at_most_three_extras_and_no_reordered_or_skipped_parent(self):
        result, actions = expand(self.parent, self.pool(8), self.config)
        after = result.smiles.iloc[0].split(';'); n = int(actions.inserted.iloc[0])
        self.assertLessEqual(n, 3); self.assertEqual(len(after), 25); self.assertEqual(after[0], 's0')
        self.assertEqual([s for s in after if s.startswith('s')], self.before[:25-n])

    def test_protected_unsupported_and_no_outsider_paths_are_exact(self):
        for group in [self.pool().assign(library_max=.95), self.pool().assign(library_max=float('nan')), self.pool().assign(neural_ok=False),
            self.pool().assign(neural_score=[100.]*25+[0.])]:
            result, _ = expand(self.parent, group, self.config)
            self.assertTrue(result.equals(self.parent))

    def test_unprocessed_row_without_pool_is_exact(self):
        parent = self.parent.assign(molecule_id='inactive')
        result, actions = expand(parent, self.pool(), self.config)
        self.assertTrue(result.equals(parent)); self.assertEqual(actions.reason.iloc[0], 'no_pool')


if __name__ == '__main__': unittest.main()
