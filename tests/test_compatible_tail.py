"""The experiment may reorder the tail, never the baseline leader or membership."""
import sys,unittest
from pathlib import Path
try:
    import numpy as np
    import pandas as pd
except ImportError:
    np=pd=None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipUnless(np is not None,'NumPy/Pandas required')
class CompatibleTailTests(unittest.TestCase):
    def test_zero_weight_and_ties_preserve_baseline(self):
        from compatible_tail import order
        self.assertEqual(order(np.arange(25),0).tolist(),list(range(25)))
        self.assertEqual(order(np.ones(25),.2).tolist(),list(range(25)))

    def test_leader_and_candidate_membership_are_protected(self):
        from compatible_tail import order
        indices=order(np.arange(25),.2)
        self.assertEqual(indices[0],0);self.assertEqual(sorted(indices.tolist()),list(range(25)))
        self.assertNotEqual(indices.tolist(),list(range(25)))

    def test_unprocessed_row_passes_through_exactly(self):
        from compatible_tail import apply
        base=pd.DataFrame(dict(molecule_id=['a','b'],smiles=['A;B;C;D;E;F','X;Y']))
        scored=pd.DataFrame(dict(molecule_id=['a']*6+['b']*2,smiles=list('ABCDEF')+list('XY'),
            baseline_rank=[1,2,3,4,5,6,1,2],c08_score=list(range(6))+[0,99],c08_ok=[True]*6+[False]*2))
        result,receipt=apply(base,scored,.2)
        self.assertEqual(result.smiles.iloc[1],'X;Y');self.assertEqual(receipt['rank_one_changes'],0)
        self.assertEqual(receipt['scored_molecules'],1);self.assertEqual(receipt['passthrough_molecules'],1)


if __name__=='__main__':unittest.main()
