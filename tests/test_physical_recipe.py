from pathlib import Path
import sys
import types
import unittest

try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError:
    np = pd = torch = None

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


@unittest.skipIf(np is None or pd is None or torch is None,'numeric environment required')
class PhysicalRecipeTests(unittest.TestCase):
    def run_policy(self,policy,lib=.5,physical_move=False):
        from physical_recipe_replay import final_order
        group = pd.DataFrame(dict(molecule_id=['m']*3,smiles=['A','B','C'],metric_key=['a','b','c'],baseline_rank=[1,2,3],library_max=[lib]*3))
        annotation = pd.DataFrame(dict(pop=[0,0,0],np_coconut=[0,1,0],np_lotus=[0,0,0],src_natural_product=[0,0,0],src_spectral_library=[0,0,0]),index=['a','b','c'])
        def decision(mid,control,*args,**kwargs):
            return ([control[-1]]+control[:-1] if physical_move else control),dict(top1_changed=physical_move)
        return final_order(group,np.array([.5,1.,0.]),np.zeros(3),np.zeros(3),None,types.SimpleNamespace(decision=decision),types.SimpleNamespace(),annotation,True,policy)

    def test_original_recipe_keeps_leader(self):
        order,_=self.run_policy('479')
        self.assertEqual(order[0],0)

    def test_fixed_np_promotion_requires_low_library_match(self):
        order,diag=self.run_policy('481')
        self.assertEqual(order[:2],[1,0])
        self.assertTrue(diag['np_promoted'])
        order,diag=self.run_policy('481',lib=.80)
        self.assertEqual(order[0],0)
        self.assertFalse(diag['np_promoted'])

    def test_np_prior_cannot_override_physical_leader(self):
        order,diag=self.run_policy('481',physical_move=True)
        self.assertEqual(order[:2],[2,0])
        self.assertFalse(diag['np_promoted'])


if __name__=='__main__':
    unittest.main()
