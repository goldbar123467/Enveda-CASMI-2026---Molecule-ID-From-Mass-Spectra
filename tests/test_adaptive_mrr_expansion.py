import sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adaptive_mrr_expansion import expand


class AdaptiveUtilityTests(unittest.TestCase):
    def test_exhaustive_utility_matches_independent_list_calculation(self):
        rng=np.random.default_rng(202611190)
        for n in [1,2,10,24,25]:
            names=['s'+str(i) for i in range(n+3)];p=rng.dirichlet(np.ones(n+4))[:-1]
            parent=pd.DataFrame(dict(molecule_id=['q'],smiles=[';'.join(names[:n])]))
            pool=pd.DataFrame(dict(molecule_id='q',smiles=names,metric_key=names,prior_rank=range(1,n+4),
                is_parent=[True]*n+[False]*3,library_max=.1,probability=p))
            cfg=dict(protect_library_at=.9,minimum_expected_mrr_gain=1e-6);frame,action=expand(parent,pool,cfg)
            selected=names[n+int(np.argmax(p[n:]))];before=names[:n]
            def score(values):return sum(p[names.index(s)]/(i+1) for i,s in enumerate(values))
            best=max([(score((before[:pos-1]+[selected]+before[pos-1:])[:25])-score(before),pos) for pos in range(2,min(n+1,25)+1)])
            self.assertAlmostEqual(action.estimated_mrr_gain.iloc[0],best[0])
            if best[0]<=1e-6:self.assertTrue(frame.equals(parent))
            else:self.assertEqual(frame.smiles.iloc[0].split(';'),(before[:best[1]-1]+[selected]+before[best[1]-1:])[:25])

    def test_library_protection_is_exact(self):
        parent=pd.DataFrame(dict(molecule_id=['q'],smiles=['a;b']))
        pool=pd.DataFrame(dict(molecule_id='q',smiles=['a','b','c'],metric_key=['a','b','c'],prior_rank=[1,2,3],
            is_parent=[True,True,False],library_max=.9,probability=[0.,0.,1.]))
        self.assertTrue(expand(parent,pool,dict(protect_library_at=.9,minimum_expected_mrr_gain=1e-6))[0].equals(parent))


if __name__=='__main__':unittest.main()
