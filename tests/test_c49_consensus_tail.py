"""Check consensus as pairwise constraints, independently of bubble passes."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c49_consensus_tail import order,utilities
from crossmodal_tail import order as original


class ConsensusTests(unittest.TestCase):
    def test_agreeing_heads_equal_original_ranking_and_cpu_rule_is_exact(self):
        rng=np.random.default_rng(49)
        for _ in range(200):
            n=int(rng.integers(1,26));f=rng.integers(0,4,n).tolist();scores=rng.normal(size=n);valid=rng.random(n)>.1
            expected=original(f,scores,valid)
            self.assertEqual(order(f,scores,valid),expected)
            self.assertEqual(order(f,scores,valid,reference_scores=scores),expected)
    def test_every_inversion_is_supported_by_both_heads(self):
        rng=np.random.default_rng(77)
        for _ in range(200):
            n=25;a=rng.normal(size=n);b=rng.normal(size=n);positions=list(range(1,n))
            result=order(['f']*n,a,[True]*n,reference_scores=b);own=utilities(positions,a);reference=utilities(positions,b)
            output={pos:i for i,pos in enumerate(result)}
            for i in positions:
                for j in positions:
                    if i<j and output[j]<output[i]:self.assertTrue(own[j]>own[i] and reference[j]>reference[i])
            # Independent insertion-sort normal form for the same partial order.
            insertion=[]
            for candidate in positions:
                insertion.append(candidate);k=len(insertion)-1
                while k and own[insertion[k]]>own[insertion[k-1]] and reference[insertion[k]]>reference[insertion[k-1]]:
                    insertion[k],insertion[k-1]=insertion[k-1],insertion[k];k-=1
            self.assertEqual(result,[0]+insertion)
    def test_disagreement_and_missing_group_features_preserve_parent(self):
        f=['f']*7;a=[0,0,0,0,0,0,99];b=[99,0,0,0,0,0,-99]
        self.assertEqual(order(f,a,[True]*7,reference_scores=b),list(range(7)))
        self.assertEqual(order(f,a,[True]*6+[False],reference_scores=a),list(range(7)))
        self.assertEqual(order(f,a,[True]*7,False,reference_scores=a),list(range(7)))


if __name__=='__main__':unittest.main()
