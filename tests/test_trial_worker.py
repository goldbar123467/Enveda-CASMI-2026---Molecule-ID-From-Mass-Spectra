from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from trial_worker import dispatch


class TrainingDispatchTests(unittest.TestCase):
    def test_full_refit_and_excluded_architecture_replica(self):
        calls=[]
        trainer=types.SimpleNamespace(OVERRIDES={},train=lambda *args:calls.append(args))
        plans=[dict(stage='refit',split='kit',seed=20,epochs=128,width=256,heads=8,layers=4,embedding=192),
               dict(stage='holdout',split='np',seed=0,epochs=109,width=256,heads=8,layers=3,embedding=256)]
        for plan in plans:
            dispatch(plan,trainer)
            self.assertEqual(calls[-1],(plan['stage'],[plan['seed']],plan['epochs'],plan['split']))
            self.assertEqual(trainer.OVERRIDES,{k:plan[k] for k in ['width','heads','layers','embedding']})

    def test_refit_cannot_claim_an_exclusion_split(self):
        trainer=types.SimpleNamespace(OVERRIDES={},train=lambda *args:self.fail('Invalid plan dispatched'))
        with self.assertRaises(AssertionError):
            dispatch(dict(stage='refit',split='np',seed=20,epochs=128,width=256,heads=8,layers=4,embedding=192),trainer)


if __name__=='__main__':
    unittest.main()
