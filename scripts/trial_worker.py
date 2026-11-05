"""Dispatch a structured, checked training plan through the Python API."""
import json
import sys


def dispatch(plan, trainer):
    assert plan['stage'] in ('holdout','refit')
    assert plan['split'] in ('kit','np')
    assert plan['stage']!='refit' or plan['split']=='kit'
    assert plan['seed']>=0 and plan['epochs']>0
    assert plan['width']%plan['heads']==0
    architecture={k:int(plan[k]) for k in ['width','heads','layers','embedding']}
    assert min(architecture.values())>0
    trainer.OVERRIDES.update(architecture)
    trainer.train(plan['stage'],[plan['seed']],plan['epochs'],plan['split'])


if __name__=='__main__':
    import local_seeds
    dispatch(json.loads(sys.argv[1]),local_seeds)
