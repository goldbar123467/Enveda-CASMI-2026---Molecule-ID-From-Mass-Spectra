"""Freeze the missing matched C09 cosine-only continuation control.

Same parents, folds, initialization/RNG, samples, schedule, microbatch, and
auxiliary BCE as C09T. Only the fingerprint term in candidate CE is removed.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=(ROOT/'scripts/c09_train.py').read_text(encoding='utf-8')
    block="""    coefficient=cfg['fp_score_weight']/cfg['fp_bits']
    primary=primary+coefficient*(p @ positive.T)
    hard=hard+coefficient*torch.einsum('bd,bkd->bk',p,negative)
"""
    assert source.count(block)==1
    source=source.replace(block,'    # Matched control: candidate CE uses cosine only; auxiliary BCE is unchanged.\n')
    before="sources=C08_SOURCES+['scripts/c09_train.py','configs/c09-training-20261109.json']"
    after="sources=C08_SOURCES+['scripts/c09_train.py','scripts/c09_control_train.py','scripts/build_c09_cosine_control.py','configs/c09-control-20261109.json']"
    assert source.count(before)==1;source=source.replace(before,after)
    source=source.replace("'configs/c09-training-20261109.json'","'configs/c09-control-20261109.json'")
    source=source.replace("'served-score'","'cosine-control'")
    source=source.replace('C09 served-score continuation','C09 matched cosine continuation control')
    source=source.replace('Use the same raw fingerprint-head agreement term as inference.','Cosine-only candidate score; serving fingerprint term is excluded from CE.')
    compile(source,'c09_cosine_control','exec')
    target=ROOT/'scripts/c09_control_train.py'
    if target.exists():assert target.read_text(encoding='utf-8')==source,'Do not rewrite a frozen scientific control'
    else:target.write_text(source,encoding='utf-8',newline='\n')
    original=json.loads((ROOT/'configs/c09-training-20261109.json').read_text())
    plan=dict(original,run_id='C09CT',output='outputs/local/c09-control-20261109',maximum_hours=3,
        arms=[dict(name='cosine-control',cache_precision='float16',hard_negatives=48,negative_pool=512)],
        objective='Matched C09T control: cosine-only candidate CE plus unchanged auxiliary BCE; identical two C08 parents, seed/data/RNG,5000-update schedule and microbatch192',
        automatic_submission=False)
    config=ROOT/'configs/c09-control-20261109.json'
    if config.exists():assert json.loads(config.read_text())==plan,'Frozen control plan differs'
    else:config.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('BUILT C09 MATCHED COSINE CONTROL',str(target),str(config))


if __name__=='__main__':main()
