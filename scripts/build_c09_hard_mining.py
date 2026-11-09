"""Freeze a useful longer RTX2070 follow-up: online hard negatives and dose checkpoints."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=(ROOT/'scripts/c09_train.py').read_text(encoding='utf-8')
    patches={
        "from torch.nn import functional as F":"from torch.nn import functional as F\nfrom online_hard_negatives import mine",
        "'configs/c09-training-20261109.json'":"'configs/c09-mined-20261109.json'",
        "sources=C08_SOURCES+['scripts/c09_train.py','configs/c09-mined-20261109.json']":"sources=C08_SOURCES+['scripts/c09_train.py','scripts/c09_hard_train.py','scripts/online_hard_negatives.py','scripts/build_c09_hard_mining.py','configs/c09-mined-20261109.json']",
        "'served-score'":"'mined128-served'",
        "[:,:cfg['hard_negatives']]":"[:,:spec['mining']['sampled_negatives']]",
        "batch,pos,neg,mask=data.batch(ids,negative,qr,'float16')":"batch,pos,pool_negative,pool_mask=data.batch(ids,negative,qr,'float16')\n                neg,mask,mining_stats=mine(model,batch,pos,pool_negative,pool_mask,cfg,cfg['hard_negatives'])\n                del pool_negative,pool_mask",
        "factor=min(1.,(step+1)/cfg['warmup_steps'])\n                if step>=cfg['warmup_steps']:factor=.5*(1+math.cos(math.pi*(step-cfg['warmup_steps'])/(cfg['steps']-cfg['warmup_steps'])))":"cycle_step=step%spec['cycle_steps']\n                factor=min(1.,(cycle_step+1)/cfg['warmup_steps'])\n                if cycle_step>=cfg['warmup_steps']:factor=.5*(1+math.cos(math.pi*(cycle_step-cfg['warmup_steps'])/(spec['cycle_steps']-cfg['warmup_steps'])))",
        "gradient_norm=value['gradient_norm'])":"gradient_norm=value['gradient_norm'],**mining_stats)",
        "updates_per_second=row['updates_per_second']))":"updates_per_second=row['updates_per_second'],mining_gain=mining_stats['mining_gain'],selected_valid_min=mining_stats['selected_valid_min']))",
        "if step and step%cfg['checkpoint_interval']==0 and not value['skipped']:save()":"if step and step%cfg['checkpoint_interval']==0 and not value['skipped']:save()\n                if step and step%spec['cycle_steps']==0 and not value['skipped']:\n                    atomic_torch(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,steps=40000+step,continuation_steps=step,parent_sha256=sha(base),fold=fold,smoke=False),folder/f'milestone-{step}.pt')"
    }
    for before,after in patches.items():
        assert before in source,before
        source=source.replace(before,after)
    source=source.replace('C09 served-score continuation','C09 online hard-negative continuation')
    compile(source,'c09_hard_mining','exec')
    target=ROOT/'scripts/c09_hard_train.py'
    if target.exists():assert target.read_text(encoding='utf-8')==source,'Frozen scientific source differs'
    else:target.write_text(source,encoding='utf-8',newline='\n')
    original=json.loads((ROOT/'configs/c09-training-20261109.json').read_text())
    plan=dict(original,run_id='C09H',output='outputs/local/c09-mined-20261109',maximum_hours=6,
        arms=[dict(name='mined128-served',cache_precision='float16',hard_negatives=48,negative_pool=512)],
        training=dict(original['training'],steps=20000),cycle_steps=5000,
        mining=dict(sampled_negatives=128,selected_negatives=48,scoring='Current eval-mode cosine+fingerprint served score; stable ties; no gradient or RNG draws',
            masks='All logical-batch positives and equal positive Morgan fingerprints; masked slots remain masked if fewer than48 valid negatives'),
        objective='Same C09T served-score CE/BCE; online48-hard-of128 instead of uniform48-of512. First5000-update cycle matches C09T schedule; later cycles explore dose with predeclared checkpoints.',
        selection='Development only; freeze model milestone/weight before viewing any new confirmation comparison. No automatic submission.')
    config=ROOT/'configs/c09-mined-20261109.json'
    if config.exists():assert json.loads(config.read_text())==plan,'Frozen mining plan differs'
    else:config.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('BUILT C09 ONLINE HARD MINING',str(target),str(config))


if __name__=='__main__':main()
