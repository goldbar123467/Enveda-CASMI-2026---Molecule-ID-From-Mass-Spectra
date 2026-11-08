"""Archive compact C08 preparation evidence and finish its tracked pilot rows."""
import csv,hashlib,json,shutil,statistics
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    spec=read(ROOT/'configs/compatible-training-20261108.json');out=ROOT/spec['output'];cache=ROOT/spec['cache']
    pilot=ROOT/'outputs/local/smoke-compatible-20261108';results=read(pilot/'results.json');assert len(results['models'])==8
    assert all(m['steps']==200 and m['amp_skips']==0 and m['smoke'] for m in results['models'])
    for p,v in read(out/'prepared.json')['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v
    assert not (out/'launch.json').exists(),'This receipt describes preparation, not a launched full run'
    stop=read(pilot/'status.json');assert stop['state']=='complete'
    lines=(ROOT/'outputs/local/compatible-20261108-smoke-stop.log').read_text().splitlines()
    boundary=[json.loads(line) for line in lines if line.startswith('{')][-1];assert boundary['step']==100
    first=read(pilot/'models/legacy16-default/fold-0/history.json');assert [r['step'] for r in first]==[100,200]
    evidence=ROOT/'docs/evidence/compatible-2026-11-08';evidence.mkdir(parents=True,exist_ok=True)
    timings=[];history={};rates={}
    for model in results['models']:
        path=pilot/'models'/model['arm']/f"fold-{model['fold']}"/'history.json';rows=read(path)
        history[f"{model['arm']}/fold-{model['fold']}"]=rows
        durations=[r['seconds']/r['updates'] for r in rows];pace=statistics.median(durations);timings.append(pace*spec['training']['steps'])
        rates[f"{model['arm']}/fold-{model['fold']}"]=spec['training']['batch_size']/pace
    actual=dict(checked_utc=datetime.now(timezone.utc).isoformat(),state='prepared',full_training_started=False,
        full_supervisor_running=False,automatic_submission=False,production_models=8,production_updates=320000,
        pilot_models=8,pilot_successful_updates=1600,pilot_amp_skips=0,
        checkpoint_restart=dict(stopped_at_successful_update=100,resumed_to=200,history_steps=[r['step'] for r in first],
             note='Actual process exit75, then fresh-process restore of model/optimizer/scaler/RNG; no initial100 updates repeated in count'),
        real_loader_spectra_per_second=rates,peak_reserved_vram_gib=max(m['gpu_peak_reserved_gib'] for m in results['models']),
        estimated_optimizer_hours=sum(timings)/3600,planning_eta_hours=[12,15],maximum_wall_hours=24,
        limits='Short pilot; not a model-quality experiment. Actual full run and pipeline validation have not run.',
        sources=read(out/'prepared.json')['source_sha256'],smoke_results=results,history=history)
    save(evidence/'execution.json',actual)
    if (out/'launch-ready.json').exists():shutil.copy2(out/'launch-ready.json',evidence/'launch-ready.json')
    for name in ['prepared.json','selection.json','identity-audit.json','fingerprint-parity.json','precision-audit.json','pools-fold-0.json','pools-fold-1.json']:
        shutil.copy2(cache/name,evidence/name)
    for name in ['cohorts-ingest_lib.csv','cohorts-ionization_mode.csv','cohorts-adduct.csv']:shutil.copy2(cache/name,evidence/name)
    for name in ['benchmark.json','benchmark-large.json']:shutil.copy2(ROOT/'outputs/local/compatible-20261108'/name,evidence/name)
    v7=ROOT/'outputs/research/2026-11-08-scan/v7';public=ROOT/'docs/evidence/scan-2026-11-08'
    for name in ['config_used.json','data_audit.json','training_result.json']:shutil.copy2(v7/name,public/('v7-'+name))
    save(public/'v7-file-hashes.json',dict(source='https://www.kaggle.com/datasets/bgunshui/casmi26-scorer-v7-model',
          files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in v7.iterdir() if p.is_file()},
          weights_downloaded_or_used=False))
    path=ROOT/'runs.csv'
    with path.open(newline='',encoding='utf-8') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
    updates={
        'B08':'Shape-only wide320 AMP benchmark: batch96/micro96 about2817 spectra/sec for48 hard negatives,0.623GiB peak reserved; smaller microbatches slower. Not model quality.',
        'B08L':'Shape-only H48 benchmark: batch384 3519.96 spectra/sec,2.238GiB peak reserved; batch768 3626.30/s,4.400GiB. Chose384 for headroom; excludes full per-step data loading/weighted objective.',
        'C08P':'Prepared1,504,015 usable FP32 views;270,127 eligible identities;999 shared-input aliases excluded,64 fingerprint parity checks exact.2 training-only negative pools; all pool entries regenerated and byte-verified after non-pool source edits. Full training not started.',
        'C08S':f"8 recipes/folds x200 successful updates=1600;0 AMP skips; deliberate100-step checkpoint exit/resume passed. Peak reserved{actual['peak_reserved_vram_gib']:.4f}GiB; estimated optimizer{actual['estimated_optimizer_hours']:.2f}h, planned12-15h/24h cap. Not quality validation."
    }
    for row in rows:
        if row['id'] in updates:row.update(status='complete',ended_utc=now,result=updates[row['id']])
        if row['id'] in ('P13','P14R2'):
            score='0.480' if row['id']=='P13' else '0.458'
            row['result']=row['result'].replace('score pending','public score'+score)
            if 'Actual public score' not in row['result']:row['result']+=f' Actual public score{score}, COMPLETE/no reported error; verified8Nov21:27UTC.'
    with path.open('w',newline='',encoding='utf-8') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(evidence=str(evidence),optimizer_hours=actual['estimated_optimizer_hours'],peak_vram=actual['peak_reserved_vram_gib'])))


if __name__=='__main__':main()
