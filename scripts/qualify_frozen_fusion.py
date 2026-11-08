"""Fresh held-out, fixed-candidate P01/F07 fusion check; not end-to-end recall."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/local/kit'))
from model import SpectrumCandidateModel
import features
from tail_fusion import paired_summary

RUN=ROOT/'outputs/local/long-frozen-20261107-r1'
OUT=RUN/'qualification'


def z(a):
    a=np.asarray(a,dtype=np.float64)
    return (a-a.mean(-1,keepdims=True))/np.maximum(a.std(-1,keepdims=True),1e-6)


def rr(a):
    ranks=1+(a[:,1:]>=a[:,:1]).sum(1)
    return np.where(ranks<=25,1/ranks,0.)


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    OUT.mkdir(exist_ok=True)
    c=np.load(RUN/'candidates.npz',allow_pickle=False);ids=c['evaluation_ids'];choices=c['evaluation_candidates']
    selected=pd.read_parquet(RUN/'selected.parquet')
    selected=selected[selected.valid & selected.mol_idx.isin(ids)]
    index={int(row):int(mol) for row,mol in zip(selected.row,selected.mol_idx)}
    records={int(i):[] for i in ids};offset=0
    columns=['ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','ionization_mode','collision_energy_ev']
    print('QUALIFICATION PREP',len(index),'precise query spectra',flush=True)
    for batch in pq.ParquetFile(ROOT/'outputs/competition-raw/train.parquet').iter_batches(batch_size=8192,columns=columns):
        local=[row-offset for row in index if offset<=row<offset+len(batch)]
        if local:
            frame=batch.take(pa.array(local)).to_pandas()
            for pos,row in zip(local,frame.itertuples(index=False)):
                x,n=features.peaks(row.ms2_mzs,row.ms2_normalized_intensities,row.precursor_mz,96)
                meta,adduct=features.metadata(row.precursor_mz,row.adduct,row.ionization_mode,row.collision_energy_ev)
                records[index[offset+pos]].append((x,n,meta,adduct))
        offset+=len(batch)
    cfg=json.loads((ROOT/'outputs/local/kit/config.json').read_text())
    fps=np.load(ROOT/'outputs/local/n0/cache/fingerprints.npy',mmap_mode='r')
    channels={}
    paths=dict(public=ROOT/'outputs/local/kit/final_model.pt',v80=ROOT/'outputs/research/v80-output/casmi_spectrum_model_ours_v80/final_model.pt')
    for name,path in paths.items():
        model=SpectrumCandidateModel(cfg).cuda().eval()
        state=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(state['state_dict']);del state
        scores=[]
        with torch.inference_mode():
            for i,candidates in zip(ids,choices):
                values=records[int(i)];assert values
                x=torch.tensor(np.stack([v[0] for v in values]),device='cuda')
                lengths=torch.tensor([v[1] for v in values],device='cuda')
                meta=torch.tensor(np.stack([v[2] for v in values]),device='cuda')
                adduct=torch.tensor([v[3] for v in values],device='cuda')
                fp=torch.tensor(np.asarray(fps[candidates]),device='cuda',dtype=torch.float32)
                m=model.encode_molecules(fp);s,p=model.encode_spectra(x,lengths,meta,adduct)
                values=s@m.T/cfg['temperature']+cfg['fp_score_weight']*(p@fp.T/cfg['fp_bits'])
                scores.append(values.mean(0).cpu().numpy())
        channels[name]=np.asarray(scores);assert np.isfinite(channels[name]).all()
        print('BASELINE CHANNEL',name,float(rr(channels[name]).mean()),flush=True)
        del model;torch.cuda.empty_cache()
    base=.5*z(channels['public'])+.5*z(channels['v80'])
    split=np.load(RUN/'split.npz',allow_pickle=False);n=len(split['development'])
    scores={}
    for seed in [202611071,202611072]:
        value=np.load(RUN/'models'/f'seed-{seed}'/'evaluation.npz',allow_pickle=False)
        assert np.array_equal(value['ids'],ids)
        assert np.isfinite(value['scores']).all(),'Nonfinite frozen evaluation scores'
        scores[str(seed)]=z(value['scores'])
    scores['equal_heads']=(scores['202611071']+scores['202611072'])/2
    # Select model and coefficient on development only, before confirmation comparison.
    trials=[]
    for model,value in scores.items():
        for weight in [.1,.2,.35,.5]:
            fused=(1-weight)*z(base[:n])+weight*z(value[:n])
            trials.append(dict(model=model,weight=weight,development_mrr25=float(rr(fused).mean())))
    winner=max(trials,key=lambda row:row['development_mrr25'])
    contract=dict(selection='Development-only fixed candidate fusion',winner=winner,trials=trials,
                  production_scope='Final-list neural_score channel only; preserve P01 outsiders, priors, physical rules and unsupported-query fallback',
                  limitation='Truth forced among256 mass-near candidates; no candidate-recall or end-to-end MRR claim; baseline encoders were not retrained to exclude these identities.')
    (OUT/'selection.json').write_text(json.dumps(contract,indent=2))
    fused=(1-winner['weight'])*z(base)+winner['weight']*z(scores[winner['model']])
    before,after=rr(base),rr(fused)
    report=dict(development=paired_summary(before[:n],after[:n]),confirmation=paired_summary(before[n:],after[n:]),
                winner=winner,positive_confirmation=bool(after[n:].mean()>before[n:].mean()),
                baseline_weights_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                qualification='Conditional ranking only; execution smoke and actual pipeline still required')
    np.savez(OUT/'scores.npz',ids=ids,choices=choices,baseline=base,frozen=scores[winner['model']],fused=fused,before=before,after=after,
             public=channels['public'],v80=channels['v80'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print('DONE FUSION QUALIFICATION',json.dumps(report),flush=True)


if __name__=='__main__':main()
