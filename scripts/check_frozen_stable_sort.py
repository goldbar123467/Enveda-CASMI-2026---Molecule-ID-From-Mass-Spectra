"""Verify the fixed selected fusion under deterministic serving tie selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json,sys
from score_frozen_channel import preprocess
from frozen_model import ProjectionRanker
from tail_fusion import paired_summary
from qualify_frozen_fusion import z,rr

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/local/long-frozen-20261107-r1';OUT=RUN/'qualification'


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    selected=pd.read_parquet(RUN/'selected.parquet');c=np.load(RUN/'candidates.npz',allow_pickle=False)
    selected=selected[selected.valid & selected.mol_idx.isin(c['evaluation_ids'])]
    lookup={int(row):int(index) for index,row in zip(selected.index,selected.row)}
    original=np.load(RUN/'spectra.npy',mmap_mode='r');vectors=np.array(np.load(RUN/'spectrum_embeddings.npy',mmap_mode='r')[selected.index],dtype=np.float32)
    location={int(row):i for i,row in enumerate(selected.index)};changed=[];inputs=[];offset=0
    cols=['ms2_mzs','ms2_normalized_intensities','precursor_mz','adduct','ionization_mode']
    for batch in pq.ParquetFile(ROOT/'outputs/competition-raw/train.parquet').iter_batches(batch_size=8192,columns=cols):
        positions=[row-offset for row in lookup if offset<=row<offset+len(batch)]
        if positions:
            frame=batch.take(pa.array(positions)).to_pandas()
            for pos,row in zip(positions,frame.itertuples(index=False)):
                index=lookup[offset+pos];value=preprocess(row);assert value is not None
                if not np.array_equal(value,original[index]):changed.append(index);inputs.append(value)
        offset+=len(batch)
    print('STABLE SORT CHANGED',len(changed),'of',len(selected),'views; fixed head/blend, no reselection',flush=True)
    encoder=torch.jit.load(str(ROOT/'outputs/publish/frozen-p14-20261107/dreams.ts'),map_location='cuda').eval()
    for start in range(0,len(changed),8):
        value=encoder(torch.tensor(np.stack(inputs[start:start+8]),device='cuda')).float().cpu().numpy().astype(np.float16).astype(np.float32)
        assert np.isfinite(value).all()
        for i,row in enumerate(changed[start:start+8]):vectors[location[row]]=value[i]
    del encoder;torch.cuda.empty_cache()
    state=torch.load(RUN/'models/seed-202611071/best.pt',map_location='cpu',weights_only=True);cfg=state['cfg']
    model=ProjectionRanker(cfg['hidden'],cfg['dimension'],cfg['dropout']).cuda().eval();model.load_state_dict(state['state_dict'])
    meta=np.load(RUN/'metadata.npy',mmap_mode='r');adduct=np.load(RUN/'adduct.npy',mmap_mode='r');molecule=np.load(RUN/'molecule_embeddings.npy',mmap_mode='r')
    scores=[]
    for mid,choices in zip(c['evaluation_ids'],c['evaluation_candidates']):
        rows=selected.index[selected.mol_idx==mid].to_numpy();q=model.encode_spectra(torch.tensor(vectors[[location[int(i)] for i in rows]],device='cuda'),
             torch.tensor(np.array(meta[rows]),device='cuda'),torch.tensor(np.array(adduct[rows]),device='cuda')).mean(0)
        m=model.encode_molecules(torch.tensor(np.array(molecule[choices],dtype=np.float32),device='cuda'))
        scores.append((m@q/model.log_temperature.exp().clamp(.01,1)).cpu().numpy())
    scores=np.asarray(scores);baseline=np.load(OUT/'scores.npz',allow_pickle=False)['baseline'];n=256
    before=rr(baseline);after=rr(.5*z(baseline)+.5*z(scores))
    report=dict(changed_views=len(changed),evaluated_views=len(selected),head=202611071,weight=.5,reselected=False,
                development=paired_summary(before[:n],after[:n]),confirmation=paired_summary(before[n:],after[n:]),
                limitation='Robustness check of a fixed blend after platform-dependent cutoff ties; reused reserved diagnostic, not a new untouched test')
    report['positive_confirmation']=report['confirmation']['delta']>0 and report['confirmation']['ci95'][0]>0
    assert report['positive_confirmation']
    (OUT/'stable_sort_report.json').write_text(json.dumps(report,indent=2));np.savez(OUT/'stable_sort_scores.npz',scores=scores,before=before,after=after)
    print('DONE STABLE SORT CHECK',json.dumps(report),flush=True)


if __name__=='__main__':main()
