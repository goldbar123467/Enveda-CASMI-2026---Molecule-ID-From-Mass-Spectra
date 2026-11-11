"""Encode C39 queries with the already pinned frozen DreaMS; no labels or fit.

Run in .venv-encoders. Unsupported acquisitions keep explicit masks; their
queries must retain the serving parent instead of receiving invented scores.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import json,hashlib,inspect,gc
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write,features
from training_monitor import system_memory

OUT=ROOT/'outputs/local/c40-crossmodal-heads-20261111'
ENC=ROOT/'outputs/pretrained/frozen-20261107'


def authorize():
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active'],'Current goal stopped'
    assert system_memory()['free_gib']>=4


def aggregate(frame,embeddings,valid,metadata,adduct):
    rows=[];matrix=[]
    for mid,g in frame.groupby('molecule_id',sort=False):
        ids=g.index.to_numpy();ids=ids[valid[ids]]
        value=np.zeros(1039,np.float32)
        if len(ids):
            value[:1024]=np.asarray(embeddings[ids],np.float32).mean(0)
            value[1024:1028]=metadata[ids].mean(0)
            value[1028:]=np.bincount(adduct[ids],minlength=11)/len(ids)
        rows.append(dict(molecule_id=mid,total_acquisitions=len(g),supported_positive_acquisitions=len(ids),supported=bool(len(ids))))
        matrix.append(value)
    return pd.DataFrame(rows),np.asarray(matrix,np.float32)


def main():
    authorize();torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name(0)
    torch.cuda.set_per_process_memory_fraction(.55)
    assert not (OUT/'query-encoding-freeze.json').exists()
    manifest=json.loads((ENC/'manifest.json').read_text())
    weight=next(x for x in manifest['files'] if x['file'].replace('\\','/')=='dreams/embedding_model.ckpt')
    assert sha(ENC/weight['file'])==weight['sha256']
    import dreams.api as api
    from dreams.models.heads.heads import ContrastiveHead
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.utils.dformats import DataFormatA
    processor=SpectrumPreprocessor(DataFormatA(),n_highest_peaks=100)
    api.PRETRAINED=ENC/'dreams'
    protocol=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),encoder_manifest_sha256=sha(ENC/'manifest.json'),
        source_sha256=sha(__file__),weight_sha256=weight['sha256'],spectrum_processor_source_sha256=hashlib.sha256(inspect.getsource(SpectrumPreprocessor).encode()).hexdigest(),
        query_path='outputs/publish/c39-source-supported-validation-20261111/validation-queries.parquet',
        query_sha256=sha(ROOT/'outputs/publish/c39-source-supported-validation-20261111/validation-queries.parquet'),
        model_frozen=True,labels_read=False,quality_metrics_computed=False,
        support='Positive supported singly charged legacy adducts, precursor<=1000Th, at least3 finite positive fragments<=1000Th; negatives retain parent fallback.',
        old_R1_processor_and_weights_reused=True,old_trained_retrieval_head_not_used=True)
    write(OUT/'query-encoding-protocol.json',protocol)
    print('C40 LOADING PINNED FROZEN DreaMS',flush=True)
    model=api.PreTrainedModel.from_ckpt(ENC/'dreams/embedding_model.ckpt',ContrastiveHead,100).model.cuda().eval()
    model.requires_grad_(False)
    q=pd.read_parquet(ROOT/protocol['query_path']).reset_index(drop=True)
    values=np.zeros((len(q),1024),np.float16);metadata=np.zeros((len(q),4),np.float32);adduct=np.zeros(len(q),np.int64);valid=np.zeros(len(q),bool)
    allowed={'[M+H]+','[M+Na]+','[M+K]+','[M+NH4]+','[M-H2O+H]+','[M-2H2O+H]+'}
    payload=[];indices=[];reasons={}
    for i,r in enumerate(q.itertuples(index=False)):
        if r.ionization_mode!='positive' or r.adduct not in allowed or not 0<float(r.precursor_mz)<=1000:
            reasons['unsupported_mode_adduct_or_precursor']=reasons.get('unsupported_mode_adduct_or_precursor',0)+1;continue
        mz=np.asarray(r.ms2_mzs,float);it=np.asarray(r.ms2_normalized_intensities,float)
        good=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(mz<=1000)&(it>0)
        if good.sum()<3:reasons['insufficient_supported_peaks']=reasons.get('insufficient_supported_peaks',0)+1;continue
        payload.append(processor(np.stack([mz[good],it[good]],1),float(r.precursor_mz),high_form=True))
        indices.append(i);metadata[i],adduct[i]=features.metadata(r.precursor_mz,r.adduct,r.ionization_mode,r.collision_energy_ev)
    assert indices
    with torch.inference_mode():
        for start in range(0,len(indices),8):
            authorize();ix=np.asarray(indices[start:start+8]);x=torch.tensor(np.asarray(payload[start:start+8]),device='cuda')
            embedding=model(x).float().cpu().numpy();assert embedding.shape==(len(ix),1024) and np.isfinite(embedding).all()
            values[ix]=embedding.astype(np.float16);valid[ix]=True
            if start%64==0:print('C40 ENCODED',start+len(ix),'of',len(indices),flush=True)
    summary,matrix=aggregate(q,values,valid,metadata,adduct)
    np.save(OUT/'query-spectrum-embeddings.npy',values)
    np.savez(OUT/'query-aggregate-features.npz',features=matrix,supported=summary.supported.to_numpy())
    summary.to_csv(OUT/'query-encoding-coverage.csv',index=False)
    q[['molecule_id','spectrum_id']].assign(embedding_valid=valid).to_csv(OUT/'spectrum-encoding-coverage.csv',index=False)
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'query-encoding-protocol.json'),
        queries=len(summary),acquisitions=len(q),supported_queries=int(summary.supported.sum()),supported_acquisitions=int(valid.sum()),
        unsupported_acquisition_reasons=reasons,maximum_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        files_sha256={n:sha(OUT/n) for n in ['query-spectrum-embeddings.npy','query-aggregate-features.npz','query-encoding-coverage.csv','spectrum-encoding-coverage.csv']},
        labels_read=False,fitted=False,scored=False,reserved_C13_read=False,model_weights_frozen=True)
    write(OUT/'query-encoding-freeze.json',report);print('DONE C40 FROZEN QUERY ENCODING',json.dumps(report),flush=True)


if __name__=='__main__':main()
