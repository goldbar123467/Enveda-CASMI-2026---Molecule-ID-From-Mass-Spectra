"""Label-blind C24 CPU/CUDA and independent serving-arithmetic audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import gc,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from precision_cache import ROOT,sha,write,features
from c24_models import PACKAGE,RUN,serving_module


@torch.inference_mode()
def main():
    import msvcrt
    lock=(RUN/'trainer.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (ROOT/'.local/compute-paused.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.7)
    out=ROOT/'outputs/local/c24-serving-audit-20261110';out.mkdir(exist_ok=True)
    assert not (out/'audit.json').exists()
    manifest,module=serving_module();model_class,_=module.load_modules(PACKAGE)
    positive=ROOT/'outputs/forks/c10-casmi2022-validation-v1'
    qpos=pd.read_parquet(positive/'c10_queries.parquet').iloc[:1]
    qneg=pd.read_parquet(ROOT/'outputs/publish/c20-negative-casmi2022-20261110/validation-queries.parquet')
    ids=list(dict.fromkeys([qneg.molecule_id.iloc[0],qneg[qneg.adduct.eq('[M+FA-H]-')].molecule_id.iloc[0],
        qneg[qneg.adduct.eq('[M+Cl]-')].molecule_id.iloc[0],'C20Q_3f38081e510120c5bad2']))
    q=pd.concat([qpos,qneg.set_index('molecule_id').loc[ids].reset_index()],ignore_index=True)
    parent=pd.concat([pd.read_csv(positive/'c10_p01.csv',dtype=str,keep_default_na=False),
        pd.read_csv(ROOT/'outputs/forks/c20-negative-development-v1/baseline_submission.csv',dtype=str,keep_default_na=False)],ignore_index=True)
    parent=parent.set_index('molecule_id').loc[q.molecule_id].reset_index()
    assert parent.molecule_id.tolist()==q.molecule_id.tolist() and len(q)==5
    # V1's partial pipeline lists are fixtures only. They are never a qualified
    # full80 baseline, and neither this audit nor its scores are a quality gate.
    q.to_parquet(out/'queries.parquet',index=False);parent.to_csv(out/'fixture-lists.csv',index=False)
    unique=list(dict.fromkeys(';'.join(parent.smiles).split(';')))
    shared={s:features.structure(s)[-1] for s in unique}
    scores={};receipts={}
    for device in ['cpu','cuda']:
        scores[device],receipts[device]=module.score(PACKAGE,'polarity-loss',parent,q,out/(device+'-scores.parquet'),device=device,shared_fps=shared)
        assert receipts[device]['scored_molecules']==5 and receipts[device]['input_errors']==0 and scores[device].c08_ok.all()
    fields=['c08_raw_fold0','c08_raw_fold1','c08_score']
    parity={c:float(np.max(np.abs(scores['cpu'][c]-scores['cuda'][c]))) for c in fields}
    assert max(parity.values())<1e-4,parity
    raw_errors=[];normalized={mid:[] for mid in parent.molecule_id}
    for entry in manifest['models']:
        ck=torch.load(PACKAGE/entry['file'],map_location='cpu',weights_only=True);cfg=ck['cfg']
        model=model_class(cfg).cpu().eval();model.load_state_dict(ck['state_dict'])
        for row in parent.itertuples(index=False):
            query=q[q.molecule_id.eq(row.molecule_id)].iloc[0]
            x,n=features.peaks(query.ms2_mzs,query.ms2_normalized_intensities,query.precursor_mz,96)
            x=x.astype(np.float16).astype(np.float32)
            meta,adduct=features.metadata(query.precursor_mz,query.adduct,query.ionization_mode,query.collision_energy_ev)
            z,p=model.encode_spectra(torch.from_numpy(x[None]),torch.tensor([n]),torch.from_numpy(meta[None]),torch.tensor([adduct]))
            fp=torch.tensor(np.stack([shared[s] for s in row.smiles.split(';')]),dtype=torch.float32)
            molecule=model.encode_molecules(fp)
            raw=np.array([float(torch.dot(m,z[0])/cfg['temperature']+cfg['fp_score_weight']*torch.dot(f,p[0])/cfg['fp_bits']) for m,f in zip(molecule,fp)])
            expected=scores['cpu'][scores['cpu'].molecule_id.eq(row.molecule_id)]['c08_raw_fold'+str(entry['fold'])].to_numpy()
            raw_errors.append(float(np.max(np.abs(raw-expected))))
            normalized[row.molecule_id].append((raw-raw.mean())/max(float(raw.std()),1e-6))
        del model,ck;gc.collect()
    assert max(raw_errors)<1e-4
    aggregate_error=max(float(np.max(np.abs(np.mean(v,axis=0)-scores['cpu'][scores['cpu'].molecule_id.eq(mid)].c08_score.to_numpy()))) for mid,v in normalized.items())
    assert aggregate_error<1e-4
    write(out/'audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),queries=5,positive=1,negative=4,
        includes_formate_chloride_high_mass=True,CPU_CUDA_max_abs_error=parity,independent_raw_max_abs_error=max(raw_errors),
        independent_standardized_mean_max_abs_error=aggregate_error,model_manifest_sha256=sha(PACKAGE/'c08-manifest.json'),
        audit_source_sha256=sha(__file__),query_sha256=sha(out/'queries.parquet'),fixture_sha256=sha(out/'fixture-lists.csv'),
        fixture_scope='One completed positive baseline and four real partial-v1 negative lists, used only for arithmetic/feature acceptance, never full-pipeline qualification.',
        labels_read=False,quality_metrics_computed=False,C13_confirmation_metrics_read=False,submission_qualified=False))
    print('DONE C24 INDEPENDENT SERVING AUDIT',json.dumps(json.loads((out/'audit.json').read_text())),flush=True)


if __name__=='__main__':main()
