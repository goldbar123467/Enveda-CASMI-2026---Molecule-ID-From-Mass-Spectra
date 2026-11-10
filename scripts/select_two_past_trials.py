"""Rank retained fitted recipes on used240 for two explicit scoring trials."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from precision_cache import ROOT,sha,write
from compatible_tail import apply
from finish_c09_confirmation import metric_vector
from tail_fusion import paired_summary

OUT=ROOT/'outputs/local/two-past-trials-20261110'
BASE=ROOT/'outputs/local/c21-negative-tail-20261110'

def stamp():return datetime.now(timezone.utc).isoformat()

def main():
    OUT.mkdir(exist_ok=True)
    assert not (ROOT/'.local/compute-paused.json').exists()
    assert json.loads((ROOT/'.local/bounded-two-submissions-20261110.json').read_text())['maximum_new_submissions']==2
    if (OUT/'selection.json').exists():
        print((OUT/'selection.json').read_text());return
    torch.set_num_threads(2);pa.set_cpu_count(2)
    assert torch.cuda.is_available() and '2070' in torch.cuda.get_device_name()
    torch.cuda.set_per_process_memory_fraction(.7)
    from c22_models import serving_module,PACKAGE
    manifest,module=serving_module()
    inputs=[BASE/n for n in ['parent.csv','queries.parquet','negative-parent.csv','negative-queries.parquet','fingerprint-smiles.parquet','fingerprints.npy','P15.csv']]
    inputs += [ROOT/'outputs/local/c10-casmi2022-20261110/labels.csv',ROOT/'outputs/local/c20-negative-development-20261110/labels.csv',PACKAGE/'c08-manifest.json']
    for group,run in [('C14-CE','c14'),('C15-NP','c15'),('C18-peak-drop','c18')]:
        inputs += [BASE/(group+'-predictions.csv'),ROOT/f'outputs/local/{run}-evaluation-20261110/development.csv']
    inputs += [ROOT/'outputs/local/c22-evaluation-20261110/development.csv',ROOT/'outputs/local/c24-evaluation-20261110/development.csv',
        ROOT/'outputs/local/c28-evaluation-20261110/C28-predictions.csv',ROOT/'outputs/local/c28-evaluation-20261110/matched_C09-predictions.csv',
        ROOT/'outputs/forks/c29-public-replay-v1/c29_all_reference.csv']
    inputs += sorted(BASE.glob('*-predictions.csv'))
    plan=OUT/'selection-protocol.json'
    if not plan.exists():
        write(plan,dict(prepared_utc=stamp(),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
            source_sha256={n:sha(ROOT/'scripts'/n) for n in ['select_two_past_trials.py','compatible_tail.py','finish_c09_confirmation.py','tail_fusion.py']},
            choice='Top two unsubmitted complete fitted recipes by full240 MRR, exact 0.20/offset3; tie lexical. All labels previously used; no reserved labels.',
            candidate_pool='Existing all-mode C09/C14/C15/C18/C22/C28/C29 and previously fixed negative-only C08/C09/C14/C15/C18/C24.',
            additional_inference='Only missing C22 negative80 inference using retained final55000 checkpoints; no training or checkpoint selection.',
            excluded_C19='Only five cross-validation fold fits and OOF predictions retained; prospective final production fit was never performed. This task uses completed production fits without new training.',
            statistical_promotion=False,user_requested_trials=2,maximum_pending=1,fresh_reserved_labels_read=False))
    protocol=json.loads(plan.read_text())
    for f,pin in protocol['input_sha256'].items():assert sha(ROOT/f)==pin,f
    for f,pin in protocol['source_sha256'].items():assert sha(ROOT/'scripts'/f)==pin,f
    parent=pd.read_csv(BASE/'parent.csv',dtype=str,keep_default_na=False)
    queries=pd.read_parquet(BASE/'queries.parquet');assert len(parent)==240 and parent.molecule_id.is_unique
    positive=set(queries.loc[queries.ionization_mode.eq('positive'),'molecule_id']);assert len(positive)==160
    def read(p):return pd.read_csv(p,dtype=str,keep_default_na=False)
    def combine(negative,positive_frame):
        result=negative.copy();lookup=positive_frame.set_index('molecule_id').smiles.to_dict()
        assert set(lookup)==positive
        result.loc[result.molecule_id.isin(positive),'smiles']=result.loc[result.molecule_id.isin(positive),'molecule_id'].map(lookup)
        return result
    candidates={}
    for name in ['C08-parent','C09-control','C14-CE','C15-NP','C18-peak-drop']:
        candidates[name+'_negative']=read(BASE/(name+'-predictions.csv'))
    for name,run in [('C14-CE','c14'),('C15-NP','c15'),('C18-peak-drop','c18')]:
        candidates[name+'_allmode']=combine(candidates[name+'_negative'],read(ROOT/f'outputs/local/{run}-evaluation-20261110/development.csv'))
    negative=read(BASE/'negative-parent.csv');score_path=OUT/'C22-negative-scores.parquet'
    if score_path.exists():
        receipt=json.loads(Path(str(score_path)+'.json').read_text());scores=pd.read_parquet(score_path)
        assert receipt['manifest_sha256']==sha(PACKAGE/'c08-manifest.json')
        assert receipt['baseline_content_sha256']==__import__('hashlib').sha256(negative.to_csv(index=False,lineterminator='\n').encode()).hexdigest()
        assert receipt['queries_sha256']==sha(BASE/'negative-queries.parquet')
    else:
        index=pd.read_parquet(BASE/'fingerprint-smiles.parquet');fp=np.load(BASE/'fingerprints.npy',mmap_mode='r')
        shared={s:np.asarray(fp[i]) for i,s in enumerate(index.smiles)}
        scores,receipt=module.score(PACKAGE,'peak-drop-dose',negative,BASE/'negative-queries.parquet',score_path,shared_fps=shared)
    assert receipt['scored_molecules']==80 and receipt['input_errors']==receipt['unprocessed_molecules']==0
    assert not receipt['unsupported_molecules']
    neg_result,_=apply(negative,scores,.2)
    full_negative=parent.copy();lookup=neg_result.set_index('molecule_id').smiles.to_dict()
    mask=~full_negative.molecule_id.isin(positive)
    full_negative.loc[mask,'smiles']=full_negative.loc[mask,'molecule_id'].map(lookup)
    candidates['C22-peak-dose_allmode']=combine(full_negative,read(ROOT/'outputs/local/c22-evaluation-20261110/development.csv'))
    candidates['C24_negative']=read(ROOT/'outputs/local/c24-evaluation-20261110/development.csv')
    candidates['C28_allmode']=read(ROOT/'outputs/local/c28-evaluation-20261110/C28-predictions.csv')
    candidates['C09_allmode']=read(ROOT/'outputs/local/c28-evaluation-20261110/matched_C09-predictions.csv')
    candidates['C29_public']=read(ROOT/'outputs/forks/c29-public-replay-v1/c29_all_reference.csv')
    for name,frame in candidates.items():
        assert frame.molecule_id.tolist()==parent.molecule_id.tolist(),name
        frame.to_csv(OUT/(name+'.csv'),index=False)
    write(OUT/'prediction-freeze.json',dict(frozen_utc=stamp(),prediction_sha256={n:sha(OUT/(n+'.csv')) for n in candidates},
        C22_manifest_sha256=sha(PACKAGE/'c08-manifest.json'),C22_negative_receipt=receipt,reserved_labels_read=False))
    labels=pd.concat([pd.read_csv(ROOT/n,dtype=str) for n in ['outputs/local/c10-casmi2022-20261110/labels.csv','outputs/local/c20-negative-development-20261110/labels.csv']],ignore_index=True).set_index('molecule_id').loc[parent.molecule_id]
    assert len(labels)==labels.key.nunique()==240
    index=pd.read_parquet(BASE/'fingerprint-smiles.parquet');identities=dict(zip(index.smiles,index.metric_key))
    from precision_cache import features
    for f in [parent]+list(candidates.values()):
        for s in set(';'.join(f.smiles).split(';')):
            if s not in identities:identities[s]=features.structure(s)[0]
    baseline=metric_vector(parent,labels,identities);mask=parent.molecule_id.isin(positive).to_numpy();results={}
    for name,frame in candidates.items():
        v=metric_vector(frame,labels,identities)
        results[name]=dict(full240=paired_summary(baseline,v),positive160=paired_summary(baseline[mask],v[mask]),negative80=paired_summary(baseline[~mask],v[~mask]),truths_lost=int(((baseline>0)&(v==0)).sum()),truths_present=int((v>0).sum()))
    order=sorted(results,key=lambda n:(-results[n]['full240']['candidate_mrr25'],n));chosen=order[:2]
    write(OUT/'selection.json',dict(completed_utc=stamp(),selected=chosen,ranking=order,results=results,protocol_sha256=sha(plan),
        prediction_freeze_sha256=sha(OUT/'prediction-freeze.json'),reserved_labels_read=False,statistical_promotion=False,
        reason='Exactly two explicit user scoring trials; ranking on already used development only. Confidence intervals and failed previous gates remain disclosed. Real smoke and independent audits required before sending.'))
    print('SELECTED TWO PAST TRIALS',json.dumps(dict(selected=chosen,results={n:results[n] for n in chosen})),flush=True)

if __name__=='__main__':
    from pathlib import Path
    main()
