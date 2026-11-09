"""C09 cloud GPU1: audit actual generated pools, then score them with C08.

The already-used C08 panel is diagnostic only. Labels are consulted only after
candidate pools and scores are complete. Never insert truth into a pool.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import pandas as pd
import numpy as np
import json,sys
from pathlib import Path
from compatible_inference import score,load_modules,digest


def run(input_root,out):
    torch.set_num_threads(2);input_root=Path(input_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def one(name):
        hits=list(input_root.rglob(name));assert len(hits)==1,(name,[str(p) for p in hits]);return hits[0]
    queries=one('c08_validation_queries.parquet')
    root=queries.parent; baseline=pd.read_csv(root/'c08_validation_p01.csv',dtype=str,keep_default_na=False)
    payload=json.loads((root/'e004_candidate_inputs.json').read_text())
    bundle=one('c08-manifest.json').parent
    _,features=load_modules(bundle);fps={};identities={}
    def identity(smiles):
        if smiles not in identities:
            try:
                key,*_,fp=features.structure(smiles,2048);identities[smiles]=key;fps[smiles]=fp
            except (ValueError,TypeError):identities[smiles]=None
        return identities[smiles]
    sources={};expanded=[]
    for row in baseline.itertuples(index=False):
        mid=str(row.molecule_id)
        channels=dict(final=row.smiles.split(';'),prefusion=payload.get('prefusion',{}).get(mid,[]),
            fusion_overflow=payload.get('fusion_overflow',{}).get(mid,[]),main_tail=payload.get('base_tail',{}).get(mid,[]),
            engine2=(payload.get('engine2',{}).get(mid) or {}).get('smiles',[]),
            pubchem=(payload.get('pubchem',{}).get(mid) or {}).get('pc',[]))
        sources[mid]={};values=[];seen=set()
        for channel,candidates in channels.items():
            # Explicit fixed resource cap, independent of labels or scores.
            keys=set()
            for s in list(candidates or [])[:256]:
                if not isinstance(s,str) or not s.strip():continue
                key=identity(s)
                if key is None:continue
                keys.add(key)
                if key not in seen:seen.add(key);values.append(s)
            sources[mid][channel]=keys
        assert values
        expanded.append(dict(molecule_id=mid,smiles=';'.join(values)))
    expanded=pd.DataFrame(expanded);expanded.to_csv(out/'expanded_candidates.csv',index=False)
    values,receipt=score(bundle,'legacy16-hard',expanded,queries,out/'expanded_scores.parquet',shared_fps=fps)
    assert receipt['input_errors']==0
    # Only now read the labelled sidecar. None of the above uses labels.
    labels=pd.read_csv(one('labels.csv'),dtype=str).set_index('molecule_id')
    rows=[]
    for mid,group in values.groupby('molecule_id',sort=False):
        truth=labels.loc[mid,'key'];order=group.sort_values('c08_score',ascending=False,kind='stable')
        keys=[identity(s) for s in order.smiles];rank=keys.index(truth)+1 if truth in keys else None
        original=[identity(s) for s in baseline.set_index('molecule_id').loc[mid,'smiles'].split(';')]
        before=original.index(truth)+1 if truth in original else None
        rows.append(dict(molecule_id=mid,partition=labels.loc[mid,'partition'],candidate_count=len(group),
            **{'present_'+name:truth in keys0 for name,keys0 in sources[mid].items()},
            union_present=rank is not None,c08_union_rank=rank,c08_union_mrr25=1/rank if rank and rank<=25 else 0.,
            p01_rank=before,p01_mrr25=1/before if before else 0.))
    frame=pd.DataFrame(rows);frame.to_csv(out/'coverage.csv',index=False)
    summary=dict(kind='previously_used_panel_diagnostic',queries=len(frame),truth_injection=False,truth_absent_mrr=0,
        candidate_generation_cap_per_channel=256,source_coverage={c:int(frame[c].sum()) for c in frame if c.startswith('present_')},
        union_truths=int(frame.union_present.sum()),final_truths=int(frame.present_final.sum()),
        union_only_truths=int((frame.union_present&~frame.present_final).sum()),
        baseline_mrr25=float(frame.p01_mrr25.mean()),c08_union_mrr25=float(frame.c08_union_mrr25.mean()),
        candidate_rows=len(values),c08_manifest_sha256=digest(bundle/'c08-manifest.json'),
        limitation='Expanded-pool ranking is a diagnostic, not the final physical/prior merge or a fresh holdout; no promotion claim.')
    (out/'coverage_report.json').write_text(json.dumps(summary,indent=2));print('DONE C09 COVERAGE',json.dumps(summary),flush=True)


if __name__=='__main__':run(*sys.argv[1:])
