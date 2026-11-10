"""Deploy the frozen negative-only tail without changing protected parent rows."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from negative_tail import merge


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def query_modes(parent,queries):
    """A mixed or unknown polarity has an explicit exact-parent fallback."""
    assert parent.columns.tolist()==['molecule_id','smiles'] and parent.molecule_id.is_unique
    assert parent.molecule_id.map(type).eq(str).all() and parent.smiles.map(type).eq(str).all()
    assert {'molecule_id','ionization_mode'}<=set(queries)
    q=queries.copy();q.molecule_id=q.molecule_id.astype(str)
    groups={str(mid):frame for mid,frame in q.groupby('molecule_id',sort=False)}
    modes={};rows=[]
    for mid in parent.molecule_id:
        frame=groups.get(mid)
        if frame is None:mode='absent';n=0
        else:
            values={str(v).strip().lower() for v in frame.ionization_mode};n=len(frame)
            if not values<={'positive','negative'}:mode='unknown'
            elif len(values)==1:mode=next(iter(values))
            else:mode='mixed'
        modes[mid]=mode;rows.append(dict(molecule_id=mid,mode=mode,spectrum_rows=n,eligible_negative=mode=='negative'))
    return modes,pd.DataFrame(rows)


def run(parent,queries,bundle,arm,scores_path,device='cuda',shared_fps=None):
    """Same score and negative_tail.merge functions as the frozen confirmation."""
    bundle=Path(bundle);scores_path=Path(scores_path)
    assert not scores_path.exists(),'Inspect an existing score artifact before any recovery'
    torch.set_num_threads(2)
    if device=='cuda':
        assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.7)
    manifest_path=bundle/'c08-manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['kind']=='compatible_scorer' and not manifest['smoke']
    models=[m for m in manifest['models'] if m['arm']==arm];assert len(models)==2 and [m['fold'] for m in models]==[0,1]
    for name,pin in manifest['files'].items():assert sha(bundle/name)==pin,name
    spec=importlib.util.spec_from_file_location('_negative_pinned_inference',bundle/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    modes,mode_audit=query_modes(parent,queries)
    eligible={mid for mid,mode in modes.items() if mode=='negative'}
    known={mid:mode for mid,mode in modes.items() if mode in ['positive','negative']}
    known_parent=parent[parent.molecule_id.isin(known)].copy()
    negative_parent=parent[parent.molecule_id.isin(eligible)].copy()
    q=queries[queries.molecule_id.astype(str).isin(eligible)].copy()
    if eligible:
        scores,receipt=module.score(bundle,arm,negative_parent,q,scores_path,device=device,shared_fps=shared_fps)
        assert receipt['manifest_sha256']==sha(manifest_path) and receipt['input_errors']==receipt['unprocessed_molecules']==0
        assert set(scores.molecule_id)==eligible and scores.groupby('molecule_id').c08_ok.nunique().eq(1).all()
        supported=set(scores[scores.c08_ok.astype(bool)].molecule_id)
        assert len(supported)==receipt['scored_molecules'] and set(receipt['unsupported_molecules'])==eligible-supported
        aggregation_error=0.
        for _,frame in scores[scores.c08_ok.astype(bool)].groupby('molecule_id',sort=False):
            expected=np.zeros(len(frame),dtype=np.float64)
            for fold in [0,1]:
                raw=frame['c08_raw_fold'+str(fold)].to_numpy(np.float64)
                expected+=.5*(raw-raw.mean())/max(float(raw.std()),1e-6)
            aggregation_error=max(aggregation_error,float(np.max(np.abs(frame.c08_score.to_numpy()-expected))))
        assert aggregation_error<1e-8
    else:
        scores=pd.DataFrame(columns=['molecule_id','smiles','baseline_rank','c08_ok','c08_score'])
        scores.to_parquet(scores_path,index=False)
        receipt=dict(scored_molecules=0,unsupported_molecules=[],input_errors=0,unprocessed_molecules=0,
            manifest_sha256=sha(manifest_path),models_loaded=0)
        supported=set();aggregation_error=0.
    result=parent.copy()
    if len(known_parent):
        changed,actions=merge(known_parent,scores,known,.2)
        lookup=changed.set_index('molecule_id').smiles.to_dict()
        for index,row in result.iterrows():
            if row.molecule_id in lookup:result.at[index,'smiles']=lookup[row.molecule_id]
        assert actions['rank_one_changes']==actions['candidate_set_changes']==0
        assert actions['scored_molecules']==len(supported) and actions['passthrough_molecules']==len(eligible-supported)
    else:actions=dict(scored_molecules=0,passthrough_molecules=0,rank_one_changes=0,candidate_set_changes=0,changed_rows=0)
    protected=~parent.molecule_id.isin(supported)
    assert result.loc[protected].equals(parent.loc[protected])
    for before,after in zip(parent.smiles,result.smiles):
        a,b=before.split(';'),after.split(';');assert b[0]==a[0] and len(a)==len(b) and set(a)==set(b)
    report=dict(kind='P16_plus_frozen_negative_tail',weight=.2,offset=3.,valid_rows=len(result),
        source_query_ids=int(queries.molecule_id.astype(str).nunique()),eligible_negative_queries=len(eligible),
        mode_counts=mode_audit['mode'].value_counts().to_dict(),neural_scored_molecules=len(supported),
        unsupported_negative_molecules=sorted(eligible-supported),protected_parent_rows=int(protected.sum()),
        positive_parent_rows_exact=int(mode_audit['mode'].eq('positive').sum()),
        mixed_unknown_absent_parent_rows_exact=int(mode_audit['mode'].isin(['mixed','unknown','absent']).sum()),
        changed_rows=int(result.smiles.ne(parent.smiles).sum()),rank_one_changes=0,candidate_set_changes=0,
        mean_of_fold_standardizations_max_abs_error=aggregation_error,arm=arm,model_manifest_sha256=sha(manifest_path),
        scores_sha256=sha(scores_path),inference=receipt,source_sha256=sha(__file__),
        device=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',quality_gate_claimed=False,automatic_submission=False)
    return result,report,mode_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','baseline','queries','scores','output','contract']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=parser.parse_args()
    assert args.baseline.resolve()!=args.output.resolve()
    contract=json.loads(args.contract.read_text())
    for name,pin in contract['source_files'].items():assert sha(args.contract.parent/name)==pin,name
    assert sha(args.bundle/'c08-manifest.json')==contract['selected_manifest_sha256'] and contract['weight']==.2 and contract['offset']==3.
    parent=pd.read_csv(args.baseline,dtype=str,keep_default_na=False);queries=pd.read_parquet(args.queries)
    # The parent has its own protected filename. A failed required stage cannot
    # leave a champion file under the scoring filename and masquerade as success.
    args.output.unlink(missing_ok=True)
    final,report,audit=run(parent,queries,args.bundle,contract['arm'],args.scores,args.device)
    final.to_csv(args.output,index=False)
    report.update(parent_sha256=sha(args.baseline),queries_sha256=sha(args.queries),csv_sha256=sha(args.output),contract_sha256=sha(args.contract))
    audit.to_csv(args.output.with_suffix('.mode-audit.csv'),index=False)
    args.output.with_suffix('.negative-receipt.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('DONE FROZEN NEGATIVE DEPLOYMENT',json.dumps(report),flush=True)


if __name__=='__main__':main()
