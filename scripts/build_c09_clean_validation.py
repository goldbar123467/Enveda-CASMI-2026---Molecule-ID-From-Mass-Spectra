"""C09 acquisition-disjoint confirmation: exact P01 and failed-P14 replay.

Build from the byte-pinned original P01/C08 builders. Require exact baseline
replay before changing its neural channel. No candidate truth injection, model
selection, or competition submission. The saved80-query panel is confirmation.
"""
import ast,hashlib,json
from pathlib import Path
from build_general_anchor import ROOT,guarded_source


def main():
    contract=ROOT/'outputs/publish/c09-validation-20261109/validation-contract.json'
    spec=json.loads(contract.read_text());assert spec['queries']==80 and spec['development']==0
    original=(ROOT/'scripts/build_compatible_validation.py').read_text(encoding='utf-8')
    for before,after in {
        'c08-validation-20261108':'c09-validation-20261109',
        'c08-external-validation':'c09-clean-validation',
        'casmi26-c08-novel-external-validation':'casmi26-c09-clean-validation',
        'casmi26 C08 novel external validation':'casmi26 c09 clean validation',
        "==128":"==80",'queries=128':'queries=80',
        'C08 full candidate':'C09 full candidate'
    }.items():original=original.replace(before,after)
    # The historical builder's ROOT is still the same trusted workspace.
    exec(compile(original,'c09_clean_validation_builder','exec'),dict(__name__='__main__',__file__=str(ROOT/'scripts/build_compatible_validation.py')))
    folder=ROOT/'kaggle/forks/c09-clean-validation';path=folder/'notebook.ipynb';nb=json.loads(path.read_text(encoding='utf-8'))
    p01=json.loads((ROOT/'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    replay_sources=[''.join(p01['cells'][i]['source']) for i in [15,17,18]]
    # These are the same physical/provenance/prior/dedup cells used in P14 v2.
    p14=json.loads((ROOT/'kaggle/forks/p14-frozen-channel/notebook.ipynb').read_text(encoding='utf-8'))
    pin=hashlib.sha256((ROOT/'outputs/publish/frozen-p14-20261107/frozen-manifest.json').read_bytes()).hexdigest()
    driver=(ROOT/'scripts/score_frozen_channel.py').read_text(encoding='utf-8')
    ending="""# Exact replay must reproduce P01 before accepting this validation route.
_clean_baseline=pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
shutil.copy2(ROOT/'submission.csv',ROOT/'c09_clean_p01.csv')
shutil.copy2(ROOT/'neural_candidate_features.parquet',ROOT/'c09_clean_neural_p01.parquet')
_replay_sources=REPLAY_SOURCES
def _clean_replay():
    global CONSENSUS_OK,REPORT,MOVES,SUBMITTED
    CONSENSUS_OK,REPORT,MOVES=build_consensus();assert CONSENSUS_OK
    SUBMITTED=choose_and_submit();assert SUBMITTED==VARIANT
    shutil.copy2(ROOT/'submission.csv',ROOT/'submission_control_V2.csv')
    for _i,_source in enumerate(_replay_sources):exec(compile(_source,'c09_downstream_'+str(_i),'exec'),globals())
    assert MERGE_REPORT.get('status')=='applied'
    return pd.read_csv(ROOT/'submission.csv',dtype=str,keep_default_na=False)
_reproduced=_clean_replay()
assert _reproduced.equals(_clean_baseline),'Unchanged complete P01 replay differs; validation cannot qualify'
_hits=list(INPUT.rglob('frozen-manifest.json'));assert len(_hits)==1
assert hashlib.sha256(_hits[0].read_bytes()).hexdigest()==P14_PIN
_frozen_bundle=_hits[0].parent
(CODE/'score_frozen_channel.py').write_text(P14_DRIVER,encoding='utf-8')
run_stage([sys.executable,CODE/'score_frozen_channel.py',_frozen_bundle,PUB_BUNDLE,ROOT/'c09_clean_neural_p01.parquet',ROOT/'effective_query_spectra.parquet',ROOT/'neural_candidate_features.parquet','cuda'],ROOT/'c09_clean_p14.log','C09 clean exact P14 negative control')
_frozen_receipt=json.loads((ROOT/'neural_candidate_features.parquet.frozen.json').read_text())
assert _frozen_receipt['input_errors']==0
_p14_final=_clean_replay();_p14_final.to_csv(ROOT/'c09_clean_p14.csv',index=False)
_clean_labels=pd.read_csv(C08_VALIDATION_SOURCE/'labels.csv',dtype=str).set_index('molecule_id')
_taut=_S.TautomerEnumerator()
def _clean_metrics(frame):
    result=[]
    for row in frame.itertuples(index=False):
        keys=[_identity(s) for s in row.smiles.split(';')];assert len(keys)==len(set(keys))
        truth=_clean_labels.loc[row.molecule_id,'key'];rank=keys.index(truth)+1 if truth in keys else None
        result.append(dict(molecule_id=row.molecule_id,rank=rank,mrr25=1/rank if rank else 0.,truth_present=rank is not None))
    return pd.DataFrame(result).set_index('molecule_id')
_before=_clean_metrics(_clean_baseline);_after=_clean_metrics(_p14_final).reindex(_before.index)
_diff=(_after.mrr25-_before.mrr25).to_numpy();_rng=np.random.default_rng(202611091)
_boot=np.array([_rng.choice(_diff,len(_diff),replace=True).mean() for _ in range(10000)])
_report=dict(kind='acquisition_disjoint_full_pipeline_confirmation',queries=len(_before),baseline_replay_exact=True,baseline_mrr25=float(_before.mrr25.mean()),p14_mrr25=float(_after.mrr25.mean()),p14_delta=float(_diff.mean()),p14_ci95=np.quantile(_boot,[.025,.975]).tolist(),p14_improved=int((_diff>0).sum()),p14_harmed=int((_diff<0).sum()),baseline_truths=int(_before.truth_present.sum()),p14_truths=int(_after.truth_present.sum()),detects_known_P14_regression=bool(_diff.mean()<0),truth_absent_mrr=0,truth_injection=False,model_selection=False,validation_contract_sha256=CONTRACT_PIN,known_public_baseline=.480,known_public_P14=.458,limitation='Acquisition-disjoint from cached competition inputs and previousC08 keys;80 external-source queries, broader historical/foundation exposure not universally excluded; no hidden-score guarantee')
_before.add_prefix('p01_').join(_after.add_prefix('p14_')).reset_index().to_csv(ROOT/'c09_clean_predictions.csv',index=False)
(ROOT/'c09_clean_report.json').write_text(json.dumps(_report,indent=2))
shutil.copy2(ROOT/'effective_query_spectra.parquet',ROOT/'c09_clean_queries.parquet')
(ROOT/'submission.csv').unlink()
print('DONE C09 CLEAN CONFIRMATION',json.dumps(_report),flush=True)
""".replace('REPLAY_SOURCES',repr(replay_sources)).replace('P14_PIN',repr(pin)).replace('P14_DRIVER',repr(driver)).replace('CONTRACT_PIN',repr(hashlib.sha256(contract.read_bytes()).hexdigest()))
    compile(ending,'c09_clean_ending','exec')
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=guarded_source(ending,'c09_clean_replay').splitlines(keepends=True),id='c09-clean-replay'))
    preamble="import torch,json\nfrom pathlib import Path\n_devices=[dict(index=i,name=torch.cuda.get_device_name(i),vram_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())]\nassert _devices\nPath('/kaggle/working/c09_clean_devices.json').write_text(json.dumps(_devices,indent=2))\nprint('C09 CLEAN ACTUAL DEVICES',json.dumps(_devices),flush=True)\n"
    nb['cells'].insert(0,dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=preamble.splitlines(keepends=True),id='c09-clean-devices'))
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'c09_clean_cell','exec')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((folder/'kernel-metadata.json').read_text());meta['dataset_sources'].append('clarkkitchen/casmi26-frozen-p14-20261107');meta['machine_shape']='NvidiaTeslaP100'
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    record=dict(campaign='C09',scoring_submission=False,confirmation_only=True,queries=80,validation_contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),p14_manifest_sha256=pin,
        baseline_replay_required=True,changed_cells='Input paths/preflight only, then exact P14 neural channel and full downstream replay')
    (folder/'clean-pin.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print('BUILT C09 CLEAN',json.dumps(record))


if __name__=='__main__':main()
