"""Package L06 control folds and a required-smoke P12 scoring notebook.

User explicitly requested one leaderboard trial despite failed external-data
confirmation. Preserve P01's fusion/physical/dedup recipe and fail on errors.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

from build_general_anchor import ROOT, guarded_source, replace_function

DATASET = 'clarkkitchen/casmi26-long-control-20261106'
SLOT = 'p12-long-control'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(run, destination):
    frozen = json.loads((run / 'manifest.json').read_text())
    assert not frozen['smoke'] and frozen['spec']['training']['epochs'] == 128
    for relative, digest in frozen['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['model.py', 'features.py']:
        shutil.copy2(ROOT / 'outputs/local/kit' / name, destination / name)
    cfg = json.loads((ROOT / 'outputs/local/kit/config.json').read_text())
    cfg.update(frozen['spec']['training'])
    (destination / 'config.json').write_text(json.dumps(cfg, indent=2))
    receipts = []
    for fold in range(5):
        folder = run / 'models/control' / f'fold-{fold}'
        receipt = json.loads((folder / 'complete.json').read_text())
        assert receipt['epochs'] == 128 and receipt['steps'] == 64000 and not receipt['smoke']
        assert sha(folder / 'model.pt') == receipt['model_sha256']
        assert sha(folder / 'preds/oof.npz') == receipt['predictions_sha256']
        shutil.copy2(folder / 'model.pt', destination / f'fold-{fold}.pt')
        receipts.append(receipt)
    manifest = dict(run_id='L06', family='control', folds=list(range(5)), epochs=128,
                    architecture={key: cfg[key] for key in ['width', 'heads', 'layers', 'embedding']},
                    files={name: sha(destination / name) for name in
                           ['model.py', 'features.py', 'config.json'] + [f'fold-{i}.pt' for i in range(5)]},
                    completion_receipts=receipts,
                    confirmation_retrieval_mrr=0.6806206157332367,
                    hidden_score_unmeasured=True, user_requested_leaderboard_trial=True,
                    limitation='Five 80%-identity fits, not full-data refits. Local retrieval does not predict Kaggle MRR.')
    (destination / 'long-control-manifest.json').write_text(json.dumps(manifest, indent=2))
    (destination / 'dataset-metadata.json').write_text(json.dumps(dict(
        title='CASMI26 L06 control fivefold models 20261106', id=DATASET,
        licenses=[dict(name='CC-BY-NC-4.0')]), indent=2))
    (destination / 'README.md').write_text(
        '# Private L06 control scoring models\n\n'
        'Five 128-epoch competition-data control fits. Mean candidate scores, not averaged weights.\n'
        'Architecture/model/features: Saner Turhaner CASMI spectrum-model lineage; retained project NOTICE applies.\n'
        'Shared-fold experiment structure inspired by Chris Deotte. No external spectra in these control fits.\n'
        'This is a user-requested leaderboard trial; hidden-score improvement is unproven.\n')
    shutil.copy2(ROOT / 'NOTICE.md', destination / 'NOTICE.md')
    return sha(destination / 'long-control-manifest.json')


def build(run, destination, output):
    pin = package(run, destination)
    notebook = json.loads((ROOT / 'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
    swap = '''def _swap_to_ours(base_bundle):
    import torch as _t
    _hits=list(Path('/kaggle/input').rglob('long-control-manifest.json'))
    assert len(_hits)==1, 'Exactly one pinned L06 control dataset is required'
    _manifest_path=_hits[0]
    assert hashlib.sha256(_manifest_path.read_bytes()).hexdigest()==MANIFEST_PIN
    _manifest=json.loads(_manifest_path.read_text())
    assert _manifest['family']=='control' and _manifest['folds']==list(range(5))
    _src=_manifest_path.parent
    _dst=ROOT/'ours_bundle';_dst.mkdir(exist_ok=True)
    for _name,_digest in _manifest['files'].items():
        _path=_src/_name
        assert hashlib.sha256(_path.read_bytes()).hexdigest()==_digest, _name
        shutil.copy2(_path,_dst/_name)
    shutil.copy2(_manifest_path,_dst/'long-control-manifest.json')
    for _fold in range(5):
        _ck=_t.load(_dst/f'fold-{_fold}.pt',map_location='cpu',weights_only=True)
        assert _ck['stage']=='crossfit_holdout' and _ck['epoch']==128 and _ck['fold']==_fold
        assert not _ck['smoke'] and _ck['contract']['family']=='control'
        assert all(_ck['cfg'][k]==v for k,v in _manifest['architecture'].items())
        del _ck
    _audit=dict(manifest_sha256=MANIFEST_PIN,models=5,family='control',epochs=128,
                architecture=_manifest['architecture'],checkpoint_sha256={k:v for k,v in _manifest['files'].items() if k.endswith('.pt')},
                neural_weights=dict(original=.5,long_control=.5),hidden_score_unmeasured=True)
    (ROOT/'long_control_model_audit.json').write_text(json.dumps(_audit,indent=2))
    print('LONG CONTROL PINNED',json.dumps(_audit),flush=True)
    return str(_dst)'''.replace('MANIFEST_PIN', repr(pin))
    replace_function(notebook, '_swap_to_ours', swap)
    scorer = '''def score_final_lists():
    _script='long_ensemble.py' if (Path(BUNDLE)/'long-control-manifest.json').exists() else 'neural_scores.py'
    run_stage([sys.executable,CODE/_script,BUNDLE,'final_model.pt',
               ROOT/'frozen_candidate_evidence.parquet',ROOT/'effective_query_spectra.parquet',
               ROOT/'neural_candidate_features.parquet','cuda'],
              ROOT/'submission_neural.log','Required final-list neural scorer')
    return True'''
    replace_function(notebook, 'score_final_lists', scorer)
    driver = (ROOT / 'scripts/score_long_ensemble.py').read_text(encoding='utf-8')
    found = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'MODULES[\'neural_scores.py\']' in source or 'MODULES["neural_scores.py"]' in source:
            source += '\nMODULES["long_ensemble.py"]=' + repr(driver) + '\n'
            source += 'MODULE_SHA256["long_ensemble.py"]=' + repr(hashlib.sha256(driver.encode()).hexdigest()) + '\n'
            found += 1
        # The outsider channel must use the same fivefold scoring rule.
        source = source.replace("CODE / 'neural_scores.py', BUNDLE", "CODE / ('long_ensemble.py' if (Path(BUNDLE)/'long-control-manifest.json').exists() else 'neural_scores.py'), BUNDLE")
        source = source.replace('REPRODUCTION AUDIT PASSED: public480 assets, ensemble, merge, dedup',
                                'P12 AUDIT PASSED: fivefold control, ensemble, merge, dedup')
        cell['source'] = source.splitlines(keepends=True)
    assert found == 1
    seal = '''_seal_a=pd.read_parquet(ROOT/'neural_public.parquet').set_index(['molecule_id','smiles'])
_seal_b=pd.read_parquet(ROOT/'neural_ours.parquet').set_index(['molecule_id','smiles']).reindex(_seal_a.index)
_seal_final=pd.read_parquet(ROOT/'neural_candidate_features.parquet').set_index(['molecule_id','smiles']).reindex(_seal_a.index)
assert not _seal_a.index.has_duplicates
_seal_eligible=_seal_a.neural_ok.astype(bool)
assert _seal_eligible.any() and _seal_b.loc[_seal_eligible,'neural_ok'].astype(bool).all()
_seal_expected=.5*_z(_seal_a.neural_score.fillna(0))+.5*_z(_seal_b.neural_score.fillna(0))
_seal_error=float((_seal_final.neural_score-_seal_expected).abs().max())
assert np.isfinite(_seal_final.loc[_seal_eligible,'neural_score']).all() and _seal_error<1e-6
_folds=[pd.read_parquet(ROOT/f'neural_candidate_features-fold-{i}.parquet').set_index(['molecule_id','smiles']).reindex(_seal_a.index) for i in range(5)]
assert all(p.loc[_seal_eligible,'neural_ok'].astype(bool).all() for p in _folds)
_fold_mean=np.mean([p.neural_score.to_numpy() for p in _folds],axis=0)
_fold_error=float(np.abs(_seal_b.neural_score.to_numpy()-_fold_mean).max())
assert _fold_error<1e-6 and MERGE_REPORT.get('status')=='applied'
_ensemble_receipt=json.loads((ROOT/'neural_candidate_features.parquet.ensemble.json').read_text())
assert _ensemble_receipt['models']==5 and _ensemble_receipt['input_errors']==0
_seal_receipt=dict(model=json.loads((ROOT/'long_control_model_audit.json').read_text()),
    neural_score_max_abs_error=_seal_error,fold_mean_max_abs_error=_fold_error,
    real_scored_molecules=int(_seal_a.loc[_seal_eligible].index.get_level_values(0).nunique()),
    pipeline=json.loads((ROOT/'execution_receipt.json').read_text()),merge_status=MERGE_REPORT['status'],
    is_hidden_rerun=IS_RERUN,ensemble=_ensemble_receipt)
(ROOT/'p12_qualification_receipt.json').write_text(json.dumps(_seal_receipt,indent=2))
print('P12 QUALIFIED',json.dumps(_seal_receipt),flush=True)
'''
    notebook['cells'].append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=seal.splitlines(keepends=True)))
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            compile(source, f'p12_source_{index}', 'exec')
            cell['source'] = guarded_source(source, f'p12_cell_{index}').splitlines(keepends=True)
            cell['execution_count'], cell['outputs'] = None, []
    notebook['cells'].insert(0, dict(cell_type='markdown', metadata={}, source=[
        '# P12: original scorer 50% + L06 control fivefold ensemble 50%\n',
        'Five pinned 128-epoch control fits; mean raw candidate scores. Retains P01 fusion, physical rules and metric dedup.\n',
        'Explicitly requested leaderboard trial despite unproven hidden-score benefit. External-data models are not used.\n',
        'Required real GPU smoke and independent numeric checks precede scoring submission. Retained upstream credits apply.\n']))
    notebook['nbformat_minor'] = 5
    for index, cell in enumerate(notebook['cells']):
        cell.setdefault('id', hashlib.sha256((str(index) + ''.join(cell['source'])).encode()).hexdigest()[:12])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'notebook.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    meta = json.loads((ROOT / 'kaggle/forks/p01-public480/kernel-metadata.json').read_text())
    meta.update(id='clarkkitchen/casmi26-' + SLOT, title='casmi26 p12 long control', kernel_sources=[])
    meta['dataset_sources'].append(DATASET)
    (output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    (output / 'model-pin.json').write_text(json.dumps(dict(dataset=DATASET, manifest_sha256=pin, models=5, family='control', epochs=128), indent=2))
    print('PACKAGED P12', pin, 'private dataset', destination, 'notebook', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/local/long-20261106')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'outputs/publish/long-control-20261106')
    parser.add_argument('--output', type=Path, default=ROOT / 'kaggle/forks' / SLOT)
    args = parser.parse_args()
    build(args.run, args.dataset, args.output)
