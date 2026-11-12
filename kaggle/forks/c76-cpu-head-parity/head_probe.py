import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='2'
os.environ['PYTHONHASHSEED']='0'
import torch
torch.set_num_threads(2)
import hashlib,json,shutil,subprocess,sys,platform
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
root=Path('/kaggle/working');inputs=Path('/kaggle/input')
bootstrap="import os, sys, subprocess, importlib.metadata, time\nos.environ['PYTHONHASHSEED'] = '0'\nNOTEBOOK_STARTED = time.monotonic()   # record total notebook time, including the retained analysis\nfrom pathlib import Path\nif not Path('/kaggle/input').exists() and Path('.runtime/rdkit2026033').exists():\n    sys.path.insert(0, str(Path('.runtime/rdkit2026033').resolve()))\ntry:\n    installed = importlib.metadata.version('rdkit')\nexcept importlib.metadata.PackageNotFoundError:\n    installed = ''\nif installed.replace('.03.', '.3.') != '2026.3.3':\n    tag=f'cp{sys.version_info.major}{sys.version_info.minor}'\n    wheels=sorted(Path('/kaggle/input').rglob(f'rdkit-2026.3.3-{tag}-*.whl'))\n    assert wheels, f'Attach the RDKit 2026.3.3 offline wheel for {tag}.'\n    subprocess.check_call([sys.executable,'-m','pip','install','--no-index','--no-deps','-q',str(wheels[0])], env={**os.environ, 'PYTHONHASHSEED': '0'})\nfrom rdkit import rdBase\nassert rdBase.rdkitVersion == '2026.03.3', rdBase.rdkitVersion\nprint('Official-metric RDKit:', rdBase.rdkitVersion)\n"
(root/'official-bootstrap.py').write_text(bootstrap,encoding='utf8')
subprocess.run([sys.executable,str(root/'official-bootstrap.py')],check=True)
hits=[p for p in inputs.rglob('pair-manifest.json') if sha(p)=='2868660b0118a2c4746510852eb22bea0c378f78777484caa45b9cbae165ebd2']
assert len(hits)==1;bundle=hits[0].parent
hits=[p for p in inputs.rglob('used-input-contract.json') if sha(p)=='424501b903c04fece6c7ade3c0105cd7ba87510d0e4ad4a9a47b6b399fb0c7d0']
assert len(hits)==1;fixture=hits[0].parent;contract=json.loads(hits[0].read_text())
assert contract['USED_only'] and contract['queries']==12 and contract['acquisitions']==51
for name,pin in contract['files_sha256'].items():assert sha(fixture/name)==pin,name
work=root/'used-work';assert not work.exists();work.mkdir()
for name in contract['files_sha256']:
    if not name.startswith('ext_'):shutil.copyfile(fixture/name,work/name)
subprocess.run([sys.executable,str(bundle/'score_c76_complete_pair.py'),'--work',str(work),
    '--bundle',str(bundle),'--reference',str(fixture),'--output',str(root/'results')],check=True)
assert not torch.cuda.is_initialized()
receipt=json.loads((root/'results/score-receipt.json').read_text())
assert receipt['queries']==12 and receipt['acquisitions']==51 and set(receipt['trials'])=={'cpu','gpu'}
result=dict(source_sha256=sha(__file__),complete_pipeline_binding_sha256='04b47cac8fe5476613834d4214d53cb51b7cc7a81162bf9516d2119acf7a03b9',
    pair_manifest_sha256='2868660b0118a2c4746510852eb22bea0c378f78777484caa45b9cbae165ebd2',used_input_contract_sha256='424501b903c04fece6c7ade3c0105cd7ba87510d0e4ad4a9a47b6b399fb0c7d0',
    queries=12,acquisitions=51,actual_GPU_disabled=True,GPU_initialized=False,
    python=platform.python_version(),numpy=__import__('numpy').__version__,torch=torch.__version__,
    official_rdkit=__import__('rdkit').__version__,quality_labels_read=False,C13_read=False,
    no_expected_head_predictions_mounted=True,USED_only=True,scoring_entries=0,
    score_receipt_sha256=sha(root/'results/score-receipt.json'))
(root/'head-probe.json').write_text(json.dumps(result,indent=2))
print('C76 ACTUAL CPU COMPLETE HEAD PROBE FINISHED',json.dumps(result),flush=True)
