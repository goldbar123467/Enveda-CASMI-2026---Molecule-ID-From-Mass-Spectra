"""Prepare a private CPU cross-host check of the identical USED head inputs."""
import base64
import hashlib
import json
import shutil
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp, unwrap, value


def main():
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    out = base / 'cpu-head-parity'; assert not out.exists()
    binding = base / 'complete-pipeline-binding.json'
    for name, pin in read(binding)['files_sha256'].items(): assert sha(ROOT / name) == pin, name
    used = read(base / 'complete-pair-used-probe/results.json'); assert used['passed']
    work = Path(used['working'])
    package = ROOT / 'outputs/publish/c76-used-head-inputs-20261112'; assert not package.exists()
    package.mkdir()
    names = ['c09e_p01_submission.csv', 'c76_p16.csv', 'c09e_external_receipt.json',
        'c09e_matches.parquet', 'effective_query_spectra.parquet', 'execution_receipt.json',
        'neural_candidate_features.parquet', 'neural_public.parquet', 'neural_ours.parquet']
    for name in names: shutil.copyfile(work / name, package / name)
    reference = ROOT / 'outputs/local/c74-complete-clean-target-20261112/target-masked-reference'
    for name in ['ext_meta.parquet', 'ext_mz.npy', 'ext_in.npy']:
        shutil.copyfile(reference / name, package / name); names.append(name)
    write(package / 'used-input-contract.json', dict(frozen_utc=stamp(), queries=12, acquisitions=51,
        files_sha256={name: sha(package / name) for name in names},
        local_used_intent_sha256=sha(base / 'complete-pair-used-probe/intent.json'),
        no_query_truth_or_expected_head_predictions_mounted=True, USED_only=True,
        quality_labels_read=False, C13_read=False, scoring_entries=0))
    write(package / 'dataset-metadata.json', dict(title='CASMI26 C76 USED head deployment inputs',
        id='clarkkitchen/casmi26-c76-used-head-inputs-20261112', licenses=[{'name': 'CC0-1.0'}]))
    folder = ROOT / 'kaggle/forks/c76-cpu-head-parity'; assert not folder.exists(); folder.mkdir()
    notebook = read(ROOT / 'kaggle/forks/c76-full-masked-validation/notebook.ipynb')
    inner = json.loads(base64.b64decode(value(unwrap(''.join(notebook['cells'][12]['source'])), 'FROZEN_B64')))
    bootstrap = ''.join(inner['cells'][2]['source'])
    pair_pin = sha(ROOT / 'outputs/publish/c76-complete-direct-pair-20261112/pair-manifest.json')
    source = '''import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='2'
os.environ['PYTHONHASHSEED']='0'
import torch
torch.set_num_threads(2)
import hashlib,json,shutil,subprocess,sys,platform
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
root=Path('/kaggle/working');inputs=Path('/kaggle/input')
bootstrap=BOOTSTRAP
(root/'official-bootstrap.py').write_text(bootstrap,encoding='utf8')
subprocess.run([sys.executable,str(root/'official-bootstrap.py')],check=True)
hits=[p for p in inputs.rglob('pair-manifest.json') if sha(p)==PAIR_PIN]
assert len(hits)==1;bundle=hits[0].parent
hits=[p for p in inputs.rglob('used-input-contract.json') if sha(p)==INPUT_PIN]
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
result=dict(source_sha256=sha(__file__),complete_pipeline_binding_sha256=BINDING_PIN,
    pair_manifest_sha256=PAIR_PIN,used_input_contract_sha256=INPUT_PIN,
    queries=12,acquisitions=51,actual_GPU_disabled=True,GPU_initialized=False,
    python=platform.python_version(),numpy=__import__('numpy').__version__,torch=torch.__version__,
    official_rdkit=__import__('rdkit').__version__,quality_labels_read=False,C13_read=False,
    no_expected_head_predictions_mounted=True,USED_only=True,scoring_entries=0,
    score_receipt_sha256=sha(root/'results/score-receipt.json'))
(root/'head-probe.json').write_text(json.dumps(result,indent=2))
print('C76 ACTUAL CPU COMPLETE HEAD PROBE FINISHED',json.dumps(result),flush=True)
'''.replace('BOOTSTRAP', repr(bootstrap)).replace('PAIR_PIN', repr(pair_pin)).replace(
        'INPUT_PIN', repr(sha(package / 'used-input-contract.json'))).replace('BINDING_PIN', repr(sha(binding)))
    compile(source, 'c76_cpu_head_probe', 'exec')
    (folder / 'head_probe.py').write_text(source, encoding='utf8', newline='\n')
    image = read(ROOT / 'kaggle/forks/c76-full-masked-validation/kernel-metadata.json')['docker_image']
    metadata = dict(id='clarkkitchen/casmi26-c76-cpu-head-parity', title='casmi26 C76 CPU complete head parity',
        code_file='head_probe.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=False, enable_tpu=False, enable_internet=False,
        dataset_sources=['clarkkitchen/casmi26-c76-complete-direct-pair-20261112',
            read(package / 'dataset-metadata.json')['id'], 'metric/rdkit-2026-3-3-wheel'],
        competition_sources=[], kernel_sources=[], docker_image=image, docker_image_pinning_type='original')
    write(folder / 'kernel-metadata.json', metadata)
    out.mkdir()
    write(out / 'freeze.json', dict(frozen_utc=stamp(), source_sha256=sha(__file__),
        probe_source_sha256=sha(folder / 'head_probe.py'), metadata_sha256=sha(folder / 'kernel-metadata.json'),
        complete_pipeline_binding_sha256=sha(binding), pair_manifest_sha256=pair_pin,
        used_input_contract_sha256=sha(package / 'used-input-contract.json'),
        local_used_score_receipt_sha256=sha(base / 'complete-pair-used-probe/score-receipt.json'),
        original_official_bootstrap_exact=True, source_label_or_quality_read=False,
        comparisons=dict(features='bitexact', final_lists='exact', scores=dict(atol=1e-10, rtol=1e-10)),
        actual_GPU_disabled=True, scoring_entries=0))
    print('C76 CPU CROSS-HOST HEAD PROBE PREPARED; NO REMOTE RELEASE YET', flush=True)


if __name__ == '__main__': main()
