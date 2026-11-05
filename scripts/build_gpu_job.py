"""Build a Kaggle T4 x2 notebook that runs scripts/local_seeds.py jobs, one job list per GPU, in parallel.

    python scripts/build_gpu_job.py <slot> "<gpu0 job>; <gpu0 job>" "<gpu1 job>; ..."
    e.g. python scripts/build_gpu_job.py n5-e128-curves "holdout 8 --epochs 128" "holdout 8 --split np --epochs 128"

A session is billed once for both GPUs. Inputs: the kit dataset and n0's cache (kernel source); the NP holdout keys
(outputs/local/seeds/np_holdout_keys.txt) are embedded. Everything lands in /kaggle/working/seeds_out/ (refits as
refit/casmi26_seed_<k>[_e<E>].pt, which SEED_ENS's rglob picks up when the kernel is attached; holdout models as
casmi26_holdout_*, which it ignores). Push with --accelerator NvidiaTeslaT4.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORKS = ROOT / "kaggle/forks"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"

BODY = r'''# ---- {slot}: scripts/local_seeds.py jobs on Kaggle T4 x2, one job list per GPU, run in parallel ----
# Saner Turhaner's scorer kit (sanpier/casmi-spectrum-model-v1, unchanged train.py); cache from casmi26-n0-scorer-prep.
import json, os, subprocess, sys, time
from pathlib import Path
W = Path('/kaggle/working'); INP = Path('/kaggle/input'); T0 = time.time()
def log(*a): print(f'[{{time.time() - T0:7.0f}}s]', *a, flush=True)
JOBS = {jobs}
KIT = next(p.parent for p in INP.rglob('train.py') if (p.parent / 'final_model.pt').is_file() and (p.parent / 'model.py').is_file())
CACHE = next(p.parent for p in INP.rglob('peaks.bin') if (p.parent / 'fingerprints.npy').is_file())
OUT = W / 'seeds_out'; OUT.mkdir(exist_ok=True)
SRC = {src}
NP_KEYS = {keys}
(W / 'local_seeds.py').write_text(SRC); (OUT / 'np_holdout_keys.txt').write_text(NP_KEYS)
import torch
log('GPUS', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'CPUS', os.cpu_count())
procs = {{}}
for g, jobs in JOBS.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), CASMI_KIT=str(KIT), CASMI_CACHE=str(CACHE), CASMI_OUT=str(OUT))
    chain = ' ; '.join(f'{{sys.executable}} -u {{W / "local_seeds.py"}} {{j}}' for j in jobs)
    procs[g] = subprocess.Popen(['bash', '-c', chain], env=env, stdout=open(W / f'gpu{{g}}.log', 'w'), stderr=subprocess.STDOUT)
    log('GPU', g, chain)
while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    for g in procs:
        lines = [l for l in (W / f'gpu{{g}}.log').read_text().splitlines() if not l.startswith(('HOLDOUT {{', 'REFIT {{'))]
        log('GPU', g, 'rc', procs[g].poll(), '|', lines[-1][:200] if lines else '')
for g in procs:
    print(f'===== gpu{{g}}.log (rc {{procs[g].returncode}})'); print((W / f'gpu{{g}}.log').read_text()[-6000:])
for f in sorted(OUT.rglob('*_report.json')):
    for k, v in json.loads(f.read_text()).items():
        print('REPORT', f.name, k, v.get('epochs'), v.get('seconds'), [round(h['diagnostic_mrr25'], 3) for h in v.get('history', []) if 'diagnostic_mrr25' in h])
log('JOB_DONE', sorted(str(p.relative_to(OUT)) for p in OUT.rglob('*.pt')))
'''


def build(slot, gpu_jobs):
    jobs = {g: [j.strip() for j in spec.split(";") if j.strip()] for g, spec in enumerate(gpu_jobs)}
    src = (ROOT / "scripts/local_seeds.py").read_text(encoding="utf-8")
    keys = (ROOT / "outputs/local/seeds/np_holdout_keys.txt").read_text(encoding="utf-8")
    code = BODY.format(slot=slot, jobs=json.dumps(jobs), src=json.dumps(src), keys=json.dumps(keys))
    compile(code, slot, "exec")
    d = FORKS / slot
    d.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"clarkkitchen/casmi26-{slot}", "title": f"casmi26 {slot.replace('-', ' ')}", "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["sanpier/casmi-spectrum-model-v1"],
            "kernel_sources": ["clarkkitchen/casmi26-n0-scorer-prep"], "competition_sources": [], "model_sources": [],
            "docker_image": DOCKER}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", d, jobs)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2:])
