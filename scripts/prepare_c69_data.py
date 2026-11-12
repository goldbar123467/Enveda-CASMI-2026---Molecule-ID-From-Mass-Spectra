"""Bind unchanged101features to the independently reviewable C68 fit mask."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import shutil
from pathlib import Path
from c45_complete_parent import ROOT,read,write,sha,stamp


def main():
    src=ROOT/'outputs/local/c66-parent-evidence-oof-20261112';boundary=ROOT/'outputs/local/c68-reserved-fit-boundary-20261112'
    out=ROOT/'outputs/local/c69-clean-parent-evidence-oof-20261112';out.mkdir(exist_ok=True)
    assert not (out/'data-freeze.json').exists()
    b=read(boundary/'boundary-freeze.json')
    assert b['fitting_reserved_identity_or_chemistry_overlap']==0 and b['global_reserved_identities']==231
    for name,pin in b['files_sha256'].items():assert sha(ROOT/name)==pin
    for name in ['queries.parquet','candidates.parquet']:shutil.copyfile(boundary/name,out/name)
    for name in ['features.npy','feature-names.json','physical-parent-changes.json','parent-score-reconstruction.json']:shutil.copyfile(src/name,out/name)
    policy=ROOT/'configs/c69-clean-parent-evidence-oof-20261112.json'
    files=[Path(__file__),boundary/'boundary-freeze.json',src/'data-freeze.json',policy]+list(out.iterdir())
    j=read(src/'data-freeze.json');j.update(prepared_utc=stamp(),source_sha256=sha(__file__),policy_sha256=sha(policy),
        boundary_sha256=sha(boundary/'boundary-freeze.json'),fitting_queries=b['fitting_queries'],fitting_candidate_rows=b['fitting_candidate_rows'],
        global_reserved_chemistry_fit_overlap=0,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    write(out/'data-freeze.json',j);print('C69 DATA FROZEN; all869evaluated, strict C68fitting quarantine',flush=True)


if __name__=='__main__':main()
