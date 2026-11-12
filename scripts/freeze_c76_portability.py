"""Freeze the explicit repair candidate before any fresh source outcomes."""
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c75-local-core-port-20261112'
    out=ROOT/'outputs/local/c76-numerical-portability-20261112';out.mkdir(exist_ok=True)
    dest=out/'candidate-freeze.json';assert not dest.exists()
    auth=read(ROOT/'.local/c38-active-goal.json');assert auth['active'] and auth['submissions_used']==0
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    result=read(base/'portable-breakpoints-v1/results.json');assert result['predeclared_component_criterion_passed']
    assert result['comparison']['scores']==result['comparison']['mass_bins_exact']==16
    pair=ROOT/'outputs/local/c71-final-clean-direct-pair-20261112'
    assert sha(pair/'cpu-model.npz')=='96c4e05c7a419f38a0299b102e9007bb317fd1ff4605d65f8663512c6f6889e6'
    assert sha(pair/'gpu-model.npz')=='85b4dcddc650ffb3036733b76473c3eaa93094b5d7b50eea1dde5436f5a52c9d'
    files=[ROOT/'configs/c76-numerical-portability-v1.json',ROOT/'scripts/c76_gl_portability.py',ROOT/'scripts/run_c76_gl_portable.py',
        ROOT/'scripts/probe_c76_extended_portability.py',ROOT/'scripts/freeze_c76_portability.py',
        pair/'complete-candidate-freeze.json',pair/'complete-pipeline-binding.json',pair/'cpu-model.npz',pair/'gpu-model.npz',
        ROOT/'outputs/local/c72-clean-target-holdout-20261112/validation-freeze.json',
        ROOT/'outputs/local/c72-clean-target-holdout-20261112/preparation-protocol.json',
        base/'portable-breakpoints-v1/intent.json',base/'portable-breakpoints-v1/results.json']
    write(dest,dict(frozen_utc=stamp(),source_sha256=sha(__file__),candidate='C76 numerical portability',
        files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        fixed_CPU_model_sha256=sha(pair/'cpu-model.npz'),fixed_GPU_model_sha256=sha(pair/'gpu-model.npz'),
        previous_complete_parent_binding_sha256=sha(pair/'complete-pipeline-binding.json'),
        complete_recipe_revision_required=True,original_C75_gate_passed=False,
        quality_outcomes_read=False,C13_read=False,scoring_entries=0,qualified_for_scoring=False))
    print('C76 NUMERICAL REPAIR CANDIDATE FROZEN; FULL FRESH AND SCORING STILL HELD',flush=True)


if __name__=='__main__':main()
