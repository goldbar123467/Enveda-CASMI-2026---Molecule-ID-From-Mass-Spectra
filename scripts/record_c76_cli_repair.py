"""Record the single CLI-arity recovery with original source/evidence preserved."""
import copy
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';archive=base/'cli-adapter-repair-v1'
    assert not (base/'candidate-freeze-r1.json').exists()
    original=read(base/'candidate-freeze.json');assert sha(base/'candidate-freeze.json')==sha(archive/'original-candidate-freeze.json')
    assert read(base/'extended-used-nine-supervisor.json')['status']=='failed'
    failed=read(base/'extended-used-nine/cpu.json.meta.json');assert failed['status']=='error' and failed['n_mols_scored']==0
    assert '4 were given' in (base/'extended-used-nine/cpu.log').read_text()
    changed=['scripts/run_c76_gl_portable.py','scripts/probe_c76_extended_portability.py']
    for name,value in original['files_sha256'].items():
        if name in changed:assert sha(archive/(ROOT/name).name)==value
        else:assert sha(ROOT/name)==value,name
    assert not (ROOT/'outputs/local/c72-clean-target-holdout-20261112/evaluation-consumption.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    revision=copy.deepcopy(original);revision['files_sha256'].update({n:sha(ROOT/n) for n in changed})
    revision.update(frozen_utc=stamp(),original_candidate_freeze_sha256=sha(base/'candidate-freeze.json'),
        operational_repair_source_sha256=sha(__file__),
        operational_repair='Preserve original optional checkpoint CLI parameter, verify its pin and pass it to both identical providers. New retry output paths only.',
        numerical_algorithm_unchanged=True,models_cases_tolerances_unchanged=True,single_reconciled_retry=True,
        original_executed_sources_archive_sha256={p.name:sha(p) for p in archive.iterdir() if p.is_file()})
    write(base/'candidate-freeze-r1.json',revision)
    print('C76 CLI-ONLY RECOVERY FROZEN; ORIGINAL FAILED ATTEMPT PRESERVED',flush=True)


if __name__=='__main__':main()
