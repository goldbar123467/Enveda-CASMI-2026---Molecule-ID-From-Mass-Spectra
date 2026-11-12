"""Switch only the two task-owned namespace links as their existing root owner."""
import os
import fcntl
from pathlib import Path
from c75_port_utils import ROOT, read, write, sha, stamp


def main():
    assert os.geteuid() == 0
    base = ROOT / 'outputs/local/c76-numerical-portability-20261112'
    out = base / 'local-production-preview'; pin = read(out / 'preview-freeze.json')
    assert read(out / 'independent-source-audit.json')['passed']
    plan = read(out / 'launch-plan.json')
    assert plan['files_sha256']['scripts/activate_c76_local_production.py'] == sha(__file__)
    for name, digest in plan['files_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert not (out / 'namespace-activation.json').exists() and not (out / 'run-intent.json').exists()
    assert read(base / 'fixture-resource-port/supervisor-complete.json')['status'] == 'complete'
    locks = []
    for path in [ROOT / 'outputs/local/c75-local-core-port-20261112/local-inference.lock', base / 'local-inference.lock']:
        handle = path.open('a+b'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    namespace = Path('/kaggle'); marker = read(namespace / '.c75-owner.json')
    assert marker['owner'] == 'enveda-c75-local-core-port-20261112' and marker['workspace'] == str(ROOT)
    assert set(p.name for p in namespace.iterdir()) == {'input', 'working', '.c75-owner.json'}
    targets = {'input': Path(pin['input']), 'working': Path(pin['working'])}
    for name, prior in [('input', pin['original_input_target']), ('working', pin['original_working_target'])]:
        link = namespace / name; destination = targets[name]
        assert link.is_symlink() and str(link.resolve()) == prior
        assert destination.is_dir() and not destination.is_symlink() and destination.stat().st_uid == 1000
    assert not any(targets['working'].iterdir())
    temporary = {name: namespace / ('.c76-production-' + name + '-link') for name in targets}
    assert all(not path.exists() and not path.is_symlink() for path in temporary.values())
    for name, destination in targets.items(): temporary[name].symlink_to(destination, target_is_directory=True)
    for name in targets: os.replace(temporary[name], namespace / name)
    assert all((namespace / name).resolve() == destination for name, destination in targets.items())
    write(out / 'namespace-activation.json', dict(activated_utc=stamp(), source_sha256=sha(__file__),
        preview_freeze_sha256=sha(out / 'preview-freeze.json'), launch_plan_sha256=sha(out / 'launch-plan.json'),
        input_target=str(targets['input']), working_target=str(targets['working']),
        only_verified_owned_namespace_links_switched=True, all_prior_working_data_retained=True,
        no_permission_or_security_policy_changes=True))
    print('C76 LOCAL PRODUCTION OWNED INPUT/WORKING LINKS ACTIVATED', flush=True)


if __name__ == '__main__': main()
