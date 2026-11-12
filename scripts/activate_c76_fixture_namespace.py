"""Switch only the verified task-owned /kaggle/working link as its root owner.

No directory permissions, datasets or application state are changed. The original
C75 input link, ownership marker and all prior working data remain intact.
"""
import os
from pathlib import Path
from c75_port_utils import ROOT,read,write,sha,stamp


def main():
    assert os.geteuid()==0
    base=ROOT/'outputs/local/c76-numerical-portability-20261112/fixture-port'
    assert read(base/'independent-source-audit.json')['passed']
    assert read(base/'supervisor-failed.json')['status']=='failed' and not (base/'run-intent.json').exists()
    assert not (base/'namespace-activation.json').exists()
    pin=read(base/'fixture-freeze.json');namespace=Path('/kaggle')
    marker=read(namespace/'.c75-owner.json')
    assert marker['owner']=='enveda-c75-local-core-port-20261112' and marker['workspace']==str(ROOT)
    assert set(p.name for p in namespace.iterdir())=={'input','working','.c75-owner.json'}
    input_link=namespace/'input';link=namespace/'working'
    assert input_link.is_symlink() and str(input_link.resolve())==pin['original_input_target']
    assert link.is_symlink() and link.resolve()==Path('/home/imsa/.local/share/enveda-c75/fixture/working')
    working=Path(pin['owned_working'])
    assert working==Path('/home/imsa/.local/share/enveda-c76/fixture/working') and not working.is_symlink()
    assert working.is_dir() and not any(working.iterdir()) and working.stat().st_uid==1000
    temporary=namespace/'.c76-owned-working-link';assert not temporary.exists() and not temporary.is_symlink()
    temporary.symlink_to(working,target_is_directory=True);os.replace(temporary,link)
    assert link.resolve()==working
    write(base/'namespace-activation.json',dict(activated_utc=stamp(),source_sha256=sha(__file__),fixture_freeze_sha256=sha(base/'fixture-freeze.json'),
        working_target=str(working),input_target=str(input_link.resolve()),owned_GL_package=pin['owned_GL_package'],
        only_verified_owned_working_link_switched=True,no_directory_permissions_changed=True,all_old_data_retained=True))
    print('C76 VERIFIED TASK WORKING POINTER ACTIVATED; PERMISSIONS AND DATA RETAINED',flush=True)


if __name__=='__main__':main()
