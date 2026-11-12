"""One guarded complete C76 USED12 run; no fresh quality, submission or restart."""
import os,subprocess,time
from c75_port_utils import ROOT,read,write,sha,stamp
from supervise_c75_local_port import authorization,ram,trim_own_cache


def main():
    base=ROOT/'outputs/local/c76-numerical-portability-20261112';out=base/'fixture-port'
    authorization();assert not (out/'run-intent.json').exists();assert ram()>=12
    source=ROOT/'scripts/run_c76_fixture.py';pin=read(out/'fixture-freeze.json')
    assert pin['notebook_sha256']==sha(out/'notebook.ipynb')
    receipt=out/'supervisor-intent.json';assert not receipt.exists()
    unit='enveda-c76-fixture-'+str(os.getpid());wsl=['wsl.exe','-d','Ubuntu-24.04','--']
    command=wsl+['systemd-run','--user','--scope','--unit='+unit,'-p','MemoryHigh=8G','-p','MemoryMax=10G','-p','MemorySwapMax=4G',
        '/usr/bin/env','PYTHONHASHSEED=0','CUDA_VISIBLE_DEVICES=0','/home/imsa/.venvs/enveda-c75-torch210/bin/python',
        '/mnt/c/Users/imsa/Documents/Enveda CASMI/scripts/'+source.name]
    record=dict(started_utc=stamp(),source_sha256=sha(source),supervisor_sha256=sha(__file__),unit=unit,
        fixture_freeze_sha256=sha(out/'fixture-freeze.json'),host_reserve_GiB=4,memory_high_GiB=8,memory_max_GiB=10,swap_max_GiB=4,
        launches=1,automatic_retry=False,USED_only=True,quality_read=False,C13_read=False,scoring_entries=0)
    write(receipt,record)
    with (out/'pipeline.log').open('w',encoding='utf8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);deadline=time.monotonic()+7200
        try:
            while child.poll() is None:
                authorization();assert time.monotonic()<deadline
                free=ram()
                if free<4:trim_own_cache();time.sleep(5);free=ram();assert free>=4,free
                write(out/'supervisor-status.json',dict(record,checked_utc=stamp(),status='running',host_free_GiB=free));time.sleep(10)
            assert child.returncode==0,child.returncode
            assert read(out/'run-complete.json')['queries']==12
        except BaseException as exc:
            if child.poll() is None:
                group=subprocess.check_output(wsl+['systemctl','--user','show',unit+'.scope','-p','ControlGroup','--value'],text=True).strip()
                assert group.endswith('/'+unit+'.scope'),group
                subprocess.run(wsl+['systemctl','--user','stop',unit+'.scope'],check=True,timeout=30);child.wait(timeout=30)
            write(out/'supervisor-failed.json',dict(record,completed_utc=stamp(),status='failed',error=repr(exc)));raise
    write(out/'supervisor-complete.json',dict(record,completed_utc=stamp(),status='complete',run_complete_sha256=sha(out/'run-complete.json')))
    print('C76 GUARDED COMPLETE USED12 RUN FINISHED',flush=True)


if __name__=='__main__':main()
