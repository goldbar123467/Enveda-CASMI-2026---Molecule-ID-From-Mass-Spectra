"""One bounded matched-runtime Engine2 diagnostic, with no automatic retry."""
import os,subprocess,time
from c75_port_utils import ROOT,read,write,sha,stamp
from supervise_c75_local_port import authorization,ram,trim_own_cache


def main():
    source=ROOT/'scripts/probe_c75_retrieval_runtime.py'
    base=ROOT/'outputs/local/c75-local-core-port-20261112';out=base/'retrieval-runtime-probe-v1'
    authorization();assert not out.exists();assert ram()>=12
    receipt=base/'retrieval-runtime-probe-v1-supervisor.json';assert not receipt.exists()
    unit='enveda-c75-retrieval-'+str(os.getpid());wsl=['wsl.exe','-d','Ubuntu-24.04','--']
    command=wsl+['systemd-run','--user','--scope','--unit='+unit,'-p','MemoryHigh=8G','-p','MemoryMax=10G','-p','MemorySwapMax=4G',
        '/usr/bin/env','PYTHONHASHSEED=0','CUDA_VISIBLE_DEVICES=0','/home/imsa/.venvs/enveda-c75-torch210/bin/python',
        '/mnt/c/Users/imsa/Documents/Enveda CASMI/scripts/'+source.name]
    record=dict(started_utc=stamp(),source_sha256=sha(source),supervisor_sha256=sha(__file__),unit=unit,
        host_reserve_GiB=4,memory_high_GiB=8,memory_max_GiB=10,swap_max_GiB=4,launches=1,automatic_retry=False,
        USED_only=True,quality_read=False,C13_read=False,scoring_entries=0)
    write(receipt,record)
    with receipt.with_suffix('.log').open('w',encoding='utf8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);deadline=time.monotonic()+2400
        try:
            while child.poll() is None:
                authorization();assert time.monotonic()<deadline
                free=ram()
                if free<4:trim_own_cache();time.sleep(5);free=ram();assert free>=4,free
                write(receipt,dict(record,checked_utc=stamp(),status='running',host_free_GiB=free));time.sleep(10)
            assert child.returncode==0,child.returncode
            assert read(out/'results.json')['parent_source_files_unchanged']
        except BaseException as exc:
            if child.poll() is None:
                group=subprocess.check_output(wsl+['systemctl','--user','show',unit+'.scope','-p','ControlGroup','--value'],text=True).strip()
                assert group.endswith('/'+unit+'.scope'),group
                subprocess.run(wsl+['systemctl','--user','stop',unit+'.scope'],check=True,timeout=30);child.wait(timeout=30)
            write(receipt,dict(record,completed_utc=stamp(),status='failed',error=repr(exc)));raise
    write(receipt,dict(record,completed_utc=stamp(),status='complete',results_sha256=sha(out/'results.json')))
    print('C75 BOUNDED MATCHED RUNTIME RETRIEVAL COMPLETE',flush=True)


if __name__=='__main__':main()
