"""Run the one fixed clean CPU/GPU fit after the prospective boundary audit."""
import argparse,json,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from c45_complete_parent import ROOT,read,write,sha
CONFIG=ROOT/'configs/c48-clean-residual-pair-20261112.json'
OUT=ROOT/'outputs/local/c48-clean-residual-heads-20261112'
OLD=ROOT/'outputs/local/c40r1-crossmodal-heads-20261111'


def main():
    import msvcrt
    lock=(OUT/'fixed-fit-worker.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert read(ROOT/'.local/c38-active-goal.json')['active']
    boundary=read(OUT/'fit-boundary-freeze.json');assert boundary['all_original_exclusions_preserved'] and boundary['new_query_identity_overlap']==boundary['original_raw_alias_survivors']==0
    assert boundary['source_sha256']==sha(ROOT/'scripts/prepare_c48_fit_boundary.py') and boundary['config_sha256']==sha(CONFIG)
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    for name,pin in boundary['files_sha256'].items():assert sha(OUT/name)==pin
    helpers=['fit_c48_clean_pair.py','prepare_c40_head_features.py','train_c40_cpu.py','train_c48_residual_gpu.py','c48_gpu_model.py','crossmodal_heads.py']
    contract=dict(config_sha256=sha(CONFIG),fit_boundary_sha256=sha(OUT/'fit-boundary-freeze.json'),
        source_sha256={name:sha(ROOT/'scripts'/name) for name in helpers},query_encoding_context_sha256=sha(OLD/'query-encoding-freeze.json'),
        C39_outcomes_known_before_design=True,auxiliary_source_outcomes_read=False,C13_read=False,maximum_CPU_fits=1,maximum_GPU_fits=1)
    pin=OUT/'fixed-fit-worker-contract.json'
    if pin.exists():assert read(pin)==contract
    else:write(pin,contract)
    old_protocol=read(OLD/'head-feature-protocol.json');assert sha(OLD/'query-encoding-freeze.json')==old_protocol['query_encoding_sha256']
    if not (OUT/'query-encoding-freeze.json').exists():shutil.copy2(OLD/'query-encoding-freeze.json',OUT/'query-encoding-freeze.json')
    assert sha(OUT/'query-encoding-freeze.json')==contract['query_encoding_context_sha256']
    write(OUT/'query-encoding-context-reuse.json',dict(source='C40R1 query encoding for the SAME C39 panel; metadata context reused byte-exactly.',
        source_sha256=contract['query_encoding_context_sha256'],no_encoder_inference_performed=True,
        auxiliary_queries_encoded=False,training_features_do_not_use_query_rows=True))
    def status(phase,**kwargs):
        write(OUT/'fixed-fit-worker-status.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,contract_sha256=sha(pin),**kwargs))
        print('C48 FIXED FIT',phase,json.dumps(kwargs),flush=True)
    for name,script in [('head-feature-freeze.json','prepare_c40_head_features.py'),('cpu-ridge/complete.json','train_c40_cpu.py'),('gpu-residual/complete.json','train_c48_residual_gpu.py')]:
        if (OUT/name).exists():
            status('completed_phase_exists_verify',artifact=name)
            if name=='head-feature-freeze.json':
                for file,pin_value in read(OUT/name)['files_sha256'].items():assert sha(OUT/file)==pin_value
            else:
                receipt=read(OUT/name);model=OUT/('cpu-ridge/model.npz' if name.startswith('cpu') else 'gpu-residual/final_model.pt')
                assert receipt['model_sha256']==sha(model) and receipt['feature_freeze_sha256']==sha(OUT/'head-feature-freeze.json')
            continue
        assert sha(CONFIG)==contract['config_sha256'] and all(sha(ROOT/'scripts'/file)==value for file,value in contract['source_sha256'].items())
        status('executing_fixed_phase',script=script)
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--plan',str(CONFIG)],cwd=ROOT,check=True)
    status('both_fixed_fits_complete',CPU_model_sha256=sha(OUT/'cpu-ridge/model.npz'),GPU_model_sha256=sha(OUT/'gpu-residual/final_model.pt'),new_source_quality_read=False)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():write(OUT/'fixed-fit-worker-attention.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),error_type=type(error).__name__,reason=str(error),automatic_retry=False))
        raise
