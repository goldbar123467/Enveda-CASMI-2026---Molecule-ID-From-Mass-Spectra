"""Independent source-diff and complete-acquisition checks before local inference."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(n,'2')
import torch
import argparse,ast,base64,hashlib,json
from pathlib import Path
import pandas as pd
from c75_port_utils import ROOT,read,write,sha,stamp,unwrap,value,replace_value
OUT=ROOT/'outputs/local/c75-local-core-port-20261112'


def child_source(code):
    values=[]
    for node in ast.walk(ast.parse(code)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and node.args:
            try:text=ast.literal_eval(node.args[0])
            except (ValueError,TypeError):continue
            if isinstance(text,str) and 'REAL ENCODER PREFLIGHT' in text:values.append(text)
    assert len(values)==1
    return values[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fixture','full']);args=parser.parse_args();mode=args.mode
    target=OUT/(mode+'-port');receipt=target/'port-source-audit.json';assert not receipt.exists()
    pin=read(target/'port-freeze.json');original_path=ROOT/pin['original_notebook']
    assert sha(original_path)==pin['original_notebook_sha256'] and sha(target/'notebook.ipynb')==pin['local_notebook_sha256']
    assert pin['source_sha256']==sha(ROOT/'scripts/prepare_c75_port.py')
    original=read(original_path);local=read(target/'notebook.ipynb')
    a={i:unwrap(''.join(c['source'])) for i,c in enumerate(original['cells']) if c['cell_type']=='code'}
    b={i:''.join(c['source']) for i,c in enumerate(local['cells']) if c['cell_type']=='code'}
    assert set(a)==set(b)
    changed={i for i in a if a[i]!=b[i]};assert changed==({0,6,12,27} if mode=='fixture' else {0,12,27})
    for i in a:
        assert hashlib.sha256(a[i].encode()).hexdigest()==pin['original_outer_cell_sha256'][str(i)]
        assert hashlib.sha256(b[i].encode()).hexdigest()==pin['local_outer_cell_sha256'][str(i)]
    prefix='c57' if mode=='fixture' else 'c73';old_package=ROOT/f'outputs/publish/{prefix}-full-masked-validation-20261111'
    oldhash=sha(old_package/'validation-contract.json');newhash=sha(target/'query-package/validation-contract.json');assert newhash==pin['query_contract_sha256']
    expected6=a[6];expected27=a[27]
    if mode=='fixture':
        expected6=expected6.replace(oldhash,newhash).replace('==482','==12').replace('queries=482','queries=12')
        expected27=expected27.replace(oldhash,newhash).replace('==482','==12').replace('queries=482','queries=12')
    expected27=expected27.replace("os.environ['CUDA_VISIBLE_DEVICES']='1'","os.environ['CUDA_VISIBLE_DEVICES']='0'")
    assert b[6]==expected6 and b[27]==expected27
    # Replacing just the payload and its checksum must restore the original
    # surrounding module definitions, ensemble code and every ranking setting.
    canon=lambda s:replace_value(replace_value(s,'FROZEN_B64','PAYLOAD'),'FROZEN_SHA256','PAYLOAD_SHA')
    assert canon(a[12])==canon(b[12])
    oldbytes=base64.b64decode(value(a[12],'FROZEN_B64'));newbytes=base64.b64decode(value(b[12],'FROZEN_B64'))
    assert hashlib.sha256(oldbytes).hexdigest()==pin['original_inner_sha256']=='3f973abbb1593e356db4f6dec149c8ad99f5afa5ded8da8bab59152bd49e60e7'
    assert hashlib.sha256(newbytes).hexdigest()==value(b[12],'FROZEN_SHA256')==pin['local_inner_sha256']
    inner_a=json.loads(oldbytes);inner_b=json.loads(newbytes)
    assert set(inner_a)==set(inner_b)
    for key in inner_a:
        if key!='cells':assert inner_a[key]==inner_b[key]
    assert len(inner_a['cells'])==len(inner_b['cells'])
    for i,(x,y) in enumerate(zip(inner_a['cells'],inner_b['cells'])):
        if i!=43:assert x==y,('Unexpected scientific cell adaptation',i)
        else:
            for key in x:
                if key!='source':assert x[key]==y[key]
            expected=''.join(x['source']).replace("'CUDA_VISIBLE_DEVICES':'1'","'CUDA_VISIBLE_DEVICES':'0'")
            assert ''.join(y['source'])==expected
    assert child_source(b[0])==child_source(a[0]).replace('P14 GPU1 REAL ENCODER PREFLIGHT','P14 LOCAL GPU0 REAL ENCODER PREFLIGHT')
    assert "CUDA_VISIBLE_DEVICES='0'" in b[0] and 'actual_two_T4s=False' in b[0] and "'2070' in" in b[0]
    package=target/'query-package';contract=read(package/'validation-contract.json')
    assert not contract['truth_injection'] and not contract['query_labels_visible_to_pipeline']
    for name,digest in contract['files'].items():assert sha(package/name)==digest
    assert set(p.name for p in package.iterdir())==set(contract['files'])|{'validation-contract.json'}
    query=pd.read_parquet(package/'validation-queries.parquet');original_query=pd.read_parquet(old_package/'validation-queries.parquet')
    ids=set(query.molecule_id);expected=original_query[original_query.molecule_id.isin(ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(query.reset_index(drop=True),expected,check_exact=True)
    sample=pd.read_csv(package/'validation-sample.csv',dtype=str)
    assert list(dict.fromkeys(query.molecule_id))==sample.molecule_id.tolist()
    assert query.molecule_id.nunique()==contract['queries']==pin['queries'] and len(query)==contract['spectra']==pin['acquisitions']
    assert not {'key','smiles','formula','inchikey14','source_number'}&set(query.columns)
    for name in ['main-reference-sid-exclusion.npy','engine2-raw-key-exclusion.json']:assert sha(package/name)==sha(old_package/name)
    write(receipt,dict(completed_utc=stamp(),source_sha256=sha(__file__),port_freeze_sha256=sha(target/'port-freeze.json'),mode=mode,
        queries=len(ids),acquisitions=len(query),all_original_acquisitions_metadata_peaks_exact=True,reference_masks_exact=True,
        original_outer_neural_ensemble_and_ranking_code_exact=True,every_inner_scientific_cell_exact_except_one_GPU_index_mapping=True,
        original_encoder_preflight_preserved_on_actual_single_card=True,input_count_and_contract_changes_only=True,
        no_query_structure_labels_mounted=True,source_and_fixture_audit_passed=True,runtime_parity_not_yet_established=True,
        new_source_quality_read=False,C13_read=False,scoring_entries=0))
    print('C75 INDEPENDENT SOURCE/ACQUISITION AUDIT PASS',mode,len(ids),len(query),'runtime parity still required',flush=True)


if __name__=='__main__':main()
