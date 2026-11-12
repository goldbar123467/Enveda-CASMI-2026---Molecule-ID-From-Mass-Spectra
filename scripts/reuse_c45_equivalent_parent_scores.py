"""Reuse one frozen head execution only if both complete parent inputs are exact.

This records a derived result, never a second GPU execution. Independent ranking
audits still reconstruct each output against its actual parent.
"""
import json,shutil
from datetime import datetime,timezone
import pandas as pd
from c45_complete_parent import ROOT,OUT,read,write,sha


def main():
    source=OUT/'heads-P01';target=OUT/'heads-P16';assert not target.exists()
    freeze=read(OUT/'complete-parent-freeze.json');audit=read(OUT/'C39-parent/independent-parent-audit.json')
    assert audit['independent_fixed_policy_reconstruction_exact'] and audit['changed_lists']==0
    for name,pin in freeze['files_sha256'].items():assert sha(OUT/name)==pin
    a=pd.read_csv(OUT/'P01.csv',dtype=str,keep_default_na=False);b=pd.read_csv(OUT/'P16.csv',dtype=str,keep_default_na=False)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    receipt=read(source/'score-receipt.json');assert receipt['baseline_sha256']==sha(OUT/'P01.csv')
    assert receipt['query_sha256']==sha(OUT/'queries.parquet')
    for name,pin in receipt['files_sha256'].items():assert sha(source/name)==pin
    ranking=read(source/'independent-ranking-audit.json');assert ranking['score_receipt_sha256']==sha(source/'score-receipt.json')
    assert all(value['independent_RRF_exact'] for value in ranking['trials'].values())
    target.mkdir()
    for name in receipt['files_sha256']:shutil.copy2(source/name,target/name)
    proof=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        complete_parent_freeze_sha256=sha(OUT/'complete-parent-freeze.json'),parent_audit_sha256=sha(OUT/'C39-parent/independent-parent-audit.json'),
        original_score_receipt_sha256=sha(source/'score-receipt.json'),original_ranking_audit_sha256=sha(source/'independent-ranking-audit.json'),
        P01_sha256=sha(OUT/'P01.csv'),P16_sha256=sha(OUT/'P16.csv'),queries_sha256=sha(OUT/'queries.parquet'),
        all_parent_input_rows_and_strings_exact=True,all_query_inputs_exact=True,
        models_or_scoring_code_changed=False,second_GPU_execution_performed=False,quality_outcomes_read=False)
    write(target/'exact-parent-reuse.json',proof)
    receipt.update(baseline_sha256=sha(OUT/'P16.csv'),execution_kind='derived_from_exact_equivalent_parent',
                   exact_parent_reuse_sha256=sha(target/'exact-parent-reuse.json'),second_GPU_execution_performed=False)
    write(target/'score-receipt.json',receipt);print('C45 EXACT EQUIVALENT PARENT REUSE',json.dumps(proof))


if __name__=='__main__':main()
