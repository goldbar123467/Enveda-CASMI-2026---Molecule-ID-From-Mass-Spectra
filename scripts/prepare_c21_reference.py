"""Freeze C21's unchanged P16 references and search C20 queries without labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from datetime import datetime,timezone
import pyarrow as pa
import pandas as pd
from rdkit import rdBase
from precision_cache import ROOT,sha,write,features
from external_reference_overlay import search
from training_monitor import system_memory


def main():
    pa.set_cpu_count(2);assert rdBase.rdkitVersion=='2026.03.3' and system_memory()['free_gib']>=4
    assert not (ROOT/'.local/compute-paused.json').exists()
    out=ROOT/'outputs/local/c21-negative-tail-20261110';out.mkdir(exist_ok=True)
    assert not (out/'reference-protocol.json').exists() and not (out/'reference-search.json').exists()
    config=ROOT/'configs/c21-negative-tail-20261110.json';policy=ROOT/'configs/c09-external-reference-20261109.json'
    reference=ROOT/'outputs/research/2026-11-09-silver/extlib'
    production=ROOT/'outputs/forks/p16-external-reference-c09-v2/p16_execution_receipt.json'
    original=json.loads(production.read_text());assert sha(policy)==original['policy_sha256']
    for name,pin in original['reference_files'].items():assert sha(reference/name)==pin,name
    inputs=ROOT/'outputs/publish/c20-negative-casmi2022-20261110';contract=json.loads((inputs/'validation-contract.json').read_text())
    for name,pin in contract['files'].items():assert sha(inputs/name)==pin,name
    queries=pd.read_parquet(inputs/'validation-queries.parquet')
    assert len(queries)==queries.molecule_id.nunique()==80 and queries.ionization_mode.eq('negative').all()
    manifests={name:ROOT/path for name,path in [
        ('C08-parent','outputs/publish/c08-models-20261108/c08-manifest.json'),
        ('C09-control','outputs/publish/c09-followup-models-20261109/followup-manifest.json'),
        ('C14-CE','outputs/publish/c14-ce-models-20261110/c08-manifest.json'),
        ('C15-NP','outputs/publish/c15-np-models-20261110/c08-manifest.json'),
        ('C18-peak-drop','outputs/publish/c18-peak-models-20261110/c08-manifest.json')]}
    cfg=json.loads(config.read_text());assert list(manifests)==cfg['model_groups_in_tie_order']
    write(out/'reference-protocol.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha(config),
        preparer_sha256=sha(__file__),source_sha256={n:sha(ROOT/'scripts'/n) for n in ['external_reference_overlay.py','c09_spectrum_audit.py']},
        feature_sha256=sha(ROOT/'outputs/local/kit/features.py'),policy_sha256=sha(policy),
        original_P16_execution_receipt_sha256=sha(production),reference_files_sha256=original['reference_files'],
        query_sha256=sha(inputs/'validation-queries.parquet'),query_contract_sha256=sha(inputs/'validation-contract.json'),
        development_labels_sha256=sha(ROOT/'outputs/local/c20-negative-development-20261110/labels.csv'),
        positive_parent_sha256=sha(ROOT/'outputs/local/c12-deep-candidates-20261110/all/parent.csv'),
        model_manifests={n:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for n,p in manifests.items()},
        labels_read=False,C20_prediction_metrics_read=False,C13_confirmation_metrics_read=False,automatic_submission=False))
    matches,report=search(queries,reference,features,json.loads(policy.read_text()))
    assert len(matches)==160 and matches.groupby('view').molecule_id.nunique().eq(80).all()
    matches.to_parquet(out/'reference-matches.parquet',index=False)
    report.update(completed_utc=datetime.now(timezone.utc).isoformat(),reference_protocol_sha256=sha(out/'reference-protocol.json'),
        matches_sha256=sha(out/'reference-matches.parquet'),config_sha256=sha(config),
        original_P16_reference_and_policy_exact=True,C20_prediction_metrics_read=False,C13_confirmation_metrics_read=False,
        ranking_recipe_selected=False,automatic_submission=False)
    write(out/'reference-search.json',report)
    print('C21 ORIGINAL P16 REFERENCE SEARCH COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
