"""Seal generic deployment integration before any reserved label metrics."""
import json
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write
import develop_c21_negative as c21
import develop_c24_negative as c24
from c24_models import PACKAGE as C24_PACKAGE,serving_module as c24_module


def main():
    out=ROOT/'outputs/local/c26-deployment-audit-20261110'
    assert not (out/'deployment-source-freeze.json').exists()
    assert not (ROOT/'outputs/local/c13-confirmation-20261110/evaluation-consumption.json').exists()
    c21.verify();c24.verify()
    audit=json.loads((out/'audit.json').read_text())
    assert audit['all120_outputs_equal_existing_negative_tail_composition'] and audit['all4_fallback_rows_exact_parent']
    assert not audit['labels_read'] and not audit['quality_metrics_computed']
    assert audit['deployment_receipt']['source_sha256']==sha(ROOT/'scripts/negative_deployment.py')
    groups=c21.groups();c24_module();groups['C24-polarity-loss']=(C24_PACKAGE,'polarity-loss',None)
    order=json.loads((c24.OUT/'development-protocol.json').read_text())['global_tie_order'];assert list(groups)==order
    models={}
    for name,(bundle,arm,_) in groups.items():
        manifest=json.loads((bundle/'c08-manifest.json').read_text())
        for filename,pin in manifest['files'].items():assert sha(bundle/filename)==pin,filename
        models[name]=dict(bundle=str(bundle.relative_to(ROOT)),arm=arm,manifest_sha256=sha(bundle/'c08-manifest.json'),
            files_sha256=manifest['files'])
    sources=['negative_deployment.py','negative_tail.py','compatible_tail.py','seal_negative_deployment.py']
    write(out/'deployment-source-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={n:sha(ROOT/'scripts'/n) for n in sources},models=models,weight=.2,offset=3.,
        C21_protocol_sha256=sha(c21.OUT/'development-protocol.json'),C24_protocol_sha256=sha(c24.OUT/'development-protocol.json'),
        C25_source_sha256=sha(ROOT/'scripts/confirm_c25_negative.py'),
        C25_config_sha256=sha(ROOT/'configs/c25-reserved-negative-20261110.json'),
        C25_control_source_freeze_sha256=sha(ROOT/'outputs/local/c25-reserved-negative-20261110/control-source-freeze.json'),
        original_P16_source_sha256=sha(ROOT/'kaggle/forks/p16-external-reference-c09/notebook.ipynb'),
        C20_recovery_notebook_sha256=sha(ROOT/'kaggle/forks/c20-generated-recovery/notebook.ipynb'),
        parity_audit_sha256=sha(out/'audit.json'),C13_confirmation_metrics_read=False,selected_group=None,
        selection='Only the global six-group winner frozen by C25 may bind one of these unchanged manifests. This seal adds no group, weight or model choice.',
        integration='Exactly the existing model score and negative_tail.merge at0.20/offset3. Homogeneous known-positive rows and explicit mixed/unknown/absent or unsupported-negative rows retain the exact P16 parent.',
        qualification='Actual Kaggle changed-path smoke and independent selected-model/output/fallback/source/identity audits still required; no release or public submission is authorized by this generic seal alone.',
        automatic_upload=False,automatic_submission=False,submission_qualified=False))
    print('DONE PRE-CONFIRMATION DEPLOYMENT SOURCE SEAL',sha(out/'deployment-source-freeze.json'),flush=True)


if __name__=='__main__':main()
