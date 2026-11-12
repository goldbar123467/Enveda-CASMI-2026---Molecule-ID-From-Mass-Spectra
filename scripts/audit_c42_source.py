"""Independent source-boundary and complete-stage failure cleanup checks."""
import ast,base64,hashlib,json,shutil,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import pandas as pd
from precision_cache import ROOT,sha,write
from build_c09_controls import unwrap
from build_c13_formate_recovery import value
from build_general_anchor import guarded_source
from submit_c09_external_once import scientific_sha


def main():
    out=ROOT/'outputs/local/c42-crossmodal-engineering-20261111'
    assert not (out/'production-source-audit.json').exists()
    original=json.loads((ROOT/'kaggle/forks/p16-external-reference-c09/notebook.ipynb').read_text(encoding='utf8'))
    base_outer=unwrap(''.join(original['cells'][12]['source']))
    base_frozen=json.loads(base64.b64decode(value(base_outer,'FROZEN_B64')))
    audits={};headpin=None
    for trial,slug in [('cpu','p20-c42-cpu-alignment'),('gpu','p21-c42-gpu-alignment')]:
        folder=ROOT/'kaggle/forks'/slug;pin=json.loads((folder/'production-pin.json').read_text())
        assert sha(folder/'notebook.ipynb')==pin['notebook_sha256']
        nb=json.loads((folder/'notebook.ipynb').read_text(encoding='utf8'));assert scientific_sha(nb)==pin['scientific_sha256']
        assert len(nb['cells'])==len(original['cells'])+1
        changed=[i for i in range(len(original['cells'])) if nb['cells'][i]['source']!=original['cells'][i]['source']]
        assert changed==[12]
        outer=unwrap(''.join(nb['cells'][12]['source']));frozen=json.loads(base64.b64decode(value(outer,'FROZEN_B64')))
        changed=[i for i in range(len(base_frozen['cells'])) if frozen['cells'][i]['source']!=base_frozen['cells'][i]['source']]
        assert changed==[27]
        expected=''.join(base_frozen['cells'][27]['source'])
        actual=''.join(frozen['cells'][27]['source'])
        inserted=actual.replace(expected.split("V4CODE = V43_DETERMINISM['work_source_root']")[0],'',1)
        assert 'patch_legacy_tree' in inserted and 'c42_legacy_compatibility.json' in inserted
        assert 'CASMI_C41' not in outer and 'install_library_guard' not in outer
        metadata=json.loads((folder/'kernel-metadata.json').read_text())
        assert metadata['is_private'] and not metadata['enable_internet'] and not any('c41' in s for s in metadata['dataset_sources'])
        body=unwrap(''.join(nb['cells'][-1]['source']));assert value(body,'C42_TRIAL')==trial
        assert pin['head_manifest_sha256']==value(body,'C42_HEAD_PIN')
        if headpin is None:headpin=pin['head_manifest_sha256']
        assert headpin==pin['head_manifest_sha256']
        # Simulate a required preflight failure after the real parent exists.
        # Only output paths and GPU inventory are fixtures; the complete sealed
        # stage body executes unchanged and must remove its scoring filename.
        with tempfile.TemporaryDirectory(prefix='c42-stage-cleanup-') as temp:
            root=Path(temp);input_root=root/'input';input_root.mkdir()
            parent=b'molecule_id,smiles\nsynthetic,CCO\n';(root/'submission.csv').write_bytes(parent)
            for n in ['p16_execution_receipt.json','c42_legacy_compatibility.json']:(root/n).write_text('{}')
            scope=dict(ROOT=root,INPUT=input_root,MERGE_REPORT={'status':'applied'},hashlib=hashlib,shutil=shutil,
                json=json,pd=pd,sys=sys,torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,get_device_name=lambda _: 'Tesla T4')))
            try:exec(guarded_source(body,'c42_missing_required_bundle',root/'submission.csv'),scope)
            except AssertionError:pass
            else:raise AssertionError('Missing required model must fail')
            assert not (root/'submission.csv').exists() and (root/'c42_parent.csv').read_bytes()==parent
        audits[trial]=dict(notebook_sha256=pin['notebook_sha256'],scientific_sha256=pin['scientific_sha256'],
            parent_only_formate_source_copy=True,other_parent_cells_exact=True,no_validation_masks_mounted=True,
            complete_stage_preflight_cleanup_passed=True,synthetic_only=True)
    write(out/'production-source-audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        head_manifest_sha256=headpin,trials=audits,quality_metrics_read=False,C13_read=False,cloud_smoke_still_required=True))
    print('C42 SOURCE AUDIT PASS',json.dumps(audits),flush=True)


if __name__=='__main__':main()
