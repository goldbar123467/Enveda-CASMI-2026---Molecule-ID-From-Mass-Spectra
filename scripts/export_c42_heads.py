"""Seal the two corrected C40R1 heads before any C39 outcome is read."""
import json,shutil
from datetime import datetime,timezone
from precision_cache import ROOT,sha,write


def main():
    source=ROOT/'outputs/local/c40r1-crossmodal-heads-20261111'
    out=ROOT/'outputs/publish/c42-crossmodal-heads-20261111'
    assert json.loads((ROOT/'.local/c38-active-goal.json').read_text())['active']
    assert not (out/'head-manifest.json').exists()
    final=json.loads((source/'original-row-finalization-audit.json').read_text())
    assert final['head_fit_boundary_passed'] and final['final_quarantined_ids_already_absent'] and not final['additional_refit_required'],final
    out.mkdir(parents=True,exist_ok=True)
    pairs={'cpu-model.npz':source/'cpu-ridge/model.npz',
           'gpu-model.pt':source/'gpu-nonlinear/final_model.pt',
           'fit-scaling.npz':source/'fit-scaling.npz',
           'features.py':ROOT/'outputs/local/kit/features.py',
           'trial-plan.json':ROOT/'configs/c42-crossmodal-trials-20261111.json'}
    for name in ['c40_gpu_model.py','crossmodal_heads.py','crossmodal_tail.py','score_c42_crossmodal.py']:
        pairs[name]=ROOT/'scripts'/name
    for kind,name in [('cpu-ridge','cpu-model.npz'),('gpu-nonlinear','gpu-model.pt')]:
        completed=json.loads((source/kind/'complete.json').read_text())
        assert completed['model_sha256']==sha(pairs[name])
        assert not completed['validation_metrics_read'] and not completed['C13_read']
    for name,path in pairs.items():shutil.copy2(path,out/name)
    parent=ROOT/'outputs/publish/frozen-p14-20261107'
    enc=json.loads((parent/'frozen-manifest.json').read_text())
    manifest=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),run_id='C42',fitted_heads='C40R1',
        files={n:sha(out/n) for n in pairs},encoder_bundle_manifest_sha256=sha(parent/'frozen-manifest.json'),
        encoders={n:enc['files'][n] for n in ['dreams.ts','molformer.ts','tokenizer.json']},
        boundary_audit_sha256=sha(source/'original-row-finalization-audit.json'),
        fit_boundary_sha256=sha(source/'fit-boundary-freeze.json'),
        feature_freeze_sha256=sha(source/'head-feature-freeze.json'),
        fitting_identities=174328,fitting_acquisitions=600815,
        spectrum_preprocessing='Stable intensity/original-index cutoff, top100+precursor1.1; float16 stored embeddings.',
        candidate_cache='Exact canonical SMILES only; over202token candidates ineligible; complete formula-group fallback.',
        CPU_trial='CPU-fitted ridge head over frozen GPU encoders and complete GPU P01 parent; not CPU-only execution.',
        GPU_trial='Fresh nonlinear GPU head, fixed final30000updates; same frozen encoders and parent.',
        query_features='Mean of all supported positive acquisitions; pure-negative/unsupported parent fallback.',
        quality_read=False,C13_read=False,qualified_for_submission=False)
    write(out/'head-manifest.json',manifest)
    write(out/'dataset-metadata.json',dict(id='clarkkitchen/casmi26-c42-crossmodal-heads-20261111',
        title='CASMI26 C42 CPU and GPU alignment heads',licenses=[dict(name='CC-BY-NC-4.0')]))
    print('SEALED C42',sha(out/'head-manifest.json'),flush=True)


if __name__=='__main__':main()
