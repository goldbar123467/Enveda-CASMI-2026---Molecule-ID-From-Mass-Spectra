"""Build free CPU-only C09 raw spectrum overlap audit; no submission path."""
import json,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'kaggle/forks/c09-spectrum-audit';out.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'scripts/c09_spectrum_audit.py').read_text(encoding='utf-8')
    features=(ROOT/'outputs/local/kit/features.py').read_text(encoding='utf-8')
    preamble="from pathlib import Path\nimport sys\nsys.path.insert(0,'/kaggle/working')\nPath('/kaggle/working/features.py').write_text("+repr(features)+",encoding='utf-8')\n"
    code=preamble+source;compile(code,'c09_spectrum_audit','exec')
    (out/'audit.py').write_text(code,encoding='utf-8')
    meta=dict(id='clarkkitchen/casmi26-c09-spectrum-audit',title='casmi26 c09 spectrum audit',code_file='audit.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        competition_sources=['enveda-CASMI26-molecule-id-mass-spectra'],dataset_sources=['clarkkitchen/casmi26-c09-validation-20261109'],kernel_sources=[],model_sources=[])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    pin=dict(scoring_submission=False,device='CPU only',source_sha256=hashlib.sha256(code.encode()).hexdigest(),features_sha256=hashlib.sha256(features.encode()).hexdigest(),
        query_labels_used=False,source='scripts/c09_spectrum_audit.py')
    (out/'audit-pin.json').write_text(json.dumps(pin,indent=2),encoding='utf-8');print('BUILT C09 CPU SPECTRUM AUDIT',json.dumps(pin))


if __name__=='__main__':main()
