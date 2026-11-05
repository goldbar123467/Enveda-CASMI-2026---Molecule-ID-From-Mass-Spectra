"""Package the independent paired T4 experiment; no training is executed locally."""
import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
names = ['local_seeds.py', 'native_training.py', 'cloud_native_pair.py']
payload = {name: (ROOT / 'scripts' / name).read_text(encoding='utf-8') for name in names}
blob = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
code = "import base64,json,zlib,sys\nfrom pathlib import Path\nroot=Path('/kaggle/working/pair_code');root.mkdir(parents=True,exist_ok=True)\n"
code += f"payload=json.loads(zlib.decompress(base64.b64decode({blob!r})))\n"
code += "for name,text in payload.items():(root/name).write_text(text)\nsys.path.insert(0,str(root))\nfrom cloud_native_pair import main\nmain()\n"
folder = ROOT / 'kaggle/forks/p05-paired-native'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'run.py').write_text(code, encoding='utf-8')
meta = dict(id='clarkkitchen/casmi26-p05-paired-native', title='casmi26 p05 paired native', code_file='run.py',
            language='python', kernel_type='script', is_private=True, enable_gpu=True, enable_internet=False,
            dataset_sources=['sanpier/casmi-spectrum-model-v1'],
            kernel_sources=['clarkkitchen/casmi26-n0-scorer-prep', 'clarkkitchen/casmi26-p04-native-unseen'],
            competition_sources=['enveda-CASMI26-molecule-id-mass-spectra'],
            docker_image='gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461')
(folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
(folder / 'source-hashes.json').write_text(json.dumps({name: hashlib.sha256(text.encode()).hexdigest() for name, text in payload.items()}, indent=2) + '\n')
print('PACKAGED', folder)
