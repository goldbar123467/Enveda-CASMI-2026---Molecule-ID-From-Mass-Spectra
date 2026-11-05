"""Prepare the fixed two-condition NP continuation; launch only after P05 completes."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
names=['local_seeds.py','np_sampling.py','finetune_sampling.py','cloud_np_continuation.py']
payload={name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in names}
for name,source in payload.items():compile(source,name,'exec')
blob=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
code="import base64,json,zlib,sys\nfrom pathlib import Path\nroot=Path('/kaggle/working/continuation_code');root.mkdir(parents=True,exist_ok=True)\n"
code+=f"payload=json.loads(zlib.decompress(base64.b64decode({blob!r})))\n"
code+="for name,source in payload.items():(root/name).write_text(source)\nsys.path.insert(0,str(root))\nfrom cloud_np_continuation import main\nmain()\n"
compile(code,'run.py','exec')
folder=ROOT/'kaggle/forks/p09-np-continuation';folder.mkdir(parents=True,exist_ok=True)
(folder/'run.py').write_text(code,encoding='utf-8')
meta=json.loads((ROOT/'kaggle/forks/p05-paired-native/kernel-metadata.json').read_text())
meta.update(id='clarkkitchen/casmi26-p09-np-continuation',title='casmi26 p09 NP continuation',
            dataset_sources=['sanpier/casmi-spectrum-model-v1','dmitriigluzdov/casmi26-pubchem-popularity-prior','qwertykeypadapple/casmi26-lotus-candidates'],
            kernel_sources=['clarkkitchen/casmi26-n0-scorer-prep','clarkkitchen/casmi26-p05-paired-native'])
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(folder/'source-hashes.json').write_text(json.dumps({name:hashlib.sha256(source.encode()).hexdigest() for name,source in payload.items()},indent=2)+'\n')
print('PACKAGED, NOT LAUNCHED',folder)
