"""Package the current fixed-epoch full refit and v4 architecture replication."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
payload={name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in ['local_seeds.py','cloud_refit_replica.py','trial_worker.py']}
for name,source in payload.items():
    compile(source,name,'exec')
blob=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
code="import base64,json,zlib,sys\nfrom pathlib import Path\nroot=Path('/kaggle/working/refit_code');root.mkdir(parents=True,exist_ok=True)\n"
code+=f"payload=json.loads(zlib.decompress(base64.b64decode({blob!r})))\n"
code+="for name,source in payload.items():(root/name).write_text(source)\nsys.path.insert(0,str(root))\nfrom cloud_refit_replica import main\nmain()\n"
compile(code,'run.py','exec')
folder=ROOT/'kaggle/forks/p07-refit-v4-replica';folder.mkdir(parents=True,exist_ok=True)
(folder/'run.py').write_text(code,encoding='utf-8')
meta=json.loads((ROOT/'kaggle/forks/p05-paired-native/kernel-metadata.json').read_text())
meta.update(id='clarkkitchen/casmi26-p07-refit-v4-replica',title='casmi26 p07 refit v4 replica')
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(folder/'source-hashes.json').write_text(json.dumps({name:hashlib.sha256(source.encode()).hexdigest() for name,source in payload.items()},indent=2)+'\n')
print('PACKAGED',folder)
