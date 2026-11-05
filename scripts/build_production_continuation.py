"""Package two full-data32-epoch continuations, gated by real full-parent GPU smoke."""
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
code="import base64,json,zlib,sys\nfrom pathlib import Path\nroot=Path('/kaggle/working/production_code');root.mkdir(parents=True,exist_ok=True)\n"
code+=f"payload=json.loads(zlib.decompress(base64.b64decode({blob!r})))\n"
code+="for name,source in payload.items():(root/name).write_text(source)\nsys.path.insert(0,str(root))\nfrom cloud_np_continuation import main\nmain(production=True)\n"
compile(code,'run.py','exec')
folder=ROOT/'kaggle/forks/p10-production-continuation';folder.mkdir(parents=True,exist_ok=True)
(folder/'run.py').write_text(code,encoding='utf-8')
meta=json.loads((ROOT/'kaggle/forks/p09-np-continuation/kernel-metadata.json').read_text())
meta.update(id='clarkkitchen/casmi26-p10-production-continuation',title='casmi26 p10 production continuation',
            kernel_sources=['clarkkitchen/casmi26-n0-scorer-prep','clarkkitchen/casmi26-p07-refit-v4-replica'])
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
(folder/'source-hashes.json').write_text(json.dumps({name:hashlib.sha256(source.encode()).hexdigest() for name,source in payload.items()},indent=2)+'\n')
print('PACKAGED, NOT LAUNCHED',folder)
