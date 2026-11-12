import importlib.metadata as m,json,platform
from pathlib import Path
versions={d.metadata['Name']:d.version for d in m.distributions() if d.metadata.get('Name')}
result=dict(python=platform.python_version(),versions=versions,GPU_requested=False,query_data_mounted=False,quality_read=False,C13_read=False,scoring_entries=0)
Path('/kaggle/working/runtime-inventory.json').write_text(json.dumps(result,indent=2,sort_keys=True))
print('C75 CPU ORIGINAL IMAGE INVENTORY',len(versions),'packages',flush=True)
