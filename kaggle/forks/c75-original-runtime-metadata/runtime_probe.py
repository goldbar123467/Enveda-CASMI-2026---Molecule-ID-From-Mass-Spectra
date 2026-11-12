import importlib.metadata as m,json,platform,sys
from pathlib import Path
packages=['torch','numpy','pandas','scipy','scikit-learn','lightgbm','numba','llvmlite','pyarrow','rdkit','nvidia-cudnn-cu12','nvidia-cublas-cu12','triton']
versions={}
for name in packages:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
result=dict(python=platform.python_version(),platform=platform.platform(),versions=versions,GPU_requested=False,query_data_mounted=False,quality_read=False,C13_read=False,scoring_entries=0)
Path('/kaggle/working/runtime-packages.json').write_text(json.dumps(result,indent=2))
print('C75 ORIGINAL IMAGE RUNTIME METADATA',json.dumps(result),flush=True)
