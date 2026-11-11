"""Validation-only reference guards; never inject or rank query identities."""
import json,hashlib,types
from pathlib import Path
import numpy as np


def mask_engine2_rows(library,base_ok,blocked_keys):
    base_ok=np.asarray(base_ok,bool);keys=library['ik'];assert len(keys)==len(base_ok)
    blocked=set(map(str,blocked_keys))
    excluded=np.fromiter((str(k) in blocked for k in keys),dtype=bool,count=len(keys))
    answer=base_ok&~excluded
    assert not np.any(answer&excluded)
    return answer,dict(reference_rows=len(keys),excluded_reference_rows=int((base_ok&excluded).sum()),
                       retained_reference_rows=int(answer.sum()),forbidden_rows_retained=0)


def guard_library(library,sid_mask):
    sid_mask=np.asarray(sid_mask,bool)
    assert sid_mask.ndim==1 and len(library.struct_key)==len(sid_mask),'Reference SID inventory differs'
    if hasattr(library,'_c41_guard_audit'):
        assert library._c41_guard_audit['sid_mask_sha256']==hashlib.sha256(sid_mask.tobytes()).hexdigest()
        return library
    sid=np.asarray(library.sid,np.int64)
    assert np.all(sid<len(sid_mask)),'Unknown extended reference SIDs'
    excluded=sid<0;excluded[~excluded]=sid_mask[sid[~excluded]]
    audit=dict(reference_rows=len(sid),excluded_reference_rows=int(excluded.sum()),window_calls=0,representative_calls=0,
               forbidden_rows_retained=0,sid_mask_sha256=hashlib.sha256(sid_mask.tobytes()).hexdigest())
    original_window=library.window;original_reps=library.representatives
    library._rep.clear()  # Never reuse representatives chosen before masking.
    def combined(other):
        if other is None:return excluded
        other=np.asarray(other,bool);assert other.shape==excluded.shape
        return excluded|other
    def window(self,target,ppm,exclude=None,mode=None):
        result=original_window(target,ppm,exclude=combined(exclude),mode=mode)
        assert not excluded[result].any();audit['window_calls']+=1
        return result
    def representatives(self,exclude=None,key='default'):
        result=original_reps(exclude=combined(exclude),key=key)
        assert not excluded[np.asarray(result['idx'],np.int64)].any();audit['representative_calls']+=1
        return result
    library.window=types.MethodType(window,library)
    library.representatives=types.MethodType(representatives,library)
    library._c41_guard_audit=audit
    return library


def install_library_guard(cls,sid_mask):
    original=cls.__init__
    assert not getattr(cls,'_c41_constructor_guard',False),'Already installed'
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs);guard_library(self,sid_mask)
    cls.__init__=initialize;cls._c41_constructor_guard=True


def patch_engine2_source(source):
    needle='    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)'
    assert source.count(needle)==1 and 'CASMI_C41_EXCLUSION_FILE' not in source
    replacement=needle+'''
    import os as _c41_os
    _c41_path = _c41_os.environ.get('CASMI_C41_EXCLUSION_FILE')
    assert _c41_path, 'C41 requires the global validation reference mask'
    from c41_reference_guard import mask_engine2_rows as _c41_mask_rows
    import json as _c41_json
    from pathlib import Path as _c41_Path
    _c41_spec = _c41_json.loads(_c41_Path(_c41_path).read_text())
    ok, _c41_audit = _c41_mask_rows(L, ok, _c41_spec['keys'])
    _c41_Path('/kaggle/working/c41_engine2_reference_mask_audit.json').write_text(_c41_json.dumps(_c41_audit, indent=2))
'''
    result=source.replace(needle,replacement)
    # The pinned engine releases raw identity strings immediately before index
    # construction. Keep them until the guard consumes them, then release them.
    release='    del L["ik"], L["smi"]'
    index='    I = build_index(L)'
    assert result.count(release)==1 and result.count(index)==1
    result=result.replace(release,'    del L["smi"]').replace(index,index+'\n    del L["ik"]')
    compile(result,'c41_masked_engine2.py','exec')
    return result


def patch_legacy_tree(source,destination):
    """Copy code only, preserving mounted assets and model vocabulary."""
    import shutil
    from legacy_formate_compat import patch_chem_source,patch_frag_source
    source,destination=Path(source),Path(destination)
    expected={'chem.py':'99474f23d8a67040b0c0f8c70e73592a6f7f656f50103e350ca473451e2169f1',
              'frag.py':'809f5582494cee4cf6dcace4b04699b8427e81022b78d42162cc0239a57773de'}
    before={n:hashlib.sha256((source/'casmi'/n).read_bytes()).hexdigest() for n in expected}
    assert before==expected,'Legacy scalar source differs from the completed-run receipts'
    assert not destination.exists(),'Fresh compatibility code directory required'
    shutil.copytree(source,destination)
    for name,patch in [('chem.py',patch_chem_source),('frag.py',patch_frag_source)]:
        p=destination/'casmi'/name;p.write_text(patch(p.read_text(encoding='utf-8')),encoding='utf-8')
    after={n:hashlib.sha256((destination/'casmi'/n).read_bytes()).hexdigest() for n in expected}
    return str(destination),dict(source_root=str(source),work_root=str(destination),before_sha256=before,after_sha256=after)
