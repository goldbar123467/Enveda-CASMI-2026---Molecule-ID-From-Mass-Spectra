"""Exercise direct/analog guards and original-library cache isolation."""
import ast,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c41_reference_guard import guard_library,mask_engine2_rows,patch_engine2_source,install_library_guard


class Library:
    def __init__(self):
        self.struct_key=['a','b','c'];self.sid=np.asarray([0,1,1,2]);self._rep={'default':dict(idx=np.arange(4))}
    def window(self,target,ppm,exclude=None,mode=None):
        return np.flatnonzero(np.ones(4,bool) if exclude is None else ~exclude)
    def representatives(self,exclude=None,key='default'):
        if key not in self._rep:self._rep[key]=dict(idx=self.window(1.,10.,exclude=exclude))
        return self._rep[key]


class ReferenceGuardTests(unittest.TestCase):
    def test_both_direct_and_cached_analog_paths_remove_whole_structure(self):
        lib=guard_library(Library(),np.asarray([False,True,False]))
        np.testing.assert_equal(lib.window(1.,10.),[0,3])
        np.testing.assert_equal(lib.representatives()['idx'],[0,3])
        np.testing.assert_equal(lib.window(1.,10.,exclude=[False,False,False,True]),[0])
        self.assertGreater(lib._c41_guard_audit['window_calls'],0)

    def test_engine2_mask_keeps_other_rows_in_original_order(self):
        allowed,audit=mask_engine2_rows({'ik':['a','b','b','c']},[True,True,False,True],['b'])
        np.testing.assert_equal(np.flatnonzero(allowed),[0,3])
        self.assertEqual(audit['excluded_reference_rows'],1)
        self.assertEqual(audit['forbidden_rows_retained'],0)

    def test_changed_sid_inventory_and_changed_engine_source_fail_closed(self):
        with self.assertRaises(AssertionError):guard_library(Library(),[False,True])
        with self.assertRaises(AssertionError):patch_engine2_source('def build_index(L):\n    return []\n')

    def test_constructor_hook_applies_before_any_new_library_is_used(self):
        class PrivateLibrary(Library):pass
        install_library_guard(PrivateLibrary,[False,True,False])
        lib=PrivateLibrary();np.testing.assert_equal(lib.window(1.,10.),[0,3])
        np.testing.assert_equal(Library().window(1.,10.),[0,1,2,3])

    def test_engine2_identity_lifetime_spans_guard_index_construction(self):
        source='''def build_index(L):
    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)
    return ok
def main(L):
    del L["ik"], L["smi"]
    I = build_index(L)
    return I
'''
        patched=patch_engine2_source(source);tree=ast.parse(patched)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        call=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign))
        identity_release=[i for i,n in enumerate(main.body) if isinstance(n,ast.Delete) and 'ik' in ast.unparse(n)]
        self.assertEqual(identity_release,[call+1])
        # Execute the exact transformed main against an index that requires the
        # raw identity field, then verify the temporary strings are released.
        scope={'build_index':lambda library:list(library['ik'])}
        exec(compile(ast.Module(body=[main],type_ignores=[]),'lifetime','exec'),scope)
        library={'ik':['a','b'],'smi':['CCO','CCC']}
        self.assertEqual(scope['main'](library),['a','b']);self.assertEqual(library,{})


if __name__=='__main__':unittest.main()
