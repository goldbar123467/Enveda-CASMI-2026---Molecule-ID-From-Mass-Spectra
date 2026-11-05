from pathlib import Path
import ast
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_general_anchor import ROOT,build,guarded_source


class SubmissionGuardTests(unittest.TestCase):
    def test_multiline_pins_and_global_state_are_preserved(self):
        source="PINNED = '''first\nsecond\n'''\nVALUE = 42\n"
        namespace={}
        exec(guarded_source(source,'test_cell'),namespace)
        self.assertEqual(namespace['PINNED'],'first\nsecond\n')
        self.assertEqual(namespace['VALUE'],42)

    def test_failure_removes_frozen_fallback_before_reraising(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'submission.csv'
            for expression,exception in [('RuntimeError("failed stage")',RuntimeError),('AssertionError("bad audit")',AssertionError),('SystemExit(1)',SystemExit)]:
                path.write_text('frozen baseline output')
                with self.assertRaises(exception):
                    exec(guarded_source('raise '+expression,'test_cell',path),{})
                self.assertFalse(path.exists())

    def test_candidate_build_preserves_frozen_bytes_and_pins_actual_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint=root/'general_full20/refit/casmi26_seed_20_e128_embedding192-heads8-layers4-width256.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'build-only fixture; no model inference')
            destination=root/'candidate'
            build(checkpoint,destination)
            original=json.loads((ROOT/'kaggle/forks/p01-public480/notebook.ipynb').read_text(encoding='utf-8'))
            generated=json.loads((destination/'notebook.ipynb').read_text(encoding='utf-8'))
            originals=[''.join(c['source']) for c in original['cells'] if c['cell_type']=='code']
            unwrapped=[]
            for c in generated['cells']:
                if c['cell_type']!='code':continue
                wrapper=ast.parse(''.join(c['source']))
                compile(wrapper,'generated_candidate','exec')
                unwrapped.append(wrapper.body[0].body[0].value.args[0].args[0].value)
            def frozen(sources):
                return [ast.literal_eval(node.value) for source in sources for node in ast.parse(source).body
                        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FROZEN_B64' for t in node.targets)]
            before,after=frozen(originals),frozen(unwrapped)
            self.assertEqual(len(before),1)
            self.assertEqual(before,after)
            pin=json.loads((destination/'model-pin.json').read_text())
            self.assertEqual(pin['sha256'],hashlib.sha256(checkpoint.read_bytes()).hexdigest())


if __name__=='__main__':
    unittest.main()
