import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_curriculum_anchor import ROOT,build,unwrap_notebook


class CurriculumCandidateTests(unittest.TestCase):
    def test_guarded_candidate_preserves_frozen_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'production/np/refit/casmi26_seed_23_e32_lr2e-05_embedding192-heads8-layers4-width256.pt'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'build-only fixture; no network inference')
            pin=json.loads((ROOT/'kaggle/forks/p08-general-anchor/model-pin.json').read_text())['sha256']
            manifest=dict(mode='np',stage='refit',smoke=False,epochs=32,parent_epoch=128,lr=2e-5,mixture_fraction=.5,
                          training_molecules=273833,heldout_identity_sha256=None,parent_sha256=pin)
            meta=checkpoint.parent.parent/'finetune_manifest.json';meta.write_text(json.dumps(manifest))
            destination=root/'candidate';build(checkpoint,destination)
            before=json.loads((ROOT/'kaggle/forks/p08-general-anchor/notebook.ipynb').read_text(encoding='utf-8'))
            after=json.loads((destination/'notebook.ipynb').read_text(encoding='utf-8'))
            unwrap_notebook(before);unwrap_notebook(after)
            def frozen(nb):
                return [ast.literal_eval(node.value) for c in nb['cells'] if c['cell_type']=='code'
                        for node in ast.parse(''.join(c['source'])).body if isinstance(node,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='FROZEN_B64' for t in node.targets)]
            self.assertEqual(len(frozen(before)),1)
            self.assertEqual(frozen(before),frozen(after))
            for cell in after['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'candidate_cell','exec')
            output_meta=json.loads((destination/'kernel-metadata.json').read_text())
            self.assertTrue(output_meta['is_private'])
            self.assertEqual(output_meta['kernel_sources'],['clarkkitchen/casmi26-p10-production-continuation'])
            manifest['smoke']=True;meta.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):build(checkpoint,root/'rejected')
            self.assertFalse((root/'rejected').exists())


if __name__=='__main__':
    unittest.main()
