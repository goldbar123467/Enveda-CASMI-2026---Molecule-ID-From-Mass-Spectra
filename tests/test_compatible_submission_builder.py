"""Compile the future notebook with synthetic metadata, never executing/uploading it."""
import contextlib,hashlib,io,json,shutil,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_compatible_submission as builder


class CompatibleSubmissionBuilderTests(unittest.TestCase):
    def test_complete_p01_is_retained_and_guarded_tail_is_last(self):
        real=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent=root/'kaggle/forks/p01-public480';parent.mkdir(parents=True)
            for name in ['notebook.ipynb','kernel-metadata.json']:shutil.copy2(real/'kaggle/forks/p01-public480'/name,parent/name)
            bundle=root/'outputs/publish/c08-models-20261108';bundle.mkdir(parents=True)
            manifest=bundle/'c08-manifest.json';manifest.write_text(json.dumps(dict(smoke=False,steps=40000,models=[{}]*8)))
            (bundle/'selection.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                explicit_user_trial=True,winner=dict(arm='precise32-hard',weight=.1))))
            with patch.object(builder,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):builder.main()
            folder=root/'kaggle/forks/p15-compatible-c08';notebook=json.loads((folder/'notebook.ipynb').read_text(encoding='utf-8'))
            original=json.loads((parent/'notebook.ipynb').read_text(encoding='utf-8'))
            self.assertEqual(len(notebook['cells']),len(original['cells'])+1)
            for cell in notebook['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'generated','exec')
            tail=''.join(notebook['cells'][-1]['source'])
            self.assertIn('c08_p01_submission.csv',tail);self.assertIn('len(SAMPLE_IDS)',tail)
            self.assertIn('candidate_set_changes',tail);self.assertIn('unlink(missing_ok=True)',tail)
            meta=json.loads((folder/'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private']);self.assertEqual(meta['id'],'clarkkitchen/casmi26-p15-compatible-c08')


if __name__=='__main__':unittest.main()
