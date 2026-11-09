import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import apply_external_reference


class RequiredStageTests(unittest.TestCase):
    def test_setup_failure_cannot_leave_baseline_scoring_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = b'molecule_id,smiles\na,C\n'
            (root / 'submission.csv').write_bytes(original)
            args = ['apply_external_reference.py', '--work', str(root), '--input-root', str(root),
                    '--contract', str(root / 'missing-contract.json')]
            with patch.object(sys, 'argv', args), self.assertRaises(FileNotFoundError):
                apply_external_reference.main()
            self.assertFalse((root / 'submission.csv').exists())
            self.assertEqual((root / 'c09e_p01_submission.csv').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
