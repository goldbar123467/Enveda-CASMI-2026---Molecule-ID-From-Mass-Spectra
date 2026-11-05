import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/update_runs.py'
SPEC = importlib.util.spec_from_file_location('pause_guard_update_runs', SCRIPT)
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


class PauseGuardTests(unittest.TestCase):
    def test_paused_background_start_cannot_refresh_or_sleep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.local').mkdir()
            (root / '.local/compute-paused.json').write_text('{"paused": true}')
            with patch.object(updater, 'ROOT', root), patch.object(updater, 'once') as refresh, patch.object(updater.time, 'sleep') as sleep:
                updater.main(loop=True)
                refresh.assert_not_called()
                sleep.assert_not_called()

    def test_pause_allows_one_explicit_manual_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.local').mkdir()
            (root / '.local/compute-paused.json').write_text('{"paused": true}')
            with patch.object(updater, 'ROOT', root), patch.object(updater, 'once') as refresh, patch.object(updater.time, 'sleep') as sleep:
                updater.main(loop=False)
                refresh.assert_called_once_with()
                sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
