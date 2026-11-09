import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import update_runs


class SeparateDiagnosticStatusTests(unittest.TestCase):
    def test_cancel_request_remains_nonterminal_until_acknowledged(self):
        row=dict(id='C09V',platform='kaggle-gpu',ref='c09-coverage-controls v1',status='queued',ended_utc='')
        with patch.object(update_runs,'kaggle_status',return_value='cancel_requested'):
            update_runs.refresh([row])
        self.assertEqual(row['status'],'cancel_requested');self.assertEqual(row['ended_utc'],'')
        with patch.object(update_runs,'kaggle_status',return_value='cancel_acknowledged'):
            update_runs.refresh([row])
        self.assertEqual(row['status'],'cancelled');self.assertTrue(row['ended_utc'])

    def test_training_completion_does_not_complete_a_running_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run';run.mkdir()
            (run/'manifest.json').write_text(json.dumps(dict(kind='compatible_scorer')))
            log=run/'development.log';log.write_text('Still computing fingerprints\n')
            row=dict(ref='run/development.log',kind='diag',status='running',expect='1')
            with patch.object(update_runs,'ROOT',root):
                state,_=update_runs.local_status(row);self.assertEqual(state,'running')
                log.write_text('DONE C09 DEVELOPMENT FREEZE {}\n')
                state,_=update_runs.local_status(row);self.assertEqual(state,'complete')


if __name__=='__main__':unittest.main()
