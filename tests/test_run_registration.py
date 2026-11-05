import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import add_run


class RunRegistrationTests(unittest.TestCase):
    def test_new_kernel_version_does_not_inherit_old_terminal_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fields=['id','platform','kind','ref','started_utc','ended_utc','status','expect','purpose','result']
            with (root/'runs.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                writer.writerow(dict(id='old',platform='kaggle-gpu',ref='job v1',status='error'))
            with patch.object(add_run,'ROOT',root),patch.object(sys,'argv',['add_run.py','retry','kaggle-gpu','train','job v2','Retry']),patch('update_runs.render') as render,patch('update_runs.refresh') as refresh:
                add_run.main()
                refresh.assert_not_called()
                render.assert_called_once()
            with (root/'runs.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['status'],'error')
            self.assertEqual(rows[1]['status'],'queued')
            self.assertEqual(rows[1]['ended_utc'],'')


if __name__=='__main__':
    unittest.main()
