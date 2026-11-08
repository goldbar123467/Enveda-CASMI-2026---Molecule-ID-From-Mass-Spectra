"""Kaggle client versions expose serialized JSON as well as enum/dict statuses."""
import sys,unittest
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finish_compatible_campaign import status_name


class CampaignStatusTests(unittest.TestCase):
    def test_actual_serialized_api_status_is_normalized(self):
        self.assertEqual(status_name('{"status":"running","failureMessage":null}'),'RUNNING')
        self.assertEqual(status_name('{"status":"complete","failureMessage":null}'),'COMPLETE')

    def test_enum_dict_and_failure_remain_distinct(self):
        self.assertEqual(status_name(SimpleNamespace(status='KernelWorkerStatus.RUNNING')),'RUNNING')
        self.assertEqual(status_name({'status':'KernelWorkerStatus.COMPLETE'}),'COMPLETE')
        self.assertEqual(status_name('KernelWorkerStatus.ERROR'),'ERROR')
        self.assertEqual(status_name('{"status":"error","failureMessage":"not COMPLETE"}'),'ERROR')


if __name__=='__main__':unittest.main()
