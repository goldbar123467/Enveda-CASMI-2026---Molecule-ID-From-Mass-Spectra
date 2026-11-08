"""Kaggle client versions expose serialized JSON as well as enum/dict statuses."""
import re,sys,unittest
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finish_compatible_campaign import status_name,VALIDATION_OUTPUTS,SMOKE_OUTPUTS


class CampaignStatusTests(unittest.TestCase):
    def test_export_filters_do_not_download_the_symlinked_training_file(self):
        self.assertIsNotNone(re.search(VALIDATION_OUTPUTS,'c08_validation_report.json'))
        self.assertIsNotNone(re.search(VALIDATION_OUTPUTS,'neural_candidate_features.parquet'))
        self.assertIsNone(re.search(VALIDATION_OUTPUTS,'c08_validation_comp/train.parquet'))
        self.assertIsNone(re.search(VALIDATION_OUTPUTS,'c08_validation_comp/test.parquet'))
        self.assertIsNotNone(re.search(SMOKE_OUTPUTS,'submission.csv.c08.json'))
        self.assertIsNone(re.search(SMOKE_OUTPUTS,'ours_bundle/final_model.pt'))

    def test_actual_serialized_api_status_is_normalized(self):
        self.assertEqual(status_name('{"status":"running","failureMessage":null}'),'RUNNING')
        self.assertEqual(status_name('{"status":"complete","failureMessage":null}'),'COMPLETE')

    def test_enum_dict_and_failure_remain_distinct(self):
        self.assertEqual(status_name(SimpleNamespace(status='KernelWorkerStatus.RUNNING')),'RUNNING')
        self.assertEqual(status_name({'status':'KernelWorkerStatus.COMPLETE'}),'COMPLETE')
        self.assertEqual(status_name('KernelWorkerStatus.ERROR'),'ERROR')
        self.assertEqual(status_name('{"status":"error","failureMessage":"not COMPLETE"}'),'ERROR')


if __name__=='__main__':unittest.main()
