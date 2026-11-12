"""Exercise actual-score sequencing in a temporary account/filesystem simulation."""
import contextlib,csv,io,json,sys,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import watch_c42_scores as observer
import submit_c42_pair


class ScoreObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/'outputs';self.out.mkdir()
        (self.root/'.local').mkdir();self.ledger=self.root/'.local/ledger.json'
        source=Path(observer.__file__);self.source=self.root/'scripts'/source.name
        self.source.parent.mkdir();self.source.write_bytes(source.read_bytes())
        q=dict(decision='execution_qualified_fixed_exploratory_pair',order=['cpu','gpu'],maximum_new_accepted_submissions=2,
            artifact_sha256={'scripts\\watch_c42_scores.py':observer.sha(self.source)})
        self.save(self.out/'submission-qualification.json',q)
        self.save(self.root/'.local/c38-active-goal.json',dict(active=True,submissions_approved=2,submissions_used=1))
        qsha=observer.sha(self.out/'submission-qualification.json')
        self.save(self.ledger,dict(submissions=[dict(trial='cpu',reference='101',status='ACCEPTED',qualification_sha256=qsha)],intents=[]))
        with (Path(__file__).resolve().parents[1]/'runs.csv').open(encoding='utf8',newline='') as stream:
            self.run_fields=next(csv.reader(stream))
        with (self.root/'runs.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=self.run_fields);writer.writeheader()
            writer.writerow(dict(platform='kaggle-submission',ref='101'))
        self.api_calls=0;self.release_calls=[]

    def save(self,path,value):path.write_text(json.dumps(value),encoding='utf8')

    def row(self,ref,status,score=None,error=None):
        return SimpleNamespace(ref=ref,status=status,public_score=score,error_description=error,description='',date=datetime.now(timezone.utc))

    def release(self,trial):
        self.assertEqual(trial,'gpu');self.assertGreaterEqual(self.api_calls,2);self.release_calls.append(trial)
        ledger=json.loads(self.ledger.read_text());self.assertEqual(len(ledger['submissions']),1)
        record=dict(trial='gpu',reference='102',status='ACCEPTED',qualification_sha256=ledger['submissions'][0]['qualification_sha256'])
        ledger['submissions'].append(record);self.save(self.ledger,ledger)
        with (self.root/'runs.csv').open('a',newline='') as stream:
            csv.DictWriter(stream,fieldnames=self.run_fields).writerow(dict(platform='kaggle-submission',ref='102'))

    def run_monitor(self,states):
        def history(*args):
            self.api_calls+=1
            if not states:raise AssertionError('Unexpected polling; simulation exhausted')
            state=states.pop(0)
            if isinstance(state,Exception):raise state
            return state
        api=SimpleNamespace(authenticate=lambda:None,competition_submissions=history)
        with contextlib.ExitStack() as stack:
            for name,value in [('ROOT',self.root),('OUT',self.out),('LEDGER',self.ledger),('__file__',str(self.source))]:
                stack.enter_context(patch.object(observer,name,value))
            stack.enter_context(patch('kaggle.api.kaggle_api_extended.KaggleApi',return_value=api))
            stack.enter_context(patch.object(submit_c42_pair,'submit',side_effect=self.release))
            stack.enter_context(patch.object(observer.time,'sleep',return_value=None))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            observer.main(True)

    def final(self):return json.loads((self.out/'actual-final-results.json').read_text())

    def test_waits_for_numeric_first_score_then_submits_once_and_stops_after_two(self):
        self.run_monitor([[self.row(101,'COMPLETE')],
            [self.row(101,'COMPLETE','.481')],
            [self.row(101,'COMPLETE','.481'),self.row(102,'COMPLETE','.479')]])
        self.assertEqual(self.release_calls,['gpu']);self.assertEqual(self.api_calls,3)
        self.assertEqual(self.final()['actual_scored_entries'],2)
        self.assertEqual(self.final()['best_public_score_including_preserved_champion'],.481)
        self.assertEqual(len(json.loads(self.ledger.read_text())['submissions']),2)

    def test_observation_failure_does_not_infer_terminal_or_retry_submission(self):
        self.run_monitor([ConnectionError('offline'),[self.row(101,'COMPLETE','.478')],
            [self.row(101,'COMPLETE','.478'),self.row(102,'COMPLETE','.477')]])
        self.assertEqual(self.release_calls,['gpu']);self.assertEqual(self.api_calls,3)
        self.assertEqual(self.final()['best_public_score_including_preserved_champion'],.480)

    def test_actual_error_stops_without_releasing_second_or_inventing_score(self):
        with self.assertRaisesRegex(RuntimeError,'actual scoring ended in error'):
            self.run_monitor([[self.row(101,'ERROR',error='runtime failure')]])
        self.assertEqual(self.release_calls,[])
        self.assertFalse((self.out/'actual-final-results.json').exists())
        self.assertTrue((self.out/'actual-scoring-error.json').exists())

    def test_qualification_missing_blocks_every_account_call(self):
        (self.out/'submission-qualification.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_monitor([])
        self.assertEqual(self.api_calls,0);self.assertEqual(self.release_calls,[])

    def test_other_pending_entry_defers_second_release(self):
        self.run_monitor([[self.row(101,'COMPLETE','.481'),self.row(999,'PENDING')],
            [self.row(101,'COMPLETE','.481'),self.row(999,'COMPLETE','.1')],
            [self.row(101,'COMPLETE','.481'),self.row(102,'COMPLETE','.482')]])
        self.assertEqual(self.release_calls,['gpu']);self.assertEqual(self.api_calls,3)


if __name__=='__main__':unittest.main()
