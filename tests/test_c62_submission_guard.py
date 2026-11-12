"""Exercise cap, source authorization and ambiguous-request recovery offline."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import submit_c62_pair as guard


class SubmissionGuardTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.out=self.root/'outputs/local/c62-two-new-submissions-20261112';self.out.mkdir(parents=True)
        (self.root/'.local').mkdir();self.ledger=self.root/'.local/c42-two-new-submissions-20261111.json'
        auth=self.root/'outputs/local/c38-source-support-20261111/authorization.json';self.save(auth,{'approved':2})
        smoke=self.root/'smoke.csv';smoke.write_text('frozen production output')
        audit=dict(execution_qualified=True,trial='cpu',remote_source_exact=True,local_cloud_final_lists_exact=True,
            independent_tail_arithmetic_exact=True,rank1_changes=0,candidate_membership_changes=0,formula_slot_changes=0,
            official_identity_duplicates=0,changed_real_preview_lists=9,real_preview_queries=12,valid_rows=400,
            csv_sha256=guard.sha(smoke),remote_scientific_sha256='remote-science')
        self.save(self.root/'audit.json',audit)
        self.q=dict(controller='C62',decision='execution_qualified_fixed_exploratory_pair',maximum_new_accepted_submissions=2,
            maximum_pending=1,authorization_sha256=guard.sha(auth),champion_retained=.480,order=['cpu','gpu'],claimed_public_improvement=False,
            artifact_sha256={},trials={'cpu':dict(kernel='user/cpu',version=1,audit_path='audit.json',smoke_output='smoke.csv')})
        self.save(self.out/'submission-qualification.json',self.q);self.qsha=guard.sha(self.out/'submission-qualification.json')
        self.active=dict(active=True,submissions_approved=2,submissions_used=0,submission_controller='C62',
            current_pair_qualification_sha256=self.qsha,scientific_gate_hold='Retired controllers remain held')
        self.save(self.root/'.local/c38-active-goal.json',self.active)
        self.api=SimpleNamespace(authenticate=lambda:None,competition_submissions=lambda *_:self.rows,
            competition_get_submission_limits=lambda *_:SimpleNamespace(num_allowed_now=5))
        self.rows=[];self.state=dict(controller='C62',maximum_new_accepted_submissions=2,submissions=[],intents=[])
        self.save(self.ledger,self.state)

    def save(self,path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8')

    def run_guard(self):
        with contextlib.ExitStack() as stack:
            for name,value in [('ROOT',self.root),('OUT',self.out),('LEDGER',self.ledger)]:stack.enter_context(patch.object(guard,name,value))
            stack.enter_context(patch('kaggle.api.kaggle_api_extended.KaggleApi',return_value=self.api))
            stack.enter_context(patch.object(guard,'register'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            guard.submit('cpu')

    def test_wrong_qualification_authorization_blocks_before_account_access(self):
        self.active['current_pair_qualification_sha256']='wrong';self.save(self.root/'.local/c38-active-goal.json',self.active)
        with patch.object(self.api,'authenticate') as auth:
            with self.assertRaises(AssertionError):self.run_guard()
            auth.assert_not_called()

    def test_shared_ledger_two_accepted_cap_blocks_third(self):
        self.state['submissions']=[dict(trial=t,qualification_sha256=self.qsha) for t in ['cpu','gpu']];self.save(self.ledger,self.state)
        with self.assertRaises(AssertionError):self.run_guard()

    def test_ambiguous_intent_never_retries_when_account_marker_missing(self):
        self.state['intents']=[dict(marker='C62-CPU-20261112',resolved=False)];self.save(self.ledger,self.state)
        with self.assertRaisesRegex(AssertionError,'Unresolved durable submission intent'):self.run_guard()

    def test_accepted_marker_reconciles_exact_intent_instead_of_repeating_api(self):
        self.state['intents']=[dict(marker='C62-CPU-20261112',resolved=False,qualification_sha256=self.qsha,
            remote_scientific_sha256='remote-science',kernel='user/cpu',version=1)];self.save(self.ledger,self.state)
        self.rows=[dict(ref=123,status='COMPLETE',public_score='.481',description='C62-CPU-20261112: frozen approved entry')]
        self.run_guard();ledger=json.loads(self.ledger.read_text())
        self.assertEqual(len(ledger['submissions']),1);self.assertEqual(ledger['submissions'][0]['reference'],'123')
        self.assertTrue(ledger['intents'][0]['resolved']);self.assertEqual(json.loads((self.root/'.local/c38-active-goal.json').read_text())['submissions_used'],1)

    def test_retired_controller_ledger_cannot_be_reused(self):
        self.state['controller']='C42';self.save(self.ledger,self.state)
        with self.assertRaisesRegex(AssertionError,'Shared ledger belongs to another pair'):self.run_guard()

    def test_no_actual_platform_allowance_blocks_release(self):
        # Quota is checked before remote execution preflight or a submit call.
        self.api.competition_get_submission_limits=lambda *_:SimpleNamespace(num_allowed_now=0)
        with self.assertRaisesRegex(AssertionError,'Actual Kaggle submission allowance exhausted'):self.run_guard()


if __name__=='__main__':unittest.main()
