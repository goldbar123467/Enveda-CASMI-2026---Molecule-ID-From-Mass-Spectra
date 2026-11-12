"""Failed scientific gates must prevent reserved-label or account mutations."""
import contextlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continue_c42_approved_pair as continuation


class ApprovedContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.paths={name:self.root/name for name in ['OUT','C39','C43','FULL']}
        for path in self.paths.values():path.mkdir()
        (self.root/'scripts').mkdir();self.audit=self.root/'scripts/audit_c42_quality.py';self.audit.write_text('synthetic audit source')
        self.called=[];self.confirmation_pass=True
        self.quality=dict(queries=352,all_queries_retained=True,missing_truths_score_zero=True,C13_read=False,
            results={t:dict(admission=dict(passed=True,reasons=[])) for t in ['cpu','gpu']})
        self.independent=dict(passed_pair=True,results={t:dict(passed=True) for t in ['cpu','gpu']})
        self.save(self.paths['C39']/'independent-ranking-audit.json',{})

    def save(self,path,value):path.write_text(json.dumps(value),encoding='utf8')

    def prepare_quality(self):
        self.save(self.paths['C39']/'quality-results.json',self.quality)
        self.independent.update(quality_results_sha256=continuation.sha(self.paths['C39']/'quality-results.json'),source_sha256=continuation.sha(self.audit))
        self.save(self.paths['C39']/'independent-quality-audit.json',self.independent)

    def run_step(self,name,environment,script,*args):
        self.called.append((script,tuple(map(str,args))))
        if script=='confirm_c43_pair.py' and args[0]=='evaluate':
            self.save(self.paths['C43']/'confirmation-results.json',dict(results={t:dict(admission=dict(passed=self.confirmation_pass,reasons=[])) for t in ['cpu','gpu']}))
        if script=='watch_c42_scores.py':
            self.save(self.paths['OUT']/'actual-final-results.json',dict(actual_scored_entries=2,no_third_entry=True,observer_stopped=True,
                entries={t:dict(status='COMPLETE',success=True,public_score=.481) for t in ['cpu','gpu']}))

    def advance(self):
        self.prepare_quality()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(continuation,'ROOT',self.root))
            for name,path in self.paths.items():stack.enter_context(patch.object(continuation,name,path))
            continuation.advance_pair(lambda:None,self.run_step,lambda *a,**k:None)

    def test_development_or_source_gate_failure_leaves_C13_and_account_untouched(self):
        self.quality['results']['gpu']['admission']['passed']=False
        self.advance();self.assertEqual(self.called,[])
        self.assertEqual(json.loads((self.paths['OUT']/'approved-pair-no-go.json').read_text())['scoring_entries'],0)

    def test_independent_exact_gate_failure_cannot_be_overruled_by_float_result(self):
        self.independent['passed_pair']=False;self.independent['results']['cpu']['passed']=False
        self.advance();self.assertEqual(self.called,[])

    def test_missing_truth_zeros_requirement_blocks_confirmation(self):
        self.quality['missing_truths_score_zero']=False
        with self.assertRaises(AssertionError):self.advance()
        self.assertEqual(self.called,[])

    def test_confirmation_failure_never_qualifies_or_submits(self):
        self.confirmation_pass=False;self.advance()
        self.assertEqual(self.called,[('confirm_c43_pair.py',('prepare',)),('confirm_c43_pair.py',('score',)),('confirm_c43_pair.py',('evaluate','--consume'))])
        self.assertFalse((self.paths['OUT']/'approved-pair-continuation-complete.json').exists())

    def test_only_qualified_pair_uses_cpu_first_then_actual_score_observer(self):
        self.advance()
        self.assertEqual(self.called,[('confirm_c43_pair.py',('prepare',)),('confirm_c43_pair.py',('score',)),('confirm_c43_pair.py',('evaluate','--consume')),
            ('qualify_c42_pair.py',()),('submit_c42_pair.py',('cpu','--dry-run')),('submit_c42_pair.py',('cpu',)),('watch_c42_scores.py',('--release-second',))])
        final=json.loads((self.paths['OUT']/'approved-pair-continuation-complete.json').read_text())
        self.assertFalse(final['native_goal_marked_complete']);self.assertEqual(final['actual_scored_entries'],2)


if __name__=='__main__':unittest.main()
