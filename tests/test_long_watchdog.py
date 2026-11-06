"""The health supervisor must not duplicate jobs or touch unrelated processes."""
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('long_run_watchdog',ROOT/'scripts/long_run_watchdog.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,11,6,16,tzinfo=timezone.utc)
        self.deadline=self.now+timedelta(hours=8)
        self.status=dict(state='training',updated_utc=self.now.isoformat())
    def decision(self,status=None,processes=None,paused=False,deadline=None):
        return M.decide(self.status if status is None else status,[{}] if processes is None else processes,
                        self.now,900,paused,deadline or self.deadline)
    def test_live_fresh_job_stays_running(self):self.assertEqual(self.decision(),'healthy')
    def test_dead_job_can_resume(self):self.assertEqual(self.decision(processes=[]),'restart')
    def test_stalled_job_can_resume(self):
        self.status['updated_utc']=(self.now-timedelta(minutes=16)).isoformat()
        self.assertEqual(self.decision(),'restart_stalled')
    def test_complete_job_is_not_restarted(self):
        self.assertEqual(self.decision(dict(state='complete'),processes=[]),'complete')
    def test_explicit_pause_stops_supervisor(self):self.assertEqual(self.decision(paused=True),'paused')
    def test_budget_does_not_reset_on_restart(self):
        self.assertEqual(self.decision(deadline=self.now-timedelta(seconds=1)),'budget_reached')
    def test_only_project_wrapper_and_child_are_owned(self):
        rows=[dict(ProcessId=1,ParentProcessId=99,CommandLine=f'"{M.ROOT}\\.venv-gpu\\Scripts\\python.exe" -u scripts/long_training.py'),
              dict(ProcessId=2,ParentProcessId=1,CommandLine='C:\\Python312\\python.exe -u scripts/long_training.py'),
              dict(ProcessId=3,ParentProcessId=77,CommandLine='C:\\other\\.venv-gpu\\Scripts\\python.exe -u scripts/long_training.py'),
              dict(ProcessId=4,ParentProcessId=99,CommandLine=f'"{M.ROOT}\\.venv-gpu\\Scripts\\python.exe" -u scripts/long_training.py --smoke')]
        self.assertEqual({p['ProcessId'] for p in M.owned_trainers(rows)},{1,2})
    def test_repeated_deterministic_error_requires_review(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);(run/'train.err').write_text('Traceback: incompatible source',encoding='utf-8')
            state=dict(restarts=1,last_error_signature=hashlib.sha256(b'Traceback: incompatible source').hexdigest())
            settings=dict(restart_minimum_free_ram_gib=6,restart_minimum_free_vram_gib=1,maximum_restarts=3)
            with patch.object(M,'RUN',run),patch.object(M,'provenance_error',return_value=None),patch.object(M,'memory',return_value=(10,4)),patch.object(M.subprocess,'Popen') as launch:
                result=M.restart([],settings,state)
            self.assertEqual(result['action'],'needs_attention');launch.assert_not_called()


if __name__=='__main__':unittest.main()
