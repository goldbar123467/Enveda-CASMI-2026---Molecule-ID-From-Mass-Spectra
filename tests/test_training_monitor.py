"""Progress/health reporting must remain accurate across recovery and completion."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/training_monitor.py'
spec = importlib.util.spec_from_file_location('training_monitor', SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'outputs/local/long-example'
        self.run.mkdir(parents=True)
        self.now = datetime(2026, 11, 6, 19, tzinfo=timezone.utc)
        self.write('manifest.json', dict(spec=dict(run_id='L06', training=dict(epochs=128))))
        self.write('status.json', dict(state='training', family='external', fold=0, epoch=2,
                                       pid=42, updated_utc=self.now.isoformat()))

    def write(self, relative, value):
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def snapshot(self, alive=lambda _pid: True):
        return monitor.build_snapshot(self.run, now=self.now, alive=alive, root=self.root)

    def test_completed_receipts_and_partial_epoch_are_counted_once(self):
        self.write('models/control/fold-0/complete.json', dict(epochs=128, metrics=dict(mrr25=.7)))
        history = [dict(epoch=i, loss=2 / i, seconds=22) for i in (1, 2)]
        self.write('models/external/fold-0/history.json', history)
        data = self.snapshot()
        self.assertEqual(data['progress']['completed_epochs'], 130)
        self.assertEqual(data['progress']['total_epochs'], 1280)
        self.assertEqual(data['progress']['completed_models'], 1)
        self.assertEqual(data['progress']['seconds_per_epoch'], 22)
        self.assertEqual(len(data['models'][1]['series']), 2)
        self.assertGreater(monitor.parse_time(data['progress']['eta_utc']), self.now)

    def test_history_can_be_newer_than_status_during_atomic_writes(self):
        self.write('models/external/fold-0/history.json', [dict(epoch=i, seconds=20) for i in range(1, 4)])
        self.assertEqual(self.snapshot()['progress']['completed_epochs'], 3)

    def test_stale_or_dead_trainer_never_gets_a_moving_eta(self):
        self.write('models/external/fold-0/history.json', [dict(epoch=1, seconds=22)])
        self.assertEqual(self.snapshot(alive=lambda _pid: False)['state'], 'stopped')
        status = monitor.read_json(self.run / 'status.json')
        status['updated_utc'] = (self.now - timedelta(minutes=16)).isoformat()
        self.write('status.json', status)
        data = self.snapshot()
        self.assertEqual(data['state'], 'stale')
        self.assertIsNone(data['progress']['eta_utc'])

    def test_completion_does_not_depend_on_retired_process(self):
        self.write('status.json', dict(state='complete', pid=42, updated_utc=self.now.isoformat()))
        self.write('stack/report.json', dict(cohorts=dict(confirmation=dict(n=123))))
        data = self.snapshot(alive=lambda _pid: False)
        self.assertEqual(data['state'], 'complete')
        self.assertIsNone(data['progress']['eta_utc'])
        self.assertEqual(data['confirmation']['n'], 123)

    def test_watchdog_overdue_and_error_are_reported(self):
        self.write('watchdog/status.json', dict(action='healthy', pid=10,
                   next_check_utc=(self.now - timedelta(minutes=3)).isoformat()))
        self.assertTrue(self.snapshot()['watchdog']['overdue'])
        self.write('failure.json', dict(error='CUDA failure', utc=(self.now + timedelta(seconds=1)).isoformat()))
        self.assertEqual(self.snapshot()['state'], 'error')
        # An old failure does not keep a successfully resumed run in error.
        self.write('failure.json', dict(error='old failure', utc=(self.now - timedelta(minutes=1)).isoformat()))
        self.assertEqual(self.snapshot()['state'], 'training')

    def test_missing_or_temporarily_unreadable_files_are_safe(self):
        (self.run / 'status.json').write_text('{', encoding='utf-8')
        data = self.snapshot(alive=lambda _pid: None)
        self.assertEqual(data['state'], 'waiting')
        self.assertIsNone(data['progress']['eta_utc'])
        self.assertEqual(data['log'], [])
        json.dumps(data, allow_nan=False)

    def test_discovery_ignores_smoke_runs(self):
        smoke = self.root / 'outputs/local/long-example-smoke'
        smoke.mkdir()
        (smoke / 'manifest.json').write_text('{"smoke":true}', encoding='utf-8')
        self.assertEqual(monitor.discover_run(self.root), self.run)

    def test_frozen_preparation_is_not_counted_as_optimizer_work(self):
        self.write('manifest.json', dict(kind='frozen_retrieval', spec=dict(run_id='F07', seeds=[11, 12], training=dict(steps=50000))))
        self.write('status.json', dict(state='training', stage='encoding molecule_embeddings', pid=42,
                   stage_completed=30000, stage_total=273833, updated_utc=self.now.isoformat()))
        data = self.snapshot()
        self.assertEqual(len(data['models']), 2)
        self.assertEqual(data['progress']['unit'], 'steps')
        self.assertEqual(data['progress']['total_epochs'], 100000)
        self.assertEqual(data['progress']['completed_epochs'], 0)
        self.assertEqual(data['stage']['completed'], 30000)
        self.assertIsNone(data['progress']['eta_utc'])
        self.assertIn('30,000/273,833', monitor.terminal_view(data))

    def test_frozen_resume_progress_never_double_counts_pilot(self):
        self.write('manifest.json', dict(kind='frozen_retrieval', spec=dict(run_id='F07', seeds=[11, 12], training=dict(steps=50000))))
        self.write('models/seed-11/history.json', [dict(step=1000, seconds=20, pilot=True, loss=3),
                                                 dict(step=2000, seconds=30, pilot=False, loss=2)])
        self.write('status.json', dict(state='training', stage='full head training', seed=11, step=2100,
                   pid=42, updated_utc=self.now.isoformat()))
        data = self.snapshot()
        self.assertEqual(data['progress']['completed_epochs'], 2100)
        self.assertEqual(data['progress']['seconds_per_epoch'], .03)
        self.assertIsNone(data['progress']['eta_utc'])  # Uncached spectra remain.
        self.write('all_spectra_encoded.json', dict(encoded_rows=50000))
        self.assertIsNotNone(self.snapshot()['progress']['eta_utc'])
        self.assertIsNone(self.snapshot(alive=lambda _: False)['progress']['eta_utc'])

    def test_http_routes_never_serve_arbitrary_project_files(self):
        data = self.snapshot()
        class FixedMonitor:
            def snapshot(self):
                return data
        server = monitor.ThreadingHTTPServer(('127.0.0.1', 0), monitor.handler_for(FixedMonitor()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            with urlopen(base + '/api/status') as response:
                self.assertEqual(json.load(response)['run_id'], 'L06')
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
            with urlopen(base + '/') as response:
                self.assertIn(b'loss-chart', response.read())
            for path in ('/manifest.json', '/../../README.md', '/outputs/local/long-example/status.json'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + path)
                self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
