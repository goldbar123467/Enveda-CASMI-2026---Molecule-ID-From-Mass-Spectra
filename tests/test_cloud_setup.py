"""No model training: validate local rejection and cloud command construction."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_cloud", ROOT / "scripts/kaggle_cloud.py")
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


class CloudSetupTests(unittest.TestCase):
    def test_workload_refuses_local_execution(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "cloud/workload.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Local workload execution is prohibited", result.stderr)

    def test_cpu_and_gpu_launch_only_kaggle(self):
        for device in ("cpu", "gpu"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                with (
                    patch.object(cloud, "require_credentials"),
                    patch.object(cloud, "prepare", return_value=folder),
                    patch.object(cloud, "kaggle", return_value=0) as remote,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(cloud.main(["run", "--device", device]), 0)
                args = remote.call_args.args
                self.assertEqual(args[:2], ("kernels", "push"))
                self.assertIn("32400", args)
                self.assertEqual("--accelerator" in args, device == "gpu")
                if device == "gpu":
                    self.assertEqual(args[-1], "NvidiaTeslaT4")

    def test_submit_targets_named_kernel_version(self):
        with (
            patch.object(cloud, "kaggle", return_value=0) as remote,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            argv = ["submit", "--device", "cpu", "--version", "3", "--message", "v1", "--username", "example"]
            self.assertEqual(cloud.main(argv), 0)
        args = remote.call_args.args
        self.assertEqual(args[:3], ("competitions", "submit", cloud.read_config()["competition"]))
        self.assertIn("example/enveda-casmi-2026-cpu", args)
        self.assertEqual(args[args.index("-v") + 1], "3")
        self.assertEqual(args[args.index("-f") + 1], "submission.csv")

    def test_submit_rejects_non_numeric_version(self):
        with patch.object(cloud, "kaggle") as remote:
            with self.assertRaisesRegex(ValueError, "number"):
                cloud.main(["submit", "--device", "cpu", "--version", "latest", "--message", "x",
                            "--username", "example"])
            remote.assert_not_called()

    def test_prepare_does_not_call_kaggle(self):
        with (
            patch.object(cloud, "prepare", return_value=Path("prepared")),
            patch.object(cloud, "kaggle") as remote,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cloud.main(["prepare", "--device", "cpu", "--username", "example"]), 0)
            remote.assert_not_called()

    def test_offline_check_does_not_call_kaggle(self):
        with (
            patch.object(cloud, "kaggle") as remote,
            patch.object(cloud, "write_json"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cloud.check(online=False), 0)
            remote.assert_not_called()

    def test_credential_import_logs_only_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / "download.json"
            test_key = "synthetic-test-key-does-not-authenticate"
            source.write_text(json.dumps({"username": "example", "key": test_key}), encoding="utf-8")
            config_path = home / "project.json"
            config_path.write_text(json.dumps(cloud.read_config()), encoding="utf-8")
            captured = io.StringIO()
            with (
                patch.object(cloud, "ROOT", home / "repo"),
                patch.object(cloud, "CONFIG", config_path),
                patch.object(cloud.Path, "home", return_value=home),
                patch.dict(cloud.os.environ, {"KAGGLE_CONFIG_DIR": str(home / ".kaggle")}),
                patch.object(cloud, "restrict_permissions") as secure,
                contextlib.redirect_stdout(captured),
            ):
                self.assertEqual(cloud.configure_auth(source), 0)
                secure.assert_called_once()
                saved = home / ".kaggle/kaggle.json"
                self.assertEqual(json.loads(saved.read_text())['key'], test_key)
                report = (home / "repo/.local/credential-record.json").read_text()
                self.assertNotIn(test_key, report)
                self.assertNotIn(test_key, captured.getvalue())
                self.assertEqual(json.loads(config_path.read_text())["username"], "example")

    def test_different_existing_credential_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            source = home / "new.json"
            source.write_text(json.dumps({"username": "example", "key": "new-synthetic-key"}))
            destination = home / ".kaggle/kaggle.json"
            destination.parent.mkdir()
            original = json.dumps({"username": "example", "key": "old-synthetic-key"})
            destination.write_text(original)
            with patch.dict(cloud.os.environ, {"KAGGLE_CONFIG_DIR": str(destination.parent)}):
                with self.assertRaisesRegex(ValueError, "different credential"):
                    cloud.configure_auth(source)
            self.assertEqual(destination.read_text(), original)


if __name__ == "__main__":
    unittest.main()
