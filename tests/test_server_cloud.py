"""Synthetic identity/transport fixtures only; never import numerical libraries or fit models."""
import hashlib
import importlib.util
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
import workload
import server_receive


class ServerCloudTests(unittest.TestCase):
    def test_server_entry_refuses_local_execution_before_reading_authorization(self):
        for entry in ("server_run.py","server_fp_check.py","server_input_check.py","server_calibration.py"):
            with self.subTest(entry=entry):
                extra = ["--input","nonexistent-synthetic-input.json"] if entry=="server_calibration.py" else []
                result = subprocess.run([sys.executable, str(ROOT / "cloud" / entry),
                                         "--authorization", "nonexistent-synthetic-proof.json",
                                         "--output", "synthetic-output",*extra], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Local workload execution is prohibited", result.stderr)

    def test_wsl_refuses_even_an_authorization_path(self):
        with patch.object(workload.sys, "platform", "linux"), patch.object(workload.platform, "release", return_value="6.6-microsoft-standard-WSL2"):
            with self.assertRaisesRegex(RuntimeError, "Local workload execution is prohibited"):
                workload.require_server_cloud("nonexistent.json")

    def test_server_identity_must_match_and_gpu_is_not_authorized(self):
        proof = {"runtime": "vast-ssh", "hostname": "synthetic-server", "container_id": "fixture-42",
                 "machine_id_sha256": "synthetic-hash", "allowed_device": "cpu", "project_root": "/workspace/synthetic"}
        identity = {k: proof[k] for k in ("hostname", "container_id", "machine_id_sha256")}
        identity["vast_guide_present"] = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "authorization.json"
            path.write_text(json.dumps(proof))
            with patch.object(workload.sys, "platform", "linux"), patch.object(workload.platform, "release", return_value="6.8-generic"), patch.object(workload, "server_identity", return_value=identity):
                self.assertEqual(workload.require_server_cloud(path)["container_id"], "fixture-42")
                with self.assertRaisesRegex(RuntimeError, "not authorized"):
                    workload.require_server_cloud(path, "gpu")
                proof["allowed_devices"] = ["cpu", "gpu"]
                path.write_text(json.dumps(proof))
                self.assertEqual(workload.require_server_cloud(path, "gpu")["allowed_devices"], ["cpu", "gpu"])
                identity["container_id"] = "different-fixture"
                with self.assertRaisesRegex(RuntimeError, "not the SSH-verified"):
                    workload.require_server_cloud(path)

    def test_complete_frame_is_verified(self):
        payload = b"synthetic-input"
        footer = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        stream = io.BytesIO(struct.pack(">I", len(payload)) + payload + struct.pack(">I", 0) + (json.dumps(footer)+"\n").encode())
        output = io.BytesIO()
        self.assertEqual(server_receive.receive_frames(stream, output), footer)
        self.assertEqual(output.getvalue(), payload)

    def test_truncated_frame_cannot_be_published(self):
        with self.assertRaisesRegex(ValueError, "Incomplete transfer"):
            server_receive.receive_frames(io.BytesIO(struct.pack(">I", 8)+b"short"), io.BytesIO())

    def test_bad_checksum_cannot_be_published(self):
        stream = io.BytesIO(struct.pack(">I", 1)+b"x"+struct.pack(">I", 0)+b'{"bytes":1,"sha256":"wrong"}\n')
        with self.assertRaisesRegex(ValueError, "integrity"):
            server_receive.receive_frames(stream, io.BytesIO())

    def test_checkpoint_zip_is_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as folder:
            incoming, destination = Path(folder)/"incoming", Path(folder)/"model.pt"
            with zipfile.ZipFile(incoming, "w") as archive:
                archive.writestr("model/data.pkl", b"synthetic checkpoint")
            original = incoming.read_bytes()
            server_receive.publish_input(incoming, destination)
            self.assertEqual(destination.read_bytes(), original)

    def test_kaggle_wrapper_extracts_only_requested_member(self):
        with tempfile.TemporaryDirectory() as folder:
            incoming, destination = Path(folder)/"incoming", Path(folder)/"fixture.npy"
            with zipfile.ZipFile(incoming, "w") as archive:
                archive.writestr("fixture.npy", b"synthetic array")
                archive.writestr("../escape", b"ignored")
            server_receive.publish_input(incoming, destination)
            self.assertEqual(destination.read_bytes(), b"synthetic array")
            self.assertFalse((Path(folder).parent/"escape").exists())


if __name__ == "__main__":
    unittest.main()
