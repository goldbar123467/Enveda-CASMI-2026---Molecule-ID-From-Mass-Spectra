"""An incomplete training run or retuned confirmation must never be reported as verified."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "finalize_science", Path(__file__).resolve().parents[1] / "scripts/finalize_science.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "inputs"
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps(dict(epochs=32, steps_per_epoch=500, seeds=[1, 2])))
        (source / "result.json").write_text(json.dumps(dict(submit_allowed=False)))
        (source / "np_cached_sweep.json").write_text("{}")
        for seed in [1, 2]:
            folder = source / f"round-{seed}"
            folder.mkdir()
            history = [dict(epoch=i, steps=500, seconds=1, retrieval_mrr25=.5) for i in range(1, 33)]
            (folder / "history.json").write_text(json.dumps(history))
            (folder / "holdout.pt").write_bytes(b"synthetic-checkpoint")
            (folder / "panel_report.json").write_text(json.dumps(dict(
                molecules=[dict(key=str(i)) for i in range(120)], development=dict(n=58),
                confirmation=dict(n=62), selected=dict(alpha=.25))))
        return source

    def test_partial_epoch_is_rejected_before_evidence_is_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            history_path = source / "round-2/history.json"
            history = json.loads(history_path.read_text())
            history[-1]["steps"] = 499
            history_path.write_text(json.dumps(history))
            with patch.object(M, "SOURCE", source), patch.object(M, "DEST", root / "evidence"):
                with self.assertRaisesRegex(AssertionError, "Partial training"):
                    M.main()
            self.assertFalse((root / "evidence").exists())

    def test_retuning_on_second_round_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            p = source / "round-2/panel_report.json"
            panel = json.loads(p.read_text())
            panel["selected"]["alpha"] = .5
            p.write_text(json.dumps(panel))
            with patch.object(M, "SOURCE", source), patch.object(M, "DEST", root / "evidence"):
                with self.assertRaisesRegex(AssertionError, "tuning is forbidden"):
                    M.main()
            self.assertFalse((root / "evidence").exists())

    def test_compact_evidence_excludes_molecule_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            dest = root / "evidence"
            with patch.object(M, "SOURCE", source), patch.object(M, "DEST", dest):
                with contextlib.redirect_stdout(io.StringIO()):
                    M.main()
            panel = json.loads((dest / "round-1.json").read_text())["panel"]
            self.assertNotIn("molecules", panel)
            self.assertFalse(list(dest.glob("*.pt")))


if __name__ == "__main__":
    unittest.main()
