"""Synthetic structure identifiers only; no model training or numerical libraries."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("candidate_workload", ROOT / "cloud/workload.py")
WORKLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKLOAD)


class CandidatePoolTests(unittest.TestCase):
    def test_external_pool_skips_existing_and_repeated_structures(self):
        known = {"coconut-a", "coconut-b"}
        keys = ["coconut-b", "bio-a", "bio-a", "bio-b", "coconut-a"]
        self.assertEqual(WORKLOAD.unseen_candidate_rows(keys, known), [1, 3])
        # Training structures must also see the structures appended from the bio source.
        self.assertEqual(known, {"coconut-a", "coconut-b", "bio-a", "bio-b"})
        self.assertEqual(WORKLOAD.unseen_candidate_rows(["bio-b", "train-a"], known), [1])

    def test_empty_external_pool_preserves_existing_keys(self):
        known = {"coconut-a"}
        self.assertEqual(WORKLOAD.unseen_candidate_rows([], known), [])
        self.assertEqual(known, {"coconut-a"})


if __name__ == "__main__":
    unittest.main()
