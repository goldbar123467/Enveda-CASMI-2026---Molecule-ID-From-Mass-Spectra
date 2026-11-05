"""Synthetic report fixtures only; no data/model processing."""
import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("phase1_gate",Path(__file__).resolve().parents[1]/"scripts/phase1_gate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.report = {"test":{"molecules":400,"fallback":0,"no_logits":0},"total_s":100,
                       "final_ranking":"learned","ranker":{"selected":{"objective":0.78}},
                       "validation":{"selected":{"objective":0.9}},"config":{"version":"fixture","pop_weight":0.15},"device":"cpu"}

    def test_valid_report_passes(self):
        self.assertTrue(module.check_report(self.report,0.79,"fixture","cpu",0.15)["passed"])

    def test_better_linear_metric_cannot_hide_bad_selected_ranker(self):
        self.report["ranker"]["selected"]["objective"] = 0.74
        self.assertFalse(module.check_report(self.report,0.79)["passed"])

    def test_partial_or_wrong_experiment_output_refuses_submission(self):
        for section,field,value in (("test","molecules",399),("test","fallback",1),("test","no_logits",1),
                                    ("config","version","different"),("config","pop_weight",0)):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.report)
                bad[section][field] = value
                self.assertFalse(module.check_report(bad,0.79,"fixture","cpu",0.15)["passed"])
        self.report["total_s"] = 10800
        self.assertFalse(module.check_report(self.report,0.79)["passed"])


if __name__=="__main__":
    unittest.main()
