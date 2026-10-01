import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location("fit",Path(__file__).resolve().parents[1]/"scripts"/"calibrate_cascade.py")
fit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)
class CalibrationTests(unittest.TestCase):
    def test_validation_does_not_retune_threshold(self):
        self.assertEqual(fit.fit({"s":[(.9,False,True),(.8,True,True)]},{"s":[(.95,True,True),(.85,False,True)]},0),[])
    def test_confidence_ties_are_not_split(self):
        self.assertEqual(fit.fit({"s":[(.9,False,True),(.9,True,True)]},{"s":[(.9,False,True)]},0),[])
    def test_accepted_only_and_heldout_counts(self):
        rules=fit.fit({"s":[(.9,False,True),(.95,False,False)]},{"s":[(.91,False,True),(.99,False,False)]},0)
        self.assertEqual(rules[0]["accepted"],1)
        self.assertEqual(rules[0]["validation_items"],2)
