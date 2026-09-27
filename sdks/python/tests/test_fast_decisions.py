import unittest
from pathlib import Path
from l2s1 import DecisionResponse, DiscriminativeEvidence, LoadOptions
from l2s1.native import _arguments

class FastDecisionTests(unittest.TestCase):
    def test_real_laya_evidence(self):
        path=Path(__file__).resolve().parents[2]/"fixtures"/"discriminative_response.json"
        result=DecisionResponse.model_validate_json(path.read_text())
        for row in result.results:
            self.assertIsInstance(row.evidence,DiscriminativeEvidence)
            self.assertNotIn("candidate_mass",row.evidence.model_dump())
            self.assertNotIn("token_id",row.evidence.model_dump()["scores"][0])
    def test_native_fixed_schema_configuration(self):
        args=_arguments(LoadOptions(model="model.gguf",fixed_schema=True))
        self.assertIn("--fixed-schema",args)
        self.assertEqual(args[args.index("--execution-mode")+1],"prefix-reuse")
        with self.assertRaises(ValueError):
            _arguments(LoadOptions(model="model.gguf",fixed_schema=True,execution_mode="fresh"))
