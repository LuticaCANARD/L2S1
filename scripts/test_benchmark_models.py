"""Report/preflight tests; no checkpoints or native runtime required."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_models import read_models, save_summary


class BenchmarkRunnerTests(unittest.TestCase):
    def test_manifest_rejects_path_traversal_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            for ids in [["../escape"], ["same", "same"]]:
                path.write_text(json.dumps({"models": [{"id": name, "path": "unused.gguf"} for name in ids]}))
                with self.assertRaises(ValueError):
                    read_models(path, None)

    def test_selection_rejects_unknown_models_and_preserves_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            model = Path(directory) / "custom.gguf"
            path.write_text(json.dumps({"models": [{"id": "custom", "path": str(model)}]}))
            with self.assertRaises(ValueError):
                read_models(path, ["missing"])
            self.assertEqual(read_models(path, ["custom"])[0]["path"], str(model))

    def test_report_retains_failures_and_undefined_accepted_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = {"suite": "test", "runs": [
                {"model": "missing", "device": "cpu", "status": "failed", "error": "missing file"},
                {"model": "slow", "device": "cpu", "status": "timeout"},
                {"model": "abstains", "device": "cpu", "status": "ok",
                 "quality": {"coverage": 0, "accepted_accuracy": None, "correct_fraction": 0, "top1_accuracy_before_abstention": 0.5},
                 "latency_ms": {"p50": 1, "p95": 2}, "decisions_per_second": 3,
                 "accepted_correct_decisions_per_second": 0},
            ]}
            save_summary(output, summary)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            markdown = (output / "summary.md").read_text()
            self.assertIn("| missing | cpu | failed |", markdown)
            self.assertIn("| slow | cpu | timeout |", markdown)
            self.assertIn("| abstains | cpu | ok | 0.0% | n/a | 0.0% | 50.0% |", markdown)


if __name__ == "__main__":
    unittest.main()
