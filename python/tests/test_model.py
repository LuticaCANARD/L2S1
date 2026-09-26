"""Optional real GGUF CPU/native batching smoke; no accuracy claim."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from l2s1 import DecisionRequest, L2S1, LoadOptions, ModelScoredEvidence


@unittest.skipUnless(os.environ.get("L2S1_MODEL"), "set L2S1_MODEL for real native GGUF smoke")
class RealModel(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_native_parallel_independent_states_and_shared_prefix_usage(self) -> None:
        root = Path(__file__).resolve().parents[2]
        request = DecisionRequest.model_validate(json.loads((root / "examples/warehouse.json").read_text()))
        with patch("l2s1.native.L2S1Client", side_effect=AssertionError("must not use HTTP")):
            engine = await L2S1.load(LoadOptions(
                model=os.environ["L2S1_MODEL"], binary_path=os.environ.get("L2S1_BINARY", "l2s1"),
                execution_mode="parallel", parallel_width=2, context=2048, batch=32, threads=2,
            ))
        async with engine:
            capabilities = await engine.capabilities()
            assert capabilities.batch is not None
            self.assertTrue(capabilities.batch.enabled)
            states = [request.state, {"temperature_c": 20}, {"temperature_c": 2}]
            responses = await engine.prepare(request.decisions).decide_batch(states)
            self.assertEqual(len(responses), len(states))
            reused = 0
            for index, response in enumerate(responses):
                self.assertTrue(response.request_id.endswith(f"/{index}"))
                self.assertEqual([r.id for r in response.results], [d.id for d in request.decisions])
                for result in response.results:
                    self.assertIsInstance(result.evidence, ModelScoredEvidence)
                    assert isinstance(result.evidence, ModelScoredEvidence)
                    self.assertAlmostEqual(sum(s.option_probability for s in result.evidence.scores), 1)
                    reused += result.usage.reused_prefix_tokens or 0
            self.assertGreater(reused, 0)
            print(json.dumps({"sdk": "python", "transport": "stdio", "execution": "native_parallel",
                              "requests": len(responses), "reused_prefix_tokens": reused}))
            await engine.decide(request)
