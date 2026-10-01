"""Optional real GGUF CPU/native batching smoke; no accuracy claim."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from l2s1 import DecisionRequest, JsonValue, L2S1, L2S1Error, LoadOptions, ModelScoredEvidence


@unittest.skipUnless(os.environ.get("L2S1_MODEL"), "set L2S1_MODEL for real native GGUF smoke")
class RealModel(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_native_parallel_independent_states_and_shared_prefix_usage(self) -> None:
        root = Path(__file__).resolve().parents[3]
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
            states: list[JsonValue] = [request.state, {"temperature_c": 20}, {"temperature_c": 2}]
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

    async def test_default_resident_reuse_and_explicit_fresh(self) -> None:
        root = Path(__file__).resolve().parents[3]
        request = DecisionRequest.model_validate(json.loads((root / "examples/warehouse.json").read_text()))
        transports: tuple[Literal["stdio", "http"], ...] = ("stdio", "http")
        request.decisions = request.decisions[:1]
        for transport in transports:
            for fixed_schema in (None, False):
                with self.subTest(transport=transport, fixed_schema=fixed_schema):
                    engine = await L2S1.load(LoadOptions(
                        model=os.environ["L2S1_MODEL"], binary_path=os.environ.get("L2S1_BINARY", "l2s1"),
                        device="cuda" if os.environ.get("L2S1_DEVICE") == "cuda" else "cpu",
                        transport=transport, fixed_schema=fixed_schema, context=2048, batch=256, threads=2,
                    ))
                    async with engine:
                        cold = await engine.decide(request)
                        warm = await engine.decide(request)
                        self.assertEqual([r.evidence for r in cold.results], [r.evidence for r in warm.results])
                        reused = sum(r.usage.reused_prefix_tokens or 0 for r in warm.results)
                        assert isinstance(warm.backend.details, dict)
                        if fixed_schema is None:
                            self.assertGreater(reused, 0)
                            self.assertEqual(warm.backend.details["prefix_plan"], "fixed-schema-split-v1")
                            caps = await engine.capabilities()
                            assert caps.prefix_reuse is not None
                            self.assertEqual(caps.prefix_reuse["schema_change"], "clear_all")
                        else:
                            self.assertEqual(reused, 0)
                            self.assertNotIn("prefix_plan", warm.backend.details)
                        plan = engine.prepare(request.decisions)
                        changed = await plan.decide({"storage_requirement": "frozen"})
                        self.assertEqual((changed.results[0].usage.reused_prefix_tokens or 0) > 0, fixed_schema is None)
                        other = request.model_copy(deep=True)
                        other.decisions[0].id = "changed_schema"
                        switched = await engine.decide(other)
                        self.assertEqual(switched.results[0].usage.reused_prefix_tokens, 0)
                        returned = await plan.decide(request.state)
                        self.assertEqual(returned.results[0].usage.reused_prefix_tokens, 0)
                        with self.assertRaises(L2S1Error) as error:
                            await plan.decide({"oversized": "word " * 4096})
                        self.assertIn(error.exception.code, ("invalid_request", "backend_error"))
                        recovered = await plan.decide(request.state)
                        self.assertEqual(recovered.results[0].usage.reused_prefix_tokens, 0)
                        print(json.dumps({"sdk": "python", "transport": transport,
                                          "fixed_schema": fixed_schema, "reused_prefix_tokens": reused,
                                          "schema_return_tokens": returned.results[0].usage.reused_prefix_tokens,
                                          "recovery_tokens": recovered.results[0].usage.reused_prefix_tokens}))
