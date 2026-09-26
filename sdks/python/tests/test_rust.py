"""Requires a compiled Rust fixture; no model/native inference is used."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from l2s1 import DecisionRequest, L2S1, L2S1Error, LoadOptions, ModelScoredEvidence

ROOT = Path(__file__).resolve().parents[3]
BINARY = os.environ.get("L2S1_TEST_BINARY")


@unittest.skipUnless(BINARY, "set L2S1_TEST_BINARY to target/debug/examples/typescript_fixture")
class RustBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_compiled_stdio_batch_without_any_http_client(self) -> None:
        assert BINARY is not None
        logs: list[str] = []
        with patch("l2s1.native.L2S1Client", side_effect=AssertionError("stdio must not construct an HTTP client")):
            engine = await L2S1.load(LoadOptions(model="fixture path with spaces", binary_path=BINARY, on_stderr=logs.append))
        async with engine:
            self.assertIn("l2s1 stdio ready", "".join(logs))
            self.assertNotIn("HTTP listening", "".join(logs))
            request = DecisionRequest.model_validate(json.loads((ROOT / "examples/warehouse.json").read_text()))
            plan = engine.prepare(request.decisions)
            responses = await plan.decide_batch([request.state, {"storage_requirement": "ambient"}])
            self.assertEqual(len(responses), 2)
            self.assertNotEqual(responses[0].request_id, responses[1].request_id)
            items = [request, DecisionRequest(state={"storage_requirement": "ambient"}, decisions=request.decisions)]
            child = await asyncio.create_subprocess_exec("node", str(ROOT / "sdks/python/tests/typescript_peer.mjs"),
                "--stdio", BINARY, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await child.communicate(json.dumps([item.model_dump(exclude_unset=True) for item in items]).encode())
            self.assertEqual(child.returncode, 0, stderr.decode())
            ts = json.loads(stdout)
            py = [item.model_dump(exclude_unset=True) for item in responses]
            for ts_item, py_item in zip(ts, py):
                ts_item.pop("request_id")
                py_item.pop("request_id")
            self.assertEqual(py, ts)
            invalid = DecisionRequest.model_validate({**request.model_dump(exclude_unset=True), "reasoning": {"mode": "thinking"}})
            with self.assertRaises(L2S1Error) as caught:
                await engine.decide_batch([invalid])
            self.assertEqual(caught.exception.code, "invalid_request")
            await engine.decide(request)

    async def test_resident_process_repeated_state_batch_and_typescript_json_equivalence(self) -> None:
        logs: list[str] = []
        assert BINARY is not None
        engine = await L2S1.load(LoadOptions(model="fixture path with spaces", binary_path=BINARY, transport="http", on_stderr=logs.append))
        try:
            capabilities = await engine.capabilities()
            self.assertEqual(capabilities.backend.runtime, "rust-fixture")
            request = DecisionRequest.model_validate(json.loads((ROOT / "examples/warehouse.json").read_text()))
            plan = engine.prepare(request.decisions)
            results = await plan.decide_batch([request.state, {"storage_requirement": "ambient"}])
            self.assertNotEqual(results[0].request_id, results[1].request_id)
            self.assertEqual([result.id for result in results[0].results], [item.id for item in request.decisions])
            self.assertIsInstance(results[0].results[0].evidence, ModelScoredEvidence)
            address = re.search(r"l2s1 HTTP listening on (127\.0\.0\.1:\d+)", "".join(logs))
            assert address is not None
            if shutil.which("node") and (ROOT / "sdks/typescript/dist/index.js").exists():
                child = await asyncio.create_subprocess_exec("node", str(ROOT / "sdks/python/tests/typescript_peer.mjs"),
                    "http://" + address[1], stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await child.communicate(request.model_dump_json(exclude_unset=True).encode())
                self.assertEqual(child.returncode, 0, stderr.decode())
                ts = json.loads(stdout)
                py = (await engine.decide(request)).model_dump(exclude_unset=True)
                ts.pop("request_id")
                py.pop("request_id")
                self.assertEqual(py, ts)
            else:
                self.fail("build TypeScript SDK and install Node for the cross-SDK boundary check")
            invalid = request.model_copy(update={"decisions": request.decisions * 2})
            with self.assertRaises(ValueError):
                await engine.decide(invalid)
            await engine.decide(request)
        finally:
            await asyncio.gather(engine.close(), engine.close())
        with self.assertRaises(L2S1Error):
            await engine.decide(request)

    async def test_remote_close_leaves_resident_owner_available(self) -> None:
        assert BINARY is not None
        logs: list[str] = []
        engine = await L2S1.load(LoadOptions(model="fixture", binary_path=BINARY, transport="http", on_stderr=logs.append))
        async with engine:
            address = re.search(r"l2s1 HTTP listening on (127\.0\.0\.1:\d+)", "".join(logs))
            assert address is not None
            async with L2S1.connect("http://" + address[1]) as remote:
                await remote.capabilities()
            await engine.capabilities()


if __name__ == "__main__":
    unittest.main()
