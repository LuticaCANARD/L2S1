from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import tempfile
import unittest
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from l2s1 import (
    BinaryKind, BinaryValue, ChoiceKind, Decision, DecisionRequest, DecisionResponse,
    L2S1, L2S1Client, L2S1Error, Level, LoadOptions, OrdinalKind, SelectionOnlyEvidence,
)
from l2s1.native import _resolve_runtime


def request(state: object = 1) -> DecisionRequest:
    return DecisionRequest.model_validate({"state": state, "decisions": [{
        "id": "cold", "instruction": "Is temperature_c below 10?", "kind": {
            "type": "binary", "false_label": "No", "true_label": "Yes",
        },
    }]})


def response(value: bool | None = False) -> dict[str, object]:
    return {"api_version": 1, "request_id": "fixture", "backend": {
        "runtime": "provider", "model": "fixture", "details": None,
    }, "policy": None, "results": [{
        "id": "cold", "value": {"type": "binary", "value": value},
        "status": "selected" if value is not None else "abstained", "abstention_reasons": [],
        "evidence": {"type": "selection_only", "selected_code": "A", "provider_model": None},
        "usage": {"input_tokens": None},
    }]}


class Models(unittest.TestCase):
    def test_default_discriminators_and_unset_extensions_match_typescript(self) -> None:
        item = Decision(id="cold", instruction="Cold?", kind=BinaryKind(false_label="No", true_label="Yes"))
        payload = DecisionRequest(state={"temperature_c": 6}, decisions=[item]).model_dump(exclude_unset=True)
        self.assertEqual(set(payload), {"state", "decisions"})
        self.assertEqual(payload["decisions"][0]["kind"]["type"], "binary")
        DecisionRequest.model_validate(payload)

    def test_rejects_invalid_structure_and_nonfinite_state(self) -> None:
        payload = request().model_dump(exclude_unset=True)
        with self.assertRaises(ValidationError):
            DecisionRequest.model_validate({**payload, "unknown": True})
        with self.assertRaises(ValidationError):
            DecisionRequest.model_validate({**payload, "state": {"bad": float("nan")}})
        with self.assertRaises(ValidationError):
            DecisionRequest.model_validate({**payload, "decisions": payload["decisions"] * 2})
        with self.assertRaises(ValidationError):
            OrdinalKind(levels=[Level(id="a", criterion="A", value=2), Level(id="b", criterion="B", value=1)])
        with self.assertRaises(ValidationError):
            ChoiceKind.model_validate({"options": [{"id": "a", "criterion": "A"}]})
        with self.assertRaises(ValidationError):
            DecisionRequest.model_validate({**payload, "decisions": [{**payload["decisions"][0], "media_ids": ["missing"]}]})

    def test_false_null_and_evidence_narrowing_are_preserved(self) -> None:
        for value in (False, None):
            parsed = DecisionResponse.model_validate(response(value))
            result = parsed.results[0]
            self.assertIsInstance(result.value, BinaryValue)
            assert isinstance(result.value, BinaryValue)
            self.assertIs(result.value.value, value)
            self.assertIsInstance(result.evidence, SelectionOnlyEvidence)
            self.assertNotIn("scores", result.evidence.model_dump())
        with self.assertRaises(ValidationError):
            DecisionResponse.model_validate({**response(), "api_version": 2})
        with self.assertRaises(ValidationError):
            DecisionResponse.model_validate({**response(), "api_version": True})

    def test_typescript_runtime_bundle_checks_identity_and_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            executable = "l2s1.exe" if os.name == "nt" else "l2s1"
            content = b"fixture executable"
            (root / "bin" / executable).write_bytes(content)
            arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
            host = "win32" if os.name == "nt" else "darwin" if platform.system() == "Darwin" else "linux"
            manifest = {"platform": f"{host}-{arch}", "version": "0.1.0", "devices": ["cpu"], "files": [{
                "path": f"bin/{executable}", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            }]}
            (root / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            binary, _ = _resolve_runtime(LoadOptions(model="fixture", runtime_dir=root))
            # macOS temporary paths may use /var, which resolves to /private/var.
            self.assertEqual(Path(binary), (root / "bin" / executable).resolve())
            (root / "bin" / executable).write_bytes(b"tampered")
            with self.assertRaises(L2S1Error) as caught:
                _resolve_runtime(LoadOptions(model="fixture", runtime_dir=root))
            self.assertEqual(caught.exception.code, "invalid_runtime")


class Transport(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_prepared_calls_and_native_batch_preserve_independent_state(self) -> None:
        seen: list[object] = []

        def handle(req: httpx.Request) -> httpx.Response:
            payload = json.loads(req.content)
            items = payload["requests"] if req.url.path == "/v1/decision-batches" else [payload]
            for item in items:
                seen.append(item["state"])
                self.assertEqual(item["decisions"][0]["instruction"], "Is temperature_c below 10?")
            if "requests" in payload:
                self.assertEqual(len(items), 2)
                return httpx.Response(200, json={"api_version": 1, "request_id": "batch", "execution": "native_parallel", "responses": [response() for _ in items]})
            return httpx.Response(200, json=response())

        class Temperature(BaseModel):
            temperature_c: int

        async with L2S1.connect("http://fixture", transport=httpx.MockTransport(handle)) as engine:
            definitions = request().decisions
            plan = engine.prepare(definitions, state_type=Temperature)
            definitions[0].instruction = "mutated"
            await plan.decide(Temperature(temperature_c=6))
            await plan.decide_batch([Temperature(temperature_c=15), Temperature(temperature_c=2)])
            self.assertEqual(await engine.decide_batch([]), [])
            with self.assertRaises(TypeError):
                await plan.decide({"temperature_c": 1})  # type: ignore[arg-type]
            invalid_state = Temperature(temperature_c=1)
            invalid_state.temperature_c = "cold"  # type: ignore[assignment]
            with self.assertWarns(UserWarning), self.assertRaises(ValidationError):
                await plan.decide(invalid_state)
        self.assertEqual(seen, [{"temperature_c": 6}, {"temperature_c": 15}, {"temperature_c": 2}])
        with self.assertRaises(L2S1Error):
            await plan.decide(Temperature(temperature_c=1))

    async def test_native_batch_failure_never_replays_as_individual_calls(self) -> None:
        seen: list[object] = []

        def handle(req: httpx.Request) -> httpx.Response:
            self.assertEqual(req.url.path, "/v1/decision-batches")
            seen.append([item["state"] for item in json.loads(req.content)["requests"]])
            return httpx.Response(503, json={"error": {"message": "busy", "code": "busy", "request_id": "r2", "user_reason": "try later"}})

        async with L2S1.connect("http://fixture", transport=httpx.MockTransport(handle)) as engine:
            with self.assertRaises(L2S1Error) as caught:
                await engine.decide_batch([request(1), request(2), request(3)])
            self.assertEqual((caught.exception.code, caught.exception.status, caught.exception.request_id), ("busy", 503, "r2"))
            self.assertEqual(caught.exception.user_reason, "try later")
        self.assertEqual(seen, [[1, 2, 3]])

    async def test_wrong_result_mapping_invalid_json_and_mutations_are_rejected(self) -> None:
        invalid = response()
        invalid["results"] = []
        for body in (httpx.Response(200, json=invalid), httpx.Response(200, content=b"not json")):
            async with L2S1Client("http://fixture", transport=httpx.MockTransport(lambda _: body)) as client:
                with self.assertRaises(L2S1Error) as caught:
                    await client.decide(request())
                self.assertEqual(caught.exception.code, "invalid_response")
        item = request()
        item.decisions.append(item.decisions[0])
        async with L2S1Client("http://fixture", transport=httpx.MockTransport(lambda _: self.fail("must not send"))) as client:
            with self.assertRaises(ValidationError):
                await client.decide(item)

    async def test_task_cancellation_timeout_and_client_close_abort_http_work(self) -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def handle(_: httpx.Request) -> httpx.Response:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return httpx.Response(200, json=response())

        client = L2S1Client("http://fixture", transport=httpx.MockTransport(handle))
        with self.assertRaises(TimeoutError):
            await client.decide(request(), timeout_ms=10)
        self.assertTrue(cancelled.is_set())
        entered.clear()
        cancelled.clear()
        task = asyncio.create_task(client.decide(request()))
        await entered.wait()
        await asyncio.gather(client.close(), client.close())
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())
        with self.assertRaises(L2S1Error):
            await client.decide(request())

    async def test_custom_backend_optional_sync_close_and_lifecycle(self) -> None:
        class Custom:
            closes = 0

            async def decide(self, request: DecisionRequest, *, timeout_ms: int | None = None) -> DecisionResponse:
                return DecisionResponse.model_validate(response())

            async def capabilities(self, *, timeout_ms: int | None = None) -> object:
                raise NotImplementedError

            def close(self) -> None:
                self.closes += 1

        backend = Custom()
        engine = L2S1.from_backend(backend)  # type: ignore[arg-type]
        await engine.decide(request())
        with self.assertRaises(L2S1Error) as caught:
            await engine.decide_batch([request()])
        self.assertEqual(caught.exception.code, "batch_unsupported")
        await asyncio.gather(engine.close(), engine.close())
        self.assertEqual(backend.closes, 1)
        with self.assertRaises(L2S1Error):
            await engine.decide(request())


if __name__ == "__main__":
    unittest.main()
