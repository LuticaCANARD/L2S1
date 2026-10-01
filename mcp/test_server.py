"""Contract tests through an official SDK client and a fixture HTTP backend."""

import asyncio
import base64
from copy import deepcopy
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import ValidationError

from server import Backend, Request

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = json.loads((ROOT / "examples/warehouse.json").read_text())


class ValidationTests(unittest.TestCase):
    def test_valid_kinds_and_exact_body(self):
        self.assertEqual(json.loads(Request.model_validate(WAREHOUSE).body()), WAREHOUSE)
        request = Request.model_validate({"state": None, "decisions": [WAREHOUSE["decisions"][1]]})
        self.assertIsNone(json.loads(request.body())["state"])

    def test_invalid_semantics_and_types(self):
        cases = []
        duplicate = deepcopy(WAREHOUSE)
        duplicate["decisions"][1]["id"] = duplicate["decisions"][0]["id"]
        cases.append(duplicate)
        duplicate_option = deepcopy(WAREHOUSE)
        duplicate_option["decisions"][0]["kind"]["options"][1]["id"] = "ambient"
        cases.append(duplicate_option)
        unordered = deepcopy(WAREHOUSE)
        unordered["decisions"][2]["kind"]["levels"][1]["value"] = 0
        cases.append(unordered)
        for value in ["1", True, float("inf")]:
            wrong_value = deepcopy(WAREHOUSE)
            wrong_value["decisions"][2]["kind"]["levels"][1]["value"] = value
            cases.append(wrong_value)
        blank = deepcopy(WAREHOUSE)
        blank["decisions"][0]["instruction"] = " \n"
        cases.append(blank)
        cases.extend([{"state": {}, "decisions": []}, {**WAREHOUSE, "unknown": 1},
                      {**WAREHOUSE, "state": {"invalid": float("nan")}}])
        for index, case in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises((ValidationError, ValueError)):
                    Request.model_validate(case)

    def test_media_references(self):
        request = deepcopy(WAREHOUSE)
        request["media"] = [{"type": "image", "id": "photo", "data_base64": base64.b64encode(b"bytes").decode()}]
        request["decisions"][0]["media_ids"] = ["photo"]
        request["decisions"][1]["media_ids"] = []
        self.assertEqual(json.loads(Request.model_validate(request).body()), request)
        for ids in [["unknown"], ["photo", "photo"]]:
            request["decisions"][0]["media_ids"] = ids
            with self.assertRaises(ValidationError):
                Request.model_validate(request)
        request["decisions"][0]["media_ids"] = ["photo"]
        request["media"][0]["data_base64"] = "not base64!"
        with self.assertRaises(ValidationError):
            Request.model_validate(request)

    def test_limits_and_origin(self):
        too_many = deepcopy(WAREHOUSE)
        too_many["decisions"] = [dict(WAREHOUSE["decisions"][1], id=f"d-{i}") for i in range(129)]
        with self.assertRaises(ValidationError):
            Request.model_validate(too_many)
        for url in ["file:///tmp/test", "http://user:pass@localhost", "http://localhost/path", "http://localhost?token=x"]:
            with self.assertRaises(ValueError):
                Backend(url, 1)
        for timeout in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                Backend("http://127.0.0.1:8080", timeout)

    def test_optional_http_fields(self):
        request = {**WAREHOUSE, "reasoning": {"mode": "direct", "max_tokens": 128},
                   "policy": {"min_top_probability": 0.9, "min_candidate_mass": 0.1},
                   "target_error_rate": 0.1, "failure_reasons": {"low_top_probability": "판단 보류"}}
        self.assertEqual(json.loads(Request.model_validate(request).body()), request)
        for key, value in [("reasoning", {"max_tokens": 0}), ("reasoning", {"mode": "auto"}),
                           ("reasoning", {"max_tokens": True}), ("target_error_rate", 1.1),
                           ("policy", {"min_top_probability": 0.5}),
                           ("failure_reasons", {"invented": "message"}),
                           ("failure_reasons", {"native_failure": "가" * 171})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                Request.model_validate({**request, key: value})


class Fixture(BaseHTTPRequestHandler):
    status = 200
    calls = []
    response = {
        "api_version": 1, "request_id": "fixture-request",
        "backend": {"runtime": "fixture", "model": "fixture.gguf", "details": None},
        "policy": {"min_top_probability": 0.8, "min_candidate_mass": 0.05},
        "results": [
            {"id": "negative", "value": {"type": "binary", "value": False},
             "status": "selected", "abstention_reasons": [],
             "evidence": {"type": "model_scored", "candidate_mass": 0.9}, "usage": {}},
            {"id": "uncertain", "value": {"type": "choice", "selected": None},
             "status": "abstained", "abstention_reasons": ["low_top_probability"],
             "evidence": {"type": "model_scored", "top_option_probability": 0.6}, "usage": {}},
        ],
    }

    def log_message(self, *_):
        pass

    def send_json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.calls.append(("GET", self.path, None))
        self.send_json(200, {"api_version": 1, "evidence": "model_scored", "fixture": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append(("POST", self.path, body))
        if self.status != 200:
            self.send_json(self.status, {"error": {"code": "busy", "message": "fixture queue full"}})
        else:
            self.send_json(200, self.response)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_errors_and_timeout_without_retries(self):
        original_client = httpx.AsyncClient
        for status, body, expected in [(200, b"[]", "JSON object"),
                                       (200, b"not json", "invalid JSON"),
                                       (200, b'{"bad":NaN}', "invalid JSON"),
                                       (302, b"redirect", "HTTP 302")]:
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, content=body)
            def client(**kwargs):
                self.assertFalse(kwargs["follow_redirects"])
                self.assertFalse(kwargs["trust_env"])
                return original_client(transport=httpx.MockTransport(handler), **kwargs)
            with self.subTest(status=status, body=body), patch("server.httpx.AsyncClient", client):
                with self.assertRaisesRegex(ValueError, expected):
                    await Backend("http://127.0.0.1:8080", 1).request("/v1/capabilities")
            self.assertEqual(len(calls), 1)
        calls = []
        def timeout(request):
            calls.append(request)
            raise httpx.ReadTimeout("fixture timeout", request=request)
        with patch("server.httpx.AsyncClient", lambda **kwargs: original_client(
                transport=httpx.MockTransport(timeout), **kwargs)):
            with self.assertRaisesRegex(ValueError, "may still be running"):
                await Backend("http://127.0.0.1:8080", 1).request("/v1/decisions", b"{}")
        self.assertEqual(len(calls), 1)

    async def test_stdio_session_and_http_contract(self):
        Fixture.calls = []
        Fixture.status = 200
        fixture = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        thread = threading.Thread(target=fixture.serve_forever, daemon=True)
        thread.start()
        try:
            params = StdioServerParameters(command=sys.executable, args=[
                str(ROOT / "mcp/server.py"), "--backend-url",
                f"http://127.0.0.1:{fixture.server_port}"], cwd="/tmp")
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    self.assertEqual({tool.name for tool in tools}, {
                        "l2s1_document", "l2s1_example", "l2s1_validate", "l2s1_capabilities", "l2s1_decide"})
                    schema = next(tool for tool in tools if tool.name == "l2s1_decide").inputSchema
                    self.assertIn("Request", schema["$defs"])
                    self.assertIn("oneOf", schema["$defs"]["Decision"]["properties"]["kind"])
                    resources = (await session.list_resources()).resources
                    for resource in resources:
                        result = await session.read_resource(resource.uri)
                        self.assertTrue(result.contents[0].text)
                    example = await session.call_tool("l2s1_example", {"name": "warehouse"})
                    self.assertEqual(example.structuredContent, WAREHOUSE)
                    doc = await session.call_tool("l2s1_document", {"name": "guide"})
                    self.assertFalse(doc.isError)
                    self.assertIn("L2S1", doc.content[0].text)
                    traversal = await session.call_tool("l2s1_document", {"name": "../../etc/passwd"})
                    self.assertTrue(traversal.isError)
                    valid = await session.call_tool("l2s1_validate", {"request": WAREHOUSE})
                    self.assertFalse(valid.isError)
                    self.assertEqual(valid.structuredContent["scope"], "structure_only")
                    self.assertEqual(Fixture.calls, [])
                    malformed = await session.call_tool("l2s1_decide", {"request": {"state": {}, "decisions": []}})
                    self.assertTrue(malformed.isError)
                    self.assertEqual(Fixture.calls, [])
                    capabilities = await session.call_tool("l2s1_capabilities")
                    self.assertTrue(capabilities.structuredContent["fixture"])
                    response = await session.call_tool("l2s1_decide", {"request": WAREHOUSE})
                    self.assertFalse(response.isError)
                    self.assertEqual(response.structuredContent, Fixture.response)
                    self.assertEqual(json.loads(response.content[0].text), Fixture.response)
                    self.assertEqual(Fixture.calls[-1], ("POST", "/v1/decisions", WAREHOUSE))
                    extended = {**WAREHOUSE, "reasoning": {"mode": "direct"},
                                "policy": {"min_top_probability": 0.8, "min_candidate_mass": 0.05},
                                "target_error_rate": 0.2,
                                "failure_reasons": {"low_top_probability": "custom explanation"}}
                    response = await session.call_tool("l2s1_decide", {"request": extended})
                    self.assertFalse(response.isError)
                    self.assertEqual(Fixture.calls[-1], ("POST", "/v1/decisions", extended))
                    selection = deepcopy(Fixture.response)
                    selection["policy"] = None
                    selection["results"][0]["evidence"] = {"type": "selection_only"}
                    original = Fixture.response
                    Fixture.response = selection
                    selection_only = await session.call_tool("l2s1_decide", {"request": WAREHOUSE})
                    self.assertEqual(selection_only.structuredContent, selection)
                    Fixture.response = original
                    Fixture.status = 503
                    failure = await session.call_tool("l2s1_decide", {"request": WAREHOUSE})
                    self.assertTrue(failure.isError)
                    self.assertIn("503", failure.content[0].text)
                    self.assertIn("busy", failure.content[0].text)
                    self.assertEqual(len([call for call in Fixture.calls if call[0] == "POST"]), 4)
                    prompts = await session.list_prompts()
                    self.assertEqual(prompts.prompts[0].name, "design_decision")
                    prompt = await session.get_prompt("design_decision", {"task": "route a support ticket"})
                    self.assertIn("route a support ticket", prompt.messages[0].content.text)
        finally:
            await asyncio.to_thread(fixture.shutdown)
            fixture.server_close()
            thread.join(timeout=2)

    async def test_backend_failure_is_a_tool_error(self):
        params = StdioServerParameters(command=sys.executable, args=[
            str(ROOT / "mcp/server.py"), "--backend-url", "http://127.0.0.1:1", "--timeout", "1"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session:
                await session.initialize()
                valid = await session.call_tool("l2s1_validate", {"request": WAREHOUSE})
                self.assertFalse(valid.isError)
                failure = await session.call_tool("l2s1_capabilities")
                self.assertTrue(failure.isError)
                self.assertIn("Cannot reach L2S1", failure.content[0].text)


if __name__ == "__main__":
    unittest.main()
