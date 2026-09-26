from __future__ import annotations

import asyncio
import json

from .backend import L2S1Error, positive_timeout
from .wire import JsonDecisionClient


class StdioClient(JsonDecisionClient):
    """JSON-line RPC to the compiled Rust child; no HTTP or socket."""
    def __init__(self, child: asyncio.subprocess.Process, timeout_ms: int) -> None:
        self._child = child
        self._timeout_ms = positive_timeout(timeout_ms)
        self._closed = False
        self._counter = 0
        self._pending: dict[str, asyncio.Future[object]] = {}
        self._write_lock = asyncio.Lock()
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        assert self._child.stdout is not None
        try:
            while line := await self._child.stdout.readline():
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("id"), str) or value["id"] not in self._pending:
                    raise L2S1Error("Invalid stdio response ID", "invalid_response")
                future = self._pending.pop(value["id"])
                if future.done():
                    continue  # Timed-out/cancelled inference is never replayed.
                if isinstance(value.get("error"), dict):
                    error = value["error"]
                    future.set_exception(L2S1Error(str(error["message"]), str(error["code"]), request_id=value["id"],
                                                  user_reason=error.get("user_reason")))
                elif "result" in value:
                    future.set_result(value["result"])
                else:
                    future.set_exception(L2S1Error("Invalid stdio envelope", "invalid_response"))
            self._fail(L2S1Error("Rust process closed", "process_closed"))
        except asyncio.CancelledError:
            self._fail(L2S1Error("stdio client closed", "backend_closed"))
        except Exception as error:
            self._fail(error)

    def _fail(self, error: Exception) -> None:
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _request(self, path: str, body: str | None, timeout_ms: int | None) -> object:
        if self._closed:
            raise L2S1Error("stdio client is closed", "backend_closed")
        if len(self._pending) >= 16:
            raise L2S1Error("stdio inference queue is full", "busy")
        timeout = positive_timeout(self._timeout_ms if timeout_ms is None else timeout_ms) / 1000
        op = {"/healthz": "health", "/v1/capabilities": "capabilities", "/v1/decisions": "decide", "/v1/decision-batches": "decide_batch"}[path]
        self._counter += 1
        call_id = f"local-{self._counter}"
        line = json.dumps({"id": call_id, "op": op, "body": None if body is None else json.loads(body)},
                          ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if len(line) > 44 * 1024 * 1024 + 1024:
            raise L2S1Error("stdio request exceeds limit", "invalid_request")
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        sent = False
        try:
            async with asyncio.timeout(timeout):
                async with self._write_lock:
                    assert self._child.stdin is not None
                    self._child.stdin.write(line)
                    sent = True
                    await self._child.stdin.drain()
                return await future
        finally:
            if not sent:
                self._pending.pop(call_id, None)
            if not future.done():
                future.cancel()

    async def close(self) -> None:
        self._fail(L2S1Error("stdio client is closed", "backend_closed"))
        self._reader.cancel()
        await asyncio.gather(self._reader, return_exceptions=True)
