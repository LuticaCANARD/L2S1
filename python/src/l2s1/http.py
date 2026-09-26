from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from .wire import JsonDecisionClient
from .backend import L2S1Error, positive_timeout
from .models import BatchResponse, Capabilities, DecisionRequest, DecisionResponse


class L2S1Client(JsonDecisionClient):
    """Async HTTP transport. Cancellation never implies native inference stopped."""

    def __init__(self, base_url: str, *, timeout_ms: int = 180_000,
                 headers: Mapping[str, str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        url = urlsplit(base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        self.base_url = base_url.rstrip("/")
        self.timeout_ms = positive_timeout(timeout_ms)
        self._client = httpx.AsyncClient(headers=headers, transport=transport, trust_env=False,
                                       follow_redirects=False)
        self._closed = False
        self._pending: set[asyncio.Task[httpx.Response]] = set()
        self._closing: asyncio.Task[None] | None = None

    async def _request(self, path: str, body: str | None, timeout_ms: int | None) -> object:
        if self._closed:
            raise L2S1Error("HTTP client is closed", "backend_closed")
        timeout = positive_timeout(self.timeout_ms if timeout_ms is None else timeout_ms) / 1000
        task = asyncio.create_task(self._client.request(
            "GET" if body is None else "POST", self.base_url + path,
            content=None if body is None else body.encode("utf-8"),
            headers={"accept": "application/json", **({"content-type": "application/json"} if body is not None else {})},
            timeout=timeout,
        ))
        self._pending.add(task)
        try:
            # Includes body consumption and pool wait, not just per-I/O timeouts.
            async with asyncio.timeout(timeout):
                response = await task
        finally:
            self._pending.discard(task)
        try:
            value: object = json.loads(response.content, parse_constant=_reject_constant)
        except (ValueError, UnicodeDecodeError) as error:
            raise L2S1Error("Server returned invalid JSON", "invalid_response", status=response.status_code) from error
        if not response.is_success:
            detail = value.get("error") if isinstance(value, dict) else None
            detail = detail if isinstance(detail, dict) else {}
            raise L2S1Error(
                str(detail.get("message", f"HTTP {response.status_code}")), str(detail.get("code", "http_error")),
                status=response.status_code, request_id=_string(detail.get("request_id")),
                user_reason=_string(detail.get("user_reason")),
            )
        return value

    async def _close(self) -> None:
        pending = list(self._pending)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self._client.aclose()

    async def close(self) -> None:
        self._closed = True
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await asyncio.shield(self._closing)

    async def __aenter__(self) -> L2S1Client:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")
