from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import ValidationError

from .backend import L2S1Error
from .models import BatchResponse, Capabilities, DecisionRequest, DecisionResponse


class JsonDecisionClient:
    """Shared typed wire validation for HTTP and compiled-process RPC."""
    _closed: bool

    async def _request(self, path: str, body: str | None, timeout_ms: int | None) -> object:
        raise NotImplementedError

    async def health(self, *, timeout_ms: int | None = None) -> None:
        value = await self._request("/healthz", None, timeout_ms)
        if not isinstance(value, dict) or value.get("status") != "ok":
            raise L2S1Error("Invalid health response", "invalid_response")

    async def capabilities(self, *, timeout_ms: int | None = None) -> Capabilities:
        value = await self._request("/v1/capabilities", None, timeout_ms)
        try:
            return Capabilities.model_validate(value)
        except ValidationError as error:
            raise L2S1Error("Invalid v1 capabilities response", "invalid_response") from error

    async def decide(self, request: DecisionRequest, *, timeout_ms: int | None = None) -> DecisionResponse:
        # Revalidate nested values even if a caller mutated an existing model.
        snapshot = DecisionRequest.model_validate(request.model_dump(exclude_unset=True))
        value = await self._request("/v1/decisions", snapshot.model_dump_json(exclude_unset=True), timeout_ms)
        try:
            response = DecisionResponse.model_validate(value)
            response.check_request(snapshot)
            return response
        except ValueError as error:
            raise L2S1Error("Server response does not match the v1 decision contract", "invalid_response") from error

    async def decide_batch(self, requests: Sequence[DecisionRequest], *, timeout_ms: int | None = None) -> list[DecisionResponse]:
        if self._closed:
            raise L2S1Error("Decision client is closed", "backend_closed")
        snapshots = [DecisionRequest.model_validate(item.model_dump(exclude_unset=True)) for item in requests]
        if not snapshots:
            return []
        body = json.dumps({"requests": [item.model_dump(exclude_unset=True) for item in snapshots]}, allow_nan=False, ensure_ascii=False)
        value = await self._request("/v1/decision-batches", body, timeout_ms)
        try:
            batch = BatchResponse.model_validate(value)
            if len(batch.responses) != len(snapshots):
                raise ValueError("incorrect native batch response count")
            for request, response in zip(snapshots, batch.responses):
                response.check_request(request)
            return batch.responses
        except ValueError as error:
            raise L2S1Error("Server response does not match the native batch contract", "invalid_response") from error
