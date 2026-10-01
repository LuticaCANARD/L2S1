from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable, Mapping, Sequence
from typing import Generic, TypeVar, cast, overload

import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter

from .backend import BatchDecisionBackend, DecisionBackend, L2S1Error
from .http import L2S1Client
from .models import Capabilities, Decision, DecisionRequest, DecisionResponse
from .native import LoadOptions, RustProcessBackend

StateT = TypeVar("StateT", contravariant=True)
ModelT = TypeVar("ModelT", bound=BaseModel)
_json: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class L2S1:
    """One async API for a local process, an HTTP server or a custom backend."""

    def __init__(self, backend: DecisionBackend) -> None:
        self._backend = backend
        self._closing: asyncio.Task[None] | None = None

    @classmethod
    async def load(cls, options: LoadOptions) -> L2S1:
        return cls(await RustProcessBackend.load(options))

    @classmethod
    def connect(cls, base_url: str, *, timeout_ms: int = 180_000,
                headers: Mapping[str, str] | None = None,
                transport: httpx.AsyncBaseTransport | None = None) -> L2S1:
        return cls(L2S1Client(base_url, timeout_ms=timeout_ms, headers=headers, transport=transport))

    @classmethod
    def from_backend(cls, backend: DecisionBackend) -> L2S1:
        """Transfer lifecycle ownership; optional close() may be sync or async."""
        return cls(backend)

    def _check_open(self) -> None:
        if self._closing is not None:
            raise L2S1Error("Backend is closed", "backend_closed")

    async def decide(self, request: DecisionRequest, *, timeout_ms: int | None = None) -> DecisionResponse:
        self._check_open()
        snapshot = DecisionRequest.model_validate(request.model_dump(exclude_unset=True))
        response = await self._backend.decide(snapshot, timeout_ms=timeout_ms)
        response.check_request(snapshot)
        return response

    async def decide_batch(self, requests: Iterable[DecisionRequest], *, timeout_ms: int | None = None) -> list[DecisionResponse]:
        """One native batch call. timeout_ms covers the batch; no serial fallback."""
        self._check_open()
        snapshots = [DecisionRequest.model_validate(item.model_dump(exclude_unset=True)) for item in requests]
        if not snapshots:
            return []
        if not isinstance(self._backend, BatchDecisionBackend):
            raise L2S1Error("Backend does not support native batching", "batch_unsupported")
        responses = await self._backend.decide_batch(snapshots, timeout_ms=timeout_ms)
        if len(responses) != len(snapshots):
            raise L2S1Error("Native batch returned incorrect response count", "invalid_response")
        for request, response in zip(snapshots, responses):
            response.check_request(request)
        return responses

    async def capabilities(self, *, timeout_ms: int | None = None) -> Capabilities:
        self._check_open()
        return await self._backend.capabilities(timeout_ms=timeout_ms)

    @overload
    def prepare(self, decisions: Sequence[Decision], *, state_type: type[ModelT]) -> PreparedDecision[ModelT]: ...

    @overload
    def prepare(self, decisions: Sequence[Decision], *, state_type: None = None) -> PreparedDecision[JsonValue]: ...

    def prepare(self, decisions: Sequence[Decision], *, state_type: type[BaseModel] | None = None) -> PreparedDecision[object]:
        self._check_open()
        return PreparedDecision(self, decisions, state_type)

    async def _close(self) -> None:
        close = getattr(self._backend, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await asyncio.shield(self._closing)

    async def __aenter__(self) -> L2S1:
        self._check_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class PreparedDecision(Generic[StateT]):
    """Snapshot fixed decisions; each call supplies an independent state.

    Preparation validates definitions, not model tokens or a persistent KV prefix.
    """

    def __init__(self, engine: L2S1, decisions: Sequence[Decision], state_type: type[BaseModel] | None) -> None:
        self._engine = engine
        request = DecisionRequest(state=None, decisions=list(decisions))
        self._decisions = request.model_dump(exclude_unset=True)["decisions"]
        self._state_type = state_type

    def _request(self, state: StateT) -> DecisionRequest:
        value: JsonValue
        if self._state_type is not None:
            if not isinstance(state, self._state_type):
                raise TypeError(f"state must be {self._state_type.__name__}")
            validated = self._state_type.model_validate(cast(BaseModel, state).model_dump())
            value = validated.model_dump(mode="json")
        else:
            value = _json.validate_python(state, strict=True)
        return DecisionRequest.model_validate({"state": value, "decisions": self._decisions})

    async def decide(self, state: StateT, *, timeout_ms: int | None = None) -> DecisionResponse:
        return await self._engine.decide(self._request(state), timeout_ms=timeout_ms)

    async def decide_batch(self, states: Iterable[StateT], *, timeout_ms: int | None = None) -> list[DecisionResponse]:
        return await self._engine.decide_batch([self._request(state) for state in states], timeout_ms=timeout_ms)
