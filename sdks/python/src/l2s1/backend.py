from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import Capabilities, DecisionRequest, DecisionResponse


class DecisionBackend(Protocol):
    """Custom adapters share the TypeScript backend's application contract."""

    async def decide(self, request: DecisionRequest, *, timeout_ms: int | None = None) -> DecisionResponse: ...
    async def capabilities(self, *, timeout_ms: int | None = None) -> Capabilities: ...


@runtime_checkable
class BatchDecisionBackend(DecisionBackend, Protocol):
    """Optional native batch capability, never a serial loop."""
    async def decide_batch(self, requests: Sequence[DecisionRequest], *, timeout_ms: int | None = None) -> list[DecisionResponse]: ...


class L2S1Error(Exception):
    def __init__(self, message: str, code: str, *, status: int | None = None,
                 request_id: str | None = None, user_reason: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id
        self.user_reason = user_reason


def positive_timeout(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2_147_483_647:
        raise ValueError("timeout_ms must be an integer in 1..2147483647")
    return value
