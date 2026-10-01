"""Checked by mypy with warn_unused_ignores: invalid inputs must be rejected."""
from typing import assert_type

from pydantic import BaseModel, JsonValue

from l2s1 import BinaryKind, BinaryValue, Decision, L2S1, ModelScoredEvidence, PreparedDecision


class Temperature(BaseModel):
    temperature_c: float


async def typed_consumer(engine: L2S1) -> None:
    definitions = [Decision(id="cold", instruction="Cold?", kind=BinaryKind(false_label="No", true_label="Yes"))]
    plan = engine.prepare(definitions, state_type=Temperature)
    assert_type(plan, PreparedDecision[Temperature])
    response = await plan.decide(Temperature(temperature_c=6))
    await plan.decide_batch([Temperature(temperature_c=2)])
    await plan.decide({"temperature_c": 6})  # type: ignore[arg-type]
    Temperature(temperature_c="cold")  # type: ignore[arg-type]
    result = response.results[0]
    if isinstance(result.value, BinaryValue):
        assert_type(result.value.value, bool | None)
    if isinstance(result.evidence, ModelScoredEvidence):
        assert_type(result.evidence.candidate_mass, float)
    plain = engine.prepare(definitions)
    assert_type(plain, PreparedDecision[JsonValue])
    await plain.decide({"temperature_c": 6})
