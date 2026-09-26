"""Typed models for the same HTTP v1 JSON contract as @l2s1/node."""
from __future__ import annotations

import base64
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, model_validator


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


Text = Annotated[str, AfterValidator(_nonblank)]
FailureReasonCode: TypeAlias = Literal["low_candidate_mass", "low_top_probability", "tied_candidates", "reasoning_limit", "native_failure"]


def _api_version(value: object) -> object:
    # Python's True == 1 must not admit an envelope rejected by TypeScript.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 1:
        raise ValueError("api_version must be the number 1")
    return value


ApiVersion: TypeAlias = Annotated[Literal[1], BeforeValidator(_api_version)]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, revalidate_instances="always")

    def model_post_init(self, context: object) -> None:
        # Discriminators are mandatory on the wire even when their constructor
        # default is implicit. Other unset extension fields stay omitted.
        if "type" in type(self).model_fields:
            self.__pydantic_fields_set__.add("type")


class ResponseModel(BaseModel):
    # Preserve future server metadata; known fields remain strictly validated.
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)


class OptionSpec(RequestModel):
    id: Text
    criterion: Text


class Level(OptionSpec):
    value: float


class BinaryKind(RequestModel):
    type: Literal["binary"] = "binary"
    false_label: Text
    true_label: Text


class ChoiceKind(RequestModel):
    type: Literal["choice"] = "choice"
    options: list[OptionSpec] = Field(min_length=2)

    @model_validator(mode="after")
    def unique_options(self) -> Self:
        _unique([item.id for item in self.options], "option")
        return self


class OrdinalKind(RequestModel):
    type: Literal["ordinal"] = "ordinal"
    levels: list[Level] = Field(min_length=2)

    @model_validator(mode="after")
    def ordered_levels(self) -> Self:
        _unique([item.id for item in self.levels], "level")
        if any(a.value >= b.value for a, b in zip(self.levels, self.levels[1:])):
            raise ValueError("ordinal values must be strictly increasing")
        return self


DecisionKind: TypeAlias = Annotated[BinaryKind | ChoiceKind | OrdinalKind, Field(discriminator="type")]


class Decision(RequestModel):
    id: Text
    instruction: Text
    kind: DecisionKind
    media_ids: list[str] | None = None


class ImageMedia(RequestModel):
    type: Literal["image"] = "image"
    id: Text
    data_base64: str = Field(max_length=((8 * 1024 * 1024 + 2) // 3) * 4)

    @model_validator(mode="after")
    def valid_image_bytes(self) -> Self:
        data = base64.b64decode(self.data_base64, validate=True)
        if not 0 < len(data) <= 8 * 1024 * 1024:
            raise ValueError("image must contain 1..8 MiB of bytes")
        return self


class DecisionPolicy(RequestModel):
    min_top_probability: float = Field(ge=0, le=1)
    min_candidate_mass: float = Field(ge=0, le=1)


class ReasoningOptions(RequestModel):
    mode: Literal["direct", "thinking"] = "direct"
    max_tokens: int = Field(default=128, ge=1, le=1024)


def _unique(ids: list[str], label: str) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} IDs must be unique")


class DecisionRequest(RequestModel):
    state: JsonValue
    decisions: list[Decision] = Field(min_length=1, max_length=128)
    media: list[ImageMedia] = Field(default_factory=list, max_length=4)
    policy: DecisionPolicy | None = None
    reasoning: ReasoningOptions | None = None
    target_error_rate: float | None = Field(default=None, ge=0, le=1)
    failure_reasons: dict[FailureReasonCode, Text] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_references(self) -> Self:
        _unique([item.id for item in self.decisions], "decision")
        _unique([item.id for item in self.media], "media")
        known = {item.id for item in self.media}
        for decision in self.decisions:
            if decision.media_ids is not None:
                _unique(decision.media_ids, "media reference")
                if not set(decision.media_ids) <= known:
                    raise ValueError("decision references unknown media ID")
        if any(len(value.encode("utf-8")) > 512 for value in self.failure_reasons.values()):
            raise ValueError("failure messages must be at most 512 UTF-8 bytes")
        if len(self.model_dump_json(exclude_unset=True).encode("utf-8")) > 44 * 1024 * 1024:
            raise ValueError("request exceeds the HTTP v1 44 MiB limit")
        return self


class BinaryValue(ResponseModel):
    type: Literal["binary"]
    value: bool | None


class ChoiceValue(ResponseModel):
    type: Literal["choice"]
    selected: str | None


class OrdinalValue(ResponseModel):
    type: Literal["ordinal"]
    selected: str | None
    level_value: float | None = None


DecisionValue: TypeAlias = Annotated[BinaryValue | ChoiceValue | OrdinalValue, Field(discriminator="type")]


class OptionScore(ResponseModel):
    id: str
    code: str
    token_id: int
    token_ids: list[int] | None = None
    raw_logit: float
    option_probability: float


class Estimate(ResponseModel):
    p_true: float | None = None
    expected_value: float | None = None


class ModelScoredEvidence(ResponseModel):
    type: Literal["model_scored"]
    scores: list[OptionScore]
    candidate_mass: float
    top_option_probability: float
    entropy_confidence: float
    scoring_method: str
    calibration_id: str | None
    truncated: bool
    code_prefix_evaluations: int
    code_evaluated_tokens: int
    estimate: Estimate


class SelectionOnlyEvidence(ResponseModel):
    type: Literal["selection_only"]
    selected_code: str | None
    provider_model: str | None


Evidence: TypeAlias = Annotated[ModelScoredEvidence | SelectionOnlyEvidence, Field(discriminator="type")]


class ReasoningUsage(ResponseModel):
    mode: Literal["direct", "thinking"]
    generated_tokens: int
    completed: bool


class Usage(ResponseModel):
    input_tokens: int | None
    reused_prefix_tokens: int | None = None
    output_tokens: int | None = None
    reasoning: ReasoningUsage | None = None


class ReasonMessage(ResponseModel):
    code: str
    message: str
    user_defined: Literal[True]


class DecisionResult(ResponseModel):
    id: str
    value: DecisionValue
    status: Literal["selected", "abstained"]
    abstention_reasons: list[str]
    reason_messages: list[ReasonMessage] | None = None
    evidence: Evidence
    usage: Usage


class BackendInfo(ResponseModel):
    runtime: str
    model: str
    details: JsonValue


class ErrorBudget(ResponseModel):
    requested_rate: float
    guaranteed: Literal[False]
    interpretation: Literal["model_score_threshold"]
    min_top_probability: float


class DecisionResponse(ResponseModel):
    api_version: ApiVersion
    request_id: str
    backend: BackendInfo
    policy: DecisionPolicy | None
    results: list[DecisionResult]
    reasoning: ReasoningOptions | None = None
    error_budget: ErrorBudget | None = None

    def check_request(self, request: DecisionRequest) -> None:
        if len(self.results) != len(request.decisions) or any(
            result.id != decision.id or result.value.type != decision.kind.type
            for result, decision in zip(self.results, request.decisions)
        ):
            raise ValueError("response result count, IDs, order or kinds do not match the request")


class BatchResponse(ResponseModel):
    api_version: ApiVersion
    request_id: str
    execution: Literal["native_parallel"]
    responses: list[DecisionResponse]


class BatchCapability(ResponseModel):
    supported: bool
    enabled: bool
    execution: Literal["native_parallel"]
    max_requests: int
    max_decisions: int
    max_decisions_per_wave: int
    text: bool
    image: bool
    mixed_media: bool
    reasoning_modes: list[str]


class CapabilityBackend(ResponseModel):
    runtime: str
    model: str


class ImageCapability(ResponseModel):
    supported: bool
    max_per_decision: int
    max_bytes_each: int


class MediaCapability(ResponseModel):
    image: ImageCapability


class ReasoningCapability(ResponseModel):
    modes: list[str]
    thinking_supported: bool


class RequestPolicyCapability(ResponseModel):
    supported: bool
    target_error_rate: str


class Capabilities(ResponseModel):
    api_version: ApiVersion
    backend: CapabilityBackend
    decision_types: list[Literal["binary", "choice", "ordinal"]]
    evidence: Literal["model_scored", "selection_only"]
    media: MediaCapability
    limits: dict[str, int | float]
    reasoning: ReasoningCapability | None = None
    request_policy: RequestPolicyCapability | None = None
    batch: BatchCapability | None = None
