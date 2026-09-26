"""L2S1 Python toolkit, compatible with @l2s1/node's HTTP v1 contract."""
from pydantic import JsonValue
from .backend import BatchDecisionBackend, DecisionBackend, L2S1Error
from .engine import L2S1, PreparedDecision
from .http import L2S1Client
from .models import (
    BackendInfo, BatchCapability, BatchResponse, BinaryKind, BinaryValue, Capabilities, CapabilityBackend, ChoiceKind,
    ChoiceValue, Decision, DecisionKind, DecisionPolicy, DecisionRequest, DecisionResponse,
    DecisionResult, DecisionValue, ErrorBudget, Estimate, Evidence, FailureReasonCode,
    ImageCapability, ImageMedia, Level, MediaCapability, ModelScoredEvidence, OptionScore,
    OptionSpec, OrdinalKind, OrdinalValue, ReasoningCapability, ReasoningOptions, ReasoningUsage,
    ReasonMessage, RequestPolicyCapability, SelectionOnlyEvidence, Usage,
)
from .native import LoadOptions, RustProcessBackend

__version__ = "0.1.1"
__all__ = [
    "BatchDecisionBackend", "BatchCapability", "BatchResponse",
    "L2S1", "PreparedDecision", "L2S1Client", "DecisionBackend", "L2S1Error", "LoadOptions", "RustProcessBackend", "JsonValue",
    "BackendInfo", "BinaryKind", "BinaryValue", "Capabilities", "CapabilityBackend", "ChoiceKind", "ChoiceValue",
    "Decision", "DecisionKind", "DecisionPolicy", "DecisionRequest", "DecisionResponse", "DecisionResult", "DecisionValue",
    "ErrorBudget", "Estimate", "Evidence", "FailureReasonCode", "ImageCapability", "ImageMedia", "Level", "MediaCapability",
    "ModelScoredEvidence", "OptionScore", "OptionSpec", "OrdinalKind", "OrdinalValue", "ReasoningCapability",
    "ReasoningOptions", "ReasoningUsage", "ReasonMessage", "RequestPolicyCapability", "SelectionOnlyEvidence", "Usage",
]
