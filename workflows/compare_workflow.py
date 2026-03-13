from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.base_agent import BaseStructuredAgent
from agents.compare_agents import (
    CompareState,
    amendment_suggestion_node,
    multimodal_draft_parser_node,
    multimodal_matrix_comparison_node,
    multimodal_prior_art_node,
    risk_assessment_node,
)
from models.compare_schemas import (
    AmendmentSuggestionReport,
    DraftBaselineReport,
    FeatureCollisionMatrix,
    PriorArtProfileSet,
    RiskAssessmentReport,
)
from tools.rag_search import RAGSearchService


MAX_WORKFLOW_RETRIES = 3
MAX_TOOL_RETRIES = 2


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CompareAgentBundle:
    draft_parser_agent: BaseStructuredAgent[DraftBaselineReport]
    prior_art_parser_agent: BaseStructuredAgent[PriorArtProfileSet]
    matrix_comparison_agent: BaseStructuredAgent[FeatureCollisionMatrix]
    risk_assessment_agent: BaseStructuredAgent[RiskAssessmentReport]
    amendment_suggestion_agent: BaseStructuredAgent[AmendmentSuggestionReport]
    rag_service: RAGSearchService


def _merge_state_with_defaults(state: CompareState) -> CompareState:
    if "error_count" not in state:
        state["error_count"] = 0
    if "tool_error_count" not in state:
        state["tool_error_count"] = 0
    if "last_error" not in state:
        state["last_error"] = None
    if "vision_warnings" not in state:
        state["vision_warnings"] = []
    if "retrieved_contexts" not in state:
        state["retrieved_contexts"] = []
    return state


def _record_node_error(state: CompareState, step: str, exc: Exception, *, is_tool: bool = False) -> dict[str, Any]:
    key = "tool_error_count" if is_tool else "error_count"
    max_retry = MAX_TOOL_RETRIES if is_tool else MAX_WORKFLOW_RETRIES
    next_count = _safe_int(state.get(key, 0), 0) + 1
    retryable = next_count < max_retry
    return {
        key: next_count,
        "status": "running" if retryable else "failed",
        "current_step": step,
        "last_error": {
            "code": "E500_TOOL_FAILURE" if is_tool else "E500_NODE_EXECUTION_FAILED",
            "message": str(exc),
            "retryable": retryable,
            "node": step,
        },
    }


def _draft_parser_step(state: CompareState, bundle: CompareAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = multimodal_draft_parser_node(state, bundle.draft_parser_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "multimodal_draft_parser_node", exc)


def _prior_art_parser_step(state: CompareState, bundle: CompareAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = multimodal_prior_art_node(state, bundle.prior_art_parser_agent, bundle.rag_service)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "multimodal_prior_art_node", exc, is_tool=True)


def _matrix_step(state: CompareState, bundle: CompareAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = multimodal_matrix_comparison_node(state, bundle.matrix_comparison_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "multimodal_matrix_comparison_node", exc, is_tool=True)


def _risk_step(state: CompareState, bundle: CompareAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = risk_assessment_node(state, bundle.risk_assessment_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "risk_assessment_node", exc)


def _amendment_step(state: CompareState, bundle: CompareAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = amendment_suggestion_node(state, bundle.amendment_suggestion_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "amendment_suggestion_node", exc)


def _route_with_retry(state: CompareState, *, key: str, max_retry: int, done: str) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and _safe_int(state.get(key, 0), 0) < max_retry:
        return "retry"
    if state.get("status") == "failed":
        return "end"
    return done


def build_compare_workflow(bundle: CompareAgentBundle, checkpointer: Any | None = None):
    graph = StateGraph(CompareState)

    graph.add_node("multimodal_draft_parser", lambda state: _draft_parser_step(state, bundle))
    graph.add_node("multimodal_prior_art", lambda state: _prior_art_parser_step(state, bundle))
    graph.add_node("multimodal_matrix_comparison", lambda state: _matrix_step(state, bundle))
    graph.add_node("risk_assessment", lambda state: _risk_step(state, bundle))
    graph.add_node("amendment_suggestion", lambda state: _amendment_step(state, bundle))

    graph.add_edge(START, "multimodal_draft_parser")
    graph.add_conditional_edges(
        "multimodal_draft_parser",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_prior"),
        {"retry": "multimodal_draft_parser", "to_prior": "multimodal_prior_art", "end": END},
    )
    graph.add_conditional_edges(
        "multimodal_prior_art",
        lambda s: _route_with_retry(s, key="tool_error_count", max_retry=MAX_TOOL_RETRIES, done="to_matrix"),
        {"retry": "multimodal_prior_art", "to_matrix": "multimodal_matrix_comparison", "end": END},
    )
    graph.add_conditional_edges(
        "multimodal_matrix_comparison",
        lambda s: _route_with_retry(s, key="tool_error_count", max_retry=MAX_TOOL_RETRIES, done="to_risk"),
        {"retry": "multimodal_matrix_comparison", "to_risk": "risk_assessment", "end": END},
    )
    graph.add_conditional_edges(
        "risk_assessment",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_amend"),
        {"retry": "risk_assessment", "to_amend": "amendment_suggestion", "end": END},
    )
    graph.add_conditional_edges(
        "amendment_suggestion",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="done"),
        {"retry": "amendment_suggestion", "done": END, "end": END},
    )
    return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()
