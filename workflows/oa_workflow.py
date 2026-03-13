from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents.base_agent import BaseStructuredAgent
from agents.oa_agents import (
    OAState,
    application_baseline_node,
    argument_writer_node,
    claim_amendment_node,
    concession_and_gap_node,
    fallback_feature_miner_node,
    multimodal_prior_art_agent_node,
    oa_parser_node,
    prior_art_stress_tester_node,
    response_traceability_node,
    spec_update_node,
    strategy_decision_node,
)
from models.oa_schemas import (
    ApplicationBaselineReport,
    ArgumentDraft,
    ClaimAmendmentResult,
    ConcessionGapReport,
    FallbackFeatureMiningReport,
    OADefectList,
    PriorArtStressTestReport,
    PriorArtTargetedReadingReport,
    ResponseTraceabilityReport,
    SpecUpdateNote,
    StrategyDecision,
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
class OAAgentBundle:
    oa_parser_agent: BaseStructuredAgent[OADefectList]
    multimodal_prior_art_agent: BaseStructuredAgent[PriorArtTargetedReadingReport]
    application_baseline_agent: BaseStructuredAgent[ApplicationBaselineReport]
    concession_gap_agent: BaseStructuredAgent[ConcessionGapReport]
    fallback_feature_miner_agent: BaseStructuredAgent[FallbackFeatureMiningReport]
    prior_art_stress_tester_agent: BaseStructuredAgent[PriorArtStressTestReport]
    strategy_decision_agent: BaseStructuredAgent[StrategyDecision]
    claim_amendment_agent: BaseStructuredAgent[ClaimAmendmentResult]
    argument_writer_agent: BaseStructuredAgent[ArgumentDraft]
    spec_update_agent: BaseStructuredAgent[SpecUpdateNote]
    response_traceability_agent: BaseStructuredAgent[ResponseTraceabilityReport]
    rag_service: RAGSearchService


def _merge_state_with_defaults(state: OAState) -> OAState:
    if "error_count" not in state:
        state["error_count"] = 0
    if "tool_error_count" not in state:
        state["tool_error_count"] = 0
    if "last_error" not in state:
        state["last_error"] = None
    if "retrieved_contexts" not in state:
        state["retrieved_contexts"] = []
    if "vision_warnings" not in state:
        state["vision_warnings"] = []
    return state


def _record_node_error(state: OAState, step: str, exc: Exception, *, is_tool: bool = False) -> dict[str, Any]:
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


def _oa_parser_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = oa_parser_node(state, bundle.oa_parser_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "oa_parser_node", exc)


def _multimodal_prior_art_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = multimodal_prior_art_agent_node(state, bundle.multimodal_prior_art_agent, bundle.rag_service)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "multimodal_prior_art_agent", exc, is_tool=True)


def _baseline_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = application_baseline_node(state, bundle.application_baseline_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "application_baseline_agent", exc)


def _concession_gap_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = concession_and_gap_node(state, bundle.concession_gap_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "concession_and_gap_node", exc)


def _fallback_miner_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = fallback_feature_miner_node(state, bundle.fallback_feature_miner_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "fallback_feature_miner_node", exc)


def _stress_tester_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = prior_art_stress_tester_node(state, bundle.prior_art_stress_tester_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "prior_art_stress_tester_node", exc, is_tool=True)


def _strategy_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = strategy_decision_node(state, bundle.strategy_decision_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "strategy_decision_node", exc)


def _claim_amendment_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = claim_amendment_node(state, bundle.claim_amendment_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "claim_amendment_agent", exc)


def _argument_writer_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = argument_writer_node(state, bundle.argument_writer_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "argument_writer_agent", exc)


def _spec_update_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = spec_update_node(state, bundle.spec_update_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "spec_update_agent", exc)


def _traceability_step(state: OAState, bundle: OAAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = response_traceability_node(state, bundle.response_traceability_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "response_traceability_node", exc)


def _route_with_retry(state: OAState, *, key: str, max_retry: int, done: str) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and _safe_int(state.get(key, 0), 0) < max_retry:
        return "retry"
    if state.get("status") == "failed":
        return "end"
    return done


def build_oa_workflow(bundle: OAAgentBundle, checkpointer: Any | None = None):
    graph = StateGraph(OAState)

    graph.add_node("oa_parser", lambda state: _oa_parser_step(state, bundle))
    graph.add_node("application_baseline", lambda state: _baseline_step(state, bundle))
    graph.add_node("multimodal_prior_art", lambda state: _multimodal_prior_art_step(state, bundle))
    graph.add_node("concession_and_gap", lambda state: _concession_gap_step(state, bundle))
    graph.add_node("fallback_feature_miner", lambda state: _fallback_miner_step(state, bundle))
    graph.add_node("prior_art_stress_tester", lambda state: _stress_tester_step(state, bundle))
    graph.add_node("strategy_decision", lambda state: _strategy_step(state, bundle))
    graph.add_node("claim_amendment", lambda state: _claim_amendment_step(state, bundle))
    graph.add_node("argument_writer", lambda state: _argument_writer_step(state, bundle))
    graph.add_node("spec_update", lambda state: _spec_update_step(state, bundle))
    graph.add_node("response_traceability", lambda state: _traceability_step(state, bundle))

    graph.add_edge(START, "application_baseline")
    graph.add_conditional_edges(
        "application_baseline",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_oa_parser"),
        {"retry": "application_baseline", "to_oa_parser": "oa_parser", "end": END},
    )
    graph.add_conditional_edges(
        "oa_parser",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_multimodal"),
        {"retry": "oa_parser", "to_multimodal": "multimodal_prior_art", "end": END},
    )
    graph.add_conditional_edges(
        "multimodal_prior_art",
        lambda s: _route_with_retry(s, key="tool_error_count", max_retry=MAX_TOOL_RETRIES, done="to_concession"),
        {"retry": "multimodal_prior_art", "to_concession": "concession_and_gap", "end": END},
    )
    graph.add_conditional_edges(
        "concession_and_gap",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_miner"),
        {"retry": "concession_and_gap", "to_miner": "fallback_feature_miner", "end": END},
    )
    graph.add_conditional_edges(
        "fallback_feature_miner",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_stress"),
        {"retry": "fallback_feature_miner", "to_stress": "prior_art_stress_tester", "end": END},
    )
    graph.add_conditional_edges(
        "prior_art_stress_tester",
        lambda s: _route_with_retry(s, key="tool_error_count", max_retry=MAX_TOOL_RETRIES, done="to_strategy"),
        {"retry": "prior_art_stress_tester", "to_strategy": "strategy_decision", "end": END},
    )
    graph.add_conditional_edges(
        "strategy_decision",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_claim_amendment"),
        {"retry": "strategy_decision", "to_claim_amendment": "claim_amendment", "end": END},
    )
    graph.add_conditional_edges(
        "claim_amendment",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_argument"),
        {"retry": "claim_amendment", "to_argument": "argument_writer", "end": END},
    )
    graph.add_conditional_edges(
        "argument_writer",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_spec_update"),
        {"retry": "argument_writer", "to_spec_update": "spec_update", "end": END},
    )
    graph.add_conditional_edges(
        "spec_update",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="to_traceability"),
        {"retry": "spec_update", "to_traceability": "response_traceability", "end": END},
    )
    graph.add_conditional_edges(
        "response_traceability",
        lambda s: _route_with_retry(s, key="error_count", max_retry=MAX_WORKFLOW_RETRIES, done="done"),
        {"retry": "response_traceability", "done": END, "end": END},
    )

    return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()


def resume_oa_workflow(graph: Any, *, thread_id: str, resume_payload: dict[str, Any]) -> dict[str, Any]:
    return graph.invoke(
        Command(resume=resume_payload),
        config={"configurable": {"thread_id": thread_id}},
    )


def cancel_oa_workflow(graph: Any, *, thread_id: str) -> dict[str, Any]:
    return graph.invoke(
        {"status": "cancelled"},
        config={"configurable": {"thread_id": thread_id}},
    )
