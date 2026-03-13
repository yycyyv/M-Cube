from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.base_agent import BaseStructuredAgent
from agents.polish_agents import (
    PolishState,
    adversarial_reviewer_node,
    claim_architect_node,
    diagnostic_analyzer_node,
    specification_amplifier_node,
    synergy_miner_node,
)
from models.polish_schemas import (
    AdversarialReviewReport,
    AmplifiedSpecification,
    ClaimArchitecturePlan,
    DiagnosticReport,
    SynergyVault,
)


MAX_WORKFLOW_RETRIES = 3
MAX_REVISION_LOOPS = 2


@dataclass(frozen=True)
class PolishAgentBundle:
    diagnostic_agent: BaseStructuredAgent[DiagnosticReport]
    synergy_miner_agent: BaseStructuredAgent[SynergyVault]
    claim_architect_agent: BaseStructuredAgent[ClaimArchitecturePlan]
    specification_amplifier_agent: BaseStructuredAgent[AmplifiedSpecification]
    adversarial_reviewer_agent: BaseStructuredAgent[AdversarialReviewReport]


def _merge_state_with_defaults(state: PolishState) -> PolishState:
    if "error_count" not in state:
        state["error_count"] = 0
    if "tool_error_count" not in state:
        state["tool_error_count"] = 0
    if "last_error" not in state:
        state["last_error"] = None
    if "polish_revision_count" not in state:
        state["polish_revision_count"] = 0
    return state


def _record_node_error(state: PolishState, step: str, exc: Exception) -> dict[str, Any]:
    next_count = int(state.get("error_count", 0)) + 1
    retryable = next_count < MAX_WORKFLOW_RETRIES
    return {
        "error_count": next_count,
        "status": "running" if retryable else "failed",
        "current_step": step,
        "last_error": {
            "code": "E500_NODE_EXECUTION_FAILED",
            "message": str(exc),
            "retryable": retryable,
            "node": step,
        },
    }


def _diagnostic_step(state: PolishState, bundle: PolishAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = diagnostic_analyzer_node(state, bundle.diagnostic_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "multimodal_diagnostic_analyzer_node", exc)


def _synergy_step(state: PolishState, bundle: PolishAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = synergy_miner_node(state, bundle.synergy_miner_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "multimodal_synergy_miner_node", exc)


def _claim_architect_step(state: PolishState, bundle: PolishAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = claim_architect_node(state, bundle.claim_architect_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "claim_architect_node", exc)


def _spec_amplifier_step(state: PolishState, bundle: PolishAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = specification_amplifier_node(state, bundle.specification_amplifier_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "specification_amplifier_node", exc)


def _adversarial_step(state: PolishState, bundle: PolishAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = adversarial_reviewer_node(state, bundle.adversarial_reviewer_agent, max_revision_loops=MAX_REVISION_LOOPS)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001
        return _record_node_error(state, "multimodal_adversarial_reviewer_node", exc)


def _route_with_retry(state: PolishState, done: str) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry"
    if state.get("status") == "failed":
        return "end"
    return done


def _route_after_review(state: PolishState) -> str:
    if state.get("status") in {"failed", "cancelled"}:
        return "end"
    report = state.get("adversarial_review_report") or {}
    if bool(report.get("pass_gate")) or state.get("polish_final_package") is not None:
        return "done"
    return "loop"


def build_polish_workflow(bundle: PolishAgentBundle, checkpointer: Any | None = None):
    graph = StateGraph(PolishState)

    graph.add_node("diagnostic_analyzer", lambda state: _diagnostic_step(state, bundle))
    graph.add_node("synergy_miner", lambda state: _synergy_step(state, bundle))
    graph.add_node("claim_architect", lambda state: _claim_architect_step(state, bundle))
    graph.add_node("specification_amplifier", lambda state: _spec_amplifier_step(state, bundle))
    graph.add_node("adversarial_reviewer", lambda state: _adversarial_step(state, bundle))

    graph.add_edge(START, "diagnostic_analyzer")
    graph.add_conditional_edges(
        "diagnostic_analyzer",
        lambda s: _route_with_retry(s, "to_synergy"),
        {"retry": "diagnostic_analyzer", "to_synergy": "synergy_miner", "end": END},
    )
    graph.add_conditional_edges(
        "synergy_miner",
        lambda s: _route_with_retry(s, "to_claim"),
        {"retry": "synergy_miner", "to_claim": "claim_architect", "end": END},
    )
    graph.add_conditional_edges(
        "claim_architect",
        lambda s: _route_with_retry(s, "to_spec"),
        {"retry": "claim_architect", "to_spec": "specification_amplifier", "end": END},
    )
    graph.add_conditional_edges(
        "specification_amplifier",
        lambda s: _route_with_retry(s, "to_review"),
        {"retry": "specification_amplifier", "to_review": "adversarial_reviewer", "end": END},
    )
    graph.add_conditional_edges(
        "adversarial_reviewer",
        _route_after_review,
        {"loop": "claim_architect", "done": END, "end": END},
    )
    return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()
