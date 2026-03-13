from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.base_agent import BaseStructuredAgent
from agents.drafter_agents import (
    DraftingState,
    draft_claims_node,
    drawing_analyze_node,
    extract_tech_node,
    logic_review_node,
    revise_claims_node,
    targeted_revise_spec_node,
    traceability_check_node,
    write_spec_node,
)
from models.draft_schemas import ClaimTraceabilityReport, ClaimsSet, ClaimsSetRevision, Specification, TechSummary
from models.image_schemas import DrawingMap
from models.review_schemas import ReviewReport


MAX_WORKFLOW_RETRIES = 3
MAX_CLAIM_REVISION_ROUNDS = 3


@dataclass(frozen=True)
class DraftAgentBundle:
    """Container for draft workflow agents to keep workflow wiring explicit."""

    extract_tech_agent: BaseStructuredAgent[TechSummary]
    draft_claims_agent: BaseStructuredAgent[ClaimsSet]
    traceability_agent: BaseStructuredAgent[ClaimTraceabilityReport]
    write_spec_agent: BaseStructuredAgent[Specification]
    logic_review_agent: BaseStructuredAgent[ReviewReport]
    revise_claims_agent: BaseStructuredAgent[ClaimsSetRevision] | None = None
    drawing_analyzer_agent: BaseStructuredAgent[DrawingMap] | None = None


def _merge_state_with_defaults(state: DraftingState) -> DraftingState:
    """Ensure retry-sensitive fields always exist before node execution."""
    if "error_count" not in state:
        state["error_count"] = 0
    if "review_issues" not in state:
        state["review_issues"] = []
    if "last_error" not in state:
        state["last_error"] = None
    if "disclosure_images" not in state:
        state["disclosure_images"] = []
    if "drawing_map" not in state:
        state["drawing_map"] = None
    if "vision_warnings" not in state:
        state["vision_warnings"] = []
    if "revision_instruction" not in state:
        state["revision_instruction"] = None
    if "claim_revision_count" not in state:
        state["claim_revision_count"] = 0
    if "apply_auto_claim_revision" not in state:
        state["apply_auto_claim_revision"] = None
    return state


def _record_node_error(state: DraftingState, step: str, exc: Exception) -> dict[str, Any]:
    """Convert node exception to structured state update for route-level retry decisions."""
    next_error_count = int(state.get("error_count", 0)) + 1
    retryable = next_error_count < MAX_WORKFLOW_RETRIES
    return {
        "status": "running" if retryable else "failed",
        "current_step": step,
        "error_count": next_error_count,
        "last_error": {
            "code": "E500_NODE_EXECUTION_FAILED",
            "message": str(exc),
            "retryable": retryable,
            "node": step,
        },
    }


def _extract_tech_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = extract_tech_node(state, bundle.extract_tech_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - convert to state-driven error routing.
        return _record_node_error(state, "extract_tech_node", exc)


def _draft_claims_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = draft_claims_node(state, bundle.draft_claims_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - convert to state-driven error routing.
        return _record_node_error(state, "draft_claims_node", exc)


def _drawing_analyze_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    if bundle.drawing_analyzer_agent is None:
        warnings = list(state.get("vision_warnings", []))
        warnings.append(
            {
                "code": "VISION_FALLBACK",
                "message": "Drawing analyzer agent not configured; continue without drawing map.",
                "node": "drawing_analyze_node",
            }
        )
        return {
            "drawing_map": {
                "figures": [],
                "overall_notes": "Drawing analyzer not configured.",
                "warnings": ["drawing_agent_not_configured"],
            },
            "vision_warnings": warnings,
            "current_step": "drawing_analyze_node",
            "status": "running",
            "last_error": None,
        }
    try:
        update = drawing_analyze_node(state, bundle.drawing_analyzer_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - vision failure should degrade instead of fail-fast.
        warnings = list(state.get("vision_warnings", []))
        warnings.append(
            {
                "code": "VISION_FALLBACK",
                "message": str(exc),
                "node": "drawing_analyze_node",
            }
        )
        return {
            "drawing_map": state.get("drawing_map")
            or {
                "figures": [],
                "overall_notes": "Drawing analyzer failed; fallback to text-only drafting.",
                "warnings": ["drawing_analyze_failed"],
            },
            "vision_warnings": warnings,
            "current_step": "drawing_analyze_node",
            "status": "running",
            "last_error": None,
        }


def _human_review_step(state: DraftingState) -> dict[str, Any]:
    """
    HITL interrupt node.
    If approved claims are not present, the graph is interrupted and waits for resume payload.
    Resume payload must contain {"approved_claims": <ClaimsSet dict>}.
    """
    state = _merge_state_with_defaults(state)
    if state.get("status") == "cancelled":
        return {"status": "cancelled", "current_step": "human_review_node"}

    approved_claims = state.get("approved_claims")
    if approved_claims:
        return {
            "claims": approved_claims,
            "status": "running",
            "current_step": "human_review_node",
            "last_error": None,
        }

    resumed = interrupt(
        {
            "event": "hitl_required",
            "message": "权利要求草稿与可追溯性检查已生成，请确认或修改后继续。",
            "claims": state.get("claims"),
            "drawing_map": state.get("drawing_map"),
            "claim_traceability": state.get("claim_traceability"),
            "session_id": state.get("session_id"),
            "trace_id": state.get("trace_id"),
        }
    )

    if not isinstance(resumed, dict) or "approved_claims" not in resumed:
        return _record_node_error(
            state,
            "human_review_node",
            ValueError("Missing 'approved_claims' in HITL resume payload."),
        )

    return {
        "approved_claims": resumed["approved_claims"],
        "claims": resumed["approved_claims"],
        "status": "running",
        "current_step": "human_review_node",
        "last_error": None,
    }


def _claims_revise_review_step(state: DraftingState) -> dict[str, Any]:
    """
    HITL interrupt before automatic claims revision.
    Resume payload supports:
    - {"apply_auto_claim_revision": true}: run revise_claims_node
    - {"approved_claims": <ClaimsSet dict>}: user manual claims, skip auto-revise
    """
    state = _merge_state_with_defaults(state)
    if state.get("status") == "cancelled":
        return {"status": "cancelled", "current_step": "claims_revise_review_node"}

    resumed = interrupt(
        {
            "event": "claims_revision_required",
            "message": "可追溯性审查发现部分权利要求可能超范围。请选择自动小改，或手动修改权利要求后继续。",
            "claims": state.get("claims"),
            "claim_traceability": state.get("claim_traceability"),
            "session_id": state.get("session_id"),
            "trace_id": state.get("trace_id"),
        }
    )

    if not isinstance(resumed, dict):
        return _record_node_error(
            state,
            "claims_revise_review_node",
            ValueError("Invalid claims revise review resume payload."),
        )

    if resumed.get("apply_auto_claim_revision") is True:
        return {
            "apply_auto_claim_revision": True,
            "status": "running",
            "current_step": "claims_revise_review_node",
            "last_error": None,
        }

    if "approved_claims" in resumed:
        approved_claims = resumed["approved_claims"]
        return {
            "approved_claims": approved_claims,
            "claims": approved_claims,
            "apply_auto_claim_revision": False,
            "status": "running",
            "current_step": "claims_revise_review_node",
            "last_error": None,
        }

    return _record_node_error(
        state,
        "claims_revise_review_node",
        ValueError("Missing 'apply_auto_claim_revision' or 'approved_claims' in resume payload."),
    )


def _spec_review_step(state: DraftingState) -> dict[str, Any]:
    """
    HITL interrupt for specification issues found by logic_review.
    Resume payload can either:
    - provide {"approved_specification": <Specification dict>} for manual edits, or
    - confirm {"apply_targeted_revision": true} to run minimal targeted revision.
    """
    state = _merge_state_with_defaults(state)
    if state.get("status") == "cancelled":
        return {"status": "cancelled", "current_step": "spec_review_node"}

    approved_specification = state.get("approved_specification")
    if approved_specification:
        return {
            "specification": approved_specification,
            "review_issues": [],
            "status": "running",
            "current_step": "spec_review_node",
            "last_error": None,
        }

    resumed = interrupt(
        {
            "event": "spec_review_required",
            "message": "说明书审校发现问题，请修改说明书后确认继续。",
            "review_issues": state.get("review_issues", []),
            "specification": state.get("specification"),
            "session_id": state.get("session_id"),
            "trace_id": state.get("trace_id"),
        }
    )

    if not isinstance(resumed, dict):
        return _record_node_error(
            state,
            "spec_review_node",
            ValueError("Invalid spec review resume payload."),
        )

    if "approved_specification" in resumed:
        return {
            "approved_specification": resumed["approved_specification"],
            "specification": resumed["approved_specification"],
            "review_issues": [],
            "revision_instruction": None,
            "status": "running",
            "current_step": "spec_review_node",
            "last_error": None,
        }

    apply_targeted_revision = resumed.get("apply_targeted_revision", True)
    if apply_targeted_revision is False:
        return _record_node_error(
            state,
            "spec_review_node",
            ValueError("Missing 'approved_specification' when apply_targeted_revision is false."),
        )

    return {
        "revision_instruction": resumed.get("revision_instruction"),
        "status": "running",
        "current_step": "spec_review_node",
        "last_error": None,
    }


def _write_spec_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = write_spec_node(state, bundle.write_spec_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - convert to state-driven error routing.
        return _record_node_error(state, "write_spec_node", exc)


def _targeted_revise_spec_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = targeted_revise_spec_node(state, bundle.write_spec_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - convert to state-driven error routing.
        return _record_node_error(state, "targeted_revise_spec_node", exc)


def _traceability_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = traceability_check_node(state, bundle.traceability_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - convert to state-driven error routing.
        return _record_node_error(state, "traceability_check_node", exc)


def _revise_claims_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        revise_agent = bundle.revise_claims_agent or cast(BaseStructuredAgent[ClaimsSetRevision], bundle.draft_claims_agent)
        update = revise_claims_node(state, revise_agent)
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - convert to state-driven error routing.
        return _record_node_error(state, "revise_claims_node", exc)


def _logic_review_step(state: DraftingState, bundle: DraftAgentBundle) -> dict[str, Any]:
    state = _merge_state_with_defaults(state)
    try:
        update = logic_review_node(state, bundle.logic_review_agent)
        # Count rewrite attempts when review reports issues to avoid infinite loops.
        issues = update.get("review_issues") or []
        if issues:
            update["error_count"] = int(state.get("error_count", 0)) + 1
        update["last_error"] = None
        return update
    except Exception as exc:  # noqa: BLE001 - convert to state-driven error routing.
        return _record_node_error(state, "logic_review_node", exc)


def _route_after_extract(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_extract"
    if state.get("status") == "failed":
        return "end"
    return "to_drawing_analyze"


def _route_after_draft_claims(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_draft_claims"
    if state.get("status") == "failed":
        return "end"
    return "to_traceability"


def _route_after_drawing_analyze(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_drawing_analyze"
    if state.get("status") == "failed":
        return "end"
    return "to_draft_claims"


def _route_after_traceability(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_traceability"
    if state.get("status") == "failed":
        return "end"
    report = state.get("claim_traceability")
    has_unsupported = False
    if isinstance(report, dict):
        reports = report.get("reports", [])
        if isinstance(reports, list):
            for item in reports:
                if not isinstance(item, dict):
                    continue
                if item.get("is_fully_supported") is False:
                    has_unsupported = True
                    break
                evidence_items = item.get("elements_evidence", [])
                if isinstance(evidence_items, list):
                    for evidence in evidence_items:
                        if not isinstance(evidence, dict):
                            continue
                        level = str(evidence.get("support_level", "")).strip().lower()
                        if level == "unsupported":
                            has_unsupported = True
                            break
                if has_unsupported:
                    break

    if not has_unsupported:
        return "to_human_review"
    if int(state.get("claim_revision_count", 0)) >= MAX_CLAIM_REVISION_ROUNDS:
        return "to_human_review_maxed"
    return "to_claims_revise_review"


def _route_after_revise_claims(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_revise_claims"
    if state.get("status") == "failed":
        return "end"
    return "to_traceability"


def _route_after_claims_revise_review(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_claims_revise_review"
    if state.get("status") == "failed":
        return "end"
    if state.get("apply_auto_claim_revision") is True:
        return "to_revise_claims"
    return "to_write_spec"


def _route_after_human_review(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("status") == "waiting_human":
        # Safety fallback: if caller mutates status externally to waiting_human, keep waiting.
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_human_review"
    if state.get("status") == "failed":
        return "end"
    return "to_write_spec"


def _route_after_write_spec(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_write_spec"
    if state.get("status") == "failed":
        return "end"
    return "to_logic_review"


def _route_after_logic_review(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_logic_review"
    if state.get("status") == "failed":
        return "end"

    issues = state.get("review_issues") or []
    if len(issues) == 0:
        return "done"
    return "to_spec_review"


def _route_after_spec_review(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_spec_review"
    if state.get("status") == "failed":
        return "end"
    return "to_targeted_revise_spec"


def _route_after_targeted_revise_spec(state: DraftingState) -> str:
    if state.get("status") == "cancelled":
        return "end"
    if state.get("last_error") and int(state.get("error_count", 0)) < MAX_WORKFLOW_RETRIES:
        return "retry_targeted_revise_spec"
    if state.get("status") == "failed":
        return "end"
    return "to_logic_review"


def build_draft_workflow(bundle: DraftAgentBundle, checkpointer: Any | None = None):
    """
    Build and compile the LangGraph workflow for drafting.
    Graph topology follows TAD section 5.1 with retry loops and HITL interrupt.
    """
    graph = StateGraph(DraftingState)

    graph.add_node("extract_tech", lambda state: _extract_tech_step(state, bundle))
    graph.add_node("draft_claims", lambda state: _draft_claims_step(state, bundle))
    graph.add_node("drawing_analyze", lambda state: _drawing_analyze_step(state, bundle))
    graph.add_node("traceability_check", lambda state: _traceability_step(state, bundle))
    graph.add_node("claims_revise_review", _claims_revise_review_step)
    graph.add_node("revise_claims", lambda state: _revise_claims_step(state, bundle))
    graph.add_node("human_review", _human_review_step)
    graph.add_node("write_spec", lambda state: _write_spec_step(state, bundle))
    graph.add_node("targeted_revise_spec", lambda state: _targeted_revise_spec_step(state, bundle))
    graph.add_node("logic_review", lambda state: _logic_review_step(state, bundle))
    graph.add_node("spec_review", _spec_review_step)

    graph.add_edge(START, "extract_tech")

    graph.add_conditional_edges(
        "extract_tech",
        _route_after_extract,
        {
            "retry_extract": "extract_tech",
            "to_drawing_analyze": "drawing_analyze",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "draft_claims",
        _route_after_draft_claims,
        {
            "retry_draft_claims": "draft_claims",
            "to_traceability": "traceability_check",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "drawing_analyze",
        _route_after_drawing_analyze,
        {
            "retry_drawing_analyze": "drawing_analyze",
            "to_draft_claims": "draft_claims",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "traceability_check",
        _route_after_traceability,
        {
            "retry_traceability": "traceability_check",
            "to_claims_revise_review": "claims_revise_review",
            "to_human_review": "human_review",
            "to_human_review_maxed": "human_review",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "claims_revise_review",
        _route_after_claims_revise_review,
        {
            "retry_claims_revise_review": "claims_revise_review",
            "to_revise_claims": "revise_claims",
            "to_write_spec": "write_spec",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "revise_claims",
        _route_after_revise_claims,
        {
            "retry_revise_claims": "revise_claims",
            "to_traceability": "traceability_check",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "human_review",
        _route_after_human_review,
        {
            "retry_human_review": "human_review",
            "to_write_spec": "write_spec",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "write_spec",
        _route_after_write_spec,
        {
            "retry_write_spec": "write_spec",
            "to_logic_review": "logic_review",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "logic_review",
        _route_after_logic_review,
        {
            "retry_logic_review": "logic_review",
            "to_spec_review": "spec_review",
            "done": END,
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "spec_review",
        _route_after_spec_review,
        {
            "retry_spec_review": "spec_review",
            "to_targeted_revise_spec": "targeted_revise_spec",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "targeted_revise_spec",
        _route_after_targeted_revise_spec,
        {
            "retry_targeted_revise_spec": "targeted_revise_spec",
            "to_logic_review": "logic_review",
            "end": END,
        },
    )

    return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()


def resume_draft_workflow(
    graph: Any,
    *,
    thread_id: str,
    resume_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Resume helper for HITL interrupt.
    Caller should persist the returned state snapshot.
    """
    return graph.invoke(
        Command(resume=resume_payload),
        config={"configurable": {"thread_id": thread_id}},
    )


def cancel_draft_workflow(graph: Any, *, thread_id: str) -> dict[str, Any]:
    """
    Cooperative cancel helper.
    Workflow nodes/routes already respect `status == cancelled`.
    """
    return graph.invoke(
        {"status": "cancelled"},
        config={"configurable": {"thread_id": thread_id}},
    )

