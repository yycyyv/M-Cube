from __future__ import annotations

import time
import re
from typing import Any, TypedDict

from models.draft_schemas import ClaimTraceabilityReport, ClaimsSet, ClaimsSetRevision, Specification, TechSummary
from models.image_schemas import DrawingMap, ImageAsset
from models.review_schemas import ReviewReport
from prompts.spec_writer_prompt import build_write_spec_prompt

from .base_agent import BaseStructuredAgent
from .drawing_analyzer_agent import run_drawing_analyzer


class DraftingState(TypedDict, total=False):
    session_id: str
    trace_id: str
    status: str
    disclosure_text: str
    disclosure_images: list[dict[str, Any]]
    tech_summary: dict[str, Any] | None
    claims: dict[str, Any] | None
    drawing_map: dict[str, Any] | None
    claim_traceability: dict[str, Any] | None
    approved_claims: dict[str, Any] | None
    approved_specification: dict[str, Any] | None
    specification: dict[str, Any] | None
    vision_warnings: list[dict[str, Any]]
    review_issues: list[dict[str, Any]]
    revision_instruction: str | None
    apply_auto_claim_revision: bool | None
    current_step: str
    error_count: int
    claim_revision_count: int
    last_error: dict[str, Any] | None
    model_name: str
    node_latency_ms: int


def _duration_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def extract_tech_node(
    state: DraftingState,
    agent: BaseStructuredAgent[TechSummary],
) -> dict[str, Any]:
    """Extract technical problem/solution/effects from disclosure text."""
    started_at = time.perf_counter()
    prompt = (
        "You are an expert patent agent. Your task is to perform an EXHAUSTIVE and HIGHLY DETAILED extraction "
        "of technical problems, technical solution, and beneficial effects from the disclosure text.\n\n"
        "### STRICT EXTRACTION RULES:\n"
        "1. COMPREHENSIVENESS: Do not provide a high-level summary only. You MUST extract every hardware component, "
        "software module, structural relationship, control logic, and method step that appears in the source text.\n"
        "2. PROBLEM BREAKDOWN: Identify the main technical problem and all secondary/sub-problems.\n"
        "3. EFFECT MAPPING: For each detailed feature, explain how it solves a sub-problem and what specific effect it produces.\n"
        "4. QUOTE-FIRST: Fill source_quotes first by quoting key original segments, then derive structured analysis from these quotes.\n"
        "5. NO HALLUCINATIONS: Do not invent terms, parameters, or scenarios not present in the disclosure.\n"
        "6. LANGUAGE: All JSON content must be in Simplified Chinese.\n"
        "7. OUTPUT: Return valid JSON strictly matching the schema. No markdown, no explanation.\n\n"
        "### FIELD FILLING REQUIREMENTS:\n"
        "- source_quotes: include as many relevant original snippets as possible.\n"
        "- background_and_core_problems: at least 3 detailed problem points.\n"
        "- core_solution_overview: one detailed paragraph explaining end-to-end technical route.\n"
        "- detailed_features: exhaustive list of features; each item must include feature_name, "
        "detailed_structure_or_step, solved_sub_problem, specific_effect.\n"
        "- overall_advantages: at least 3 global beneficial effects.\n\n"
        "Return valid JSON only.\n\n"
        f"[DISCLOSURE_TEXT]\n{state['disclosure_text']}"
    )
    result = agent.run_structured(prompt=prompt, output_model=TechSummary)
    return {
        "tech_summary": result.model_dump(),
        "current_step": "extract_tech_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def draft_claims_node(
    state: DraftingState,
    agent: BaseStructuredAgent[ClaimsSet],
) -> dict[str, Any]:
    """Generate independent/dependent claims based on extracted tech summary."""
    started_at = time.perf_counter()
    if state.get("tech_summary") is None:
        raise ValueError("tech_summary is required before drafting claims.")
    drawing_map = state.get("drawing_map", "None provided.")
    prompt = (
        "You are an expert Chinese patent attorney. Your task is to draft a highly professional set of "
        "patent claims based on the provided technical summary.\n\n"
        "### CLAIM DRAFTING RULES:\n"
        "1. HIERARCHY: Start with a broad Independent Claim (Claim 1) covering the core inventive concept. "
        "Follow with multiple Dependent Claims that progressively narrow the scope by adding specific structural, "
        "logical, or methodological details.\n"
        "2. ANTECEDENT BASIS: Ensure strict antecedent basis. Introduce a new technical element with "
        "'一个' or '一种', and subsequently refer back to it strictly using '所述' or '该'.\n"
        "3. FEATURE BREAKDOWN: Break down the independent claim into distinct structural or functional "
        "elements. Do not lump everything into a single massive paragraph.\n"
        "4. NO NEW MATTER: Use ONLY the technical elements disclosed in the technical summary. "
        "Do not invent unmentioned components.\n"
        "5. DRAWING HINTS: If a DRAWING_MAP is provided below, use it exclusively to understand correct structural "
        "relationships and connectivity between parts.\n"
        "6. LANGUAGE: All generated content within the JSON MUST be in professional Chinese (简体中文) patent terminology.\n\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[TECH_SUMMARY]\n{state['tech_summary']}\n\n"
        f"[DRAWING_MAP]\n{drawing_map}"
    )
    result = agent.run_structured(prompt=prompt, output_model=ClaimsSet)
    return {
        "claims": result.model_dump(),
        "current_step": "draft_claims_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def drawing_analyze_node(
    state: DraftingState,
    agent: BaseStructuredAgent[DrawingMap],
) -> dict[str, Any]:
    """
    Analyze disclosure drawings and map figure numerals/relations for downstream spec writing.
    This node is non-blocking: if images are missing, it returns warning-friendly empty DrawingMap.
    """
    started_at = time.perf_counter()
    image_payload = state.get("disclosure_images", [])
    disclosure_images = [ImageAsset.model_validate(item) for item in image_payload]
    drawing_map = run_drawing_analyzer(
        disclosure_text=state["disclosure_text"],
        disclosure_images=disclosure_images,
        agent=agent,
    )
    warnings = list(state.get("vision_warnings", []))
    for item in drawing_map.warnings:
        warnings.append({"code": "VISION_WARNING", "message": item, "node": "drawing_analyze_node"})
    return {
        "drawing_map": drawing_map.model_dump(),
        "vision_warnings": warnings,
        "current_step": "drawing_analyze_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def traceability_check_node(
    state: DraftingState,
    agent: BaseStructuredAgent[ClaimTraceabilityReport],
) -> dict[str, Any]:
    """Map each drafted claim to direct evidence snippets for human validation."""
    started_at = time.perf_counter()
    if state.get("claims") is None:
        raise ValueError("claims is required before traceability check.")
    prompt = (
        "You are a strict Patent Quality Control Auditor. Your task is to perform an "
        "Element-by-Element Traceability Analysis mapping the drafted claims back to "
        "the original disclosure text.\n\n"
        "### AUDIT RULES:\n"
        "1. ELEMENT BREAKDOWN: For each claim, break it down into individual technical features "
        "(components, steps, conditions).\n"
        "2. VERBATIM EXTRACTION: For EACH feature, find the EXACT quoting from the disclosure text "
        "that serves as its antecedent basis. Do not paraphrase the evidence.\n"
        "3. HALLUCINATION DETECTION: If a feature in the claim CANNOT be found in the disclosure text, "
        "you MUST flag it as 'Unsupported'.\n"
        "4. EXPLICIT vs IMPLICIT: Distinguish whether the support is 'Explicit' (word-for-word match) or "
        "'Implicit' (logically inferred from context).\n"
        "5. LANGUAGE: The analysis content within the JSON MUST be in Chinese.\n\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[DISCLOSURE_TEXT]\n{state['disclosure_text']}\n\n"
        f"[DRAFTED_CLAIMS]\n{state['claims']}"
    )
    result = agent.run_structured(prompt=prompt, output_model=ClaimTraceabilityReport)
    return {
        "claim_traceability": result.model_dump(),
        "current_step": "traceability_check_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def revise_claims_node(
    state: DraftingState,
    agent: BaseStructuredAgent[ClaimsSetRevision],
) -> dict[str, Any]:
    """
    Auto-correct claims based on traceability report.
    Removes or generalizes unsupported features to align with disclosure.
    """
    started_at = time.perf_counter()
    if not state.get("claims") or not state.get("claim_traceability"):
        raise ValueError("Missing claims or traceability report for revision.")

    prompt = (
        "You are an Expert Patent Attorney revising a drafted set of claims based on a rigorous Traceability Audit Report.\n\n"
        "### REVISION RULES:\n"
        "1. SURGICAL FIXES: Focus strictly on features marked 'Unsupported' or claims marked not fully supported.\n"
        "2. DELETION OR GENERALIZATION: Delete unsupported features, or generalize them to wording explicitly supported by DISCLOSURE_TEXT.\n"
        "3. NO NEW MATTER: Do NOT add any components, functions, or steps not explicitly disclosed.\n"
        "4. MAINTAIN DEPENDENCY: Keep claim dependencies logical and grammatically valid after edits.\n"
        "5. CLAIM COUNT LOCK: You MUST preserve the original number of claims and keep the original claim numbers. "
        "Do NOT delete any claim. If a claim has unsupported detail, rewrite/generalize it to supported wording instead.\n"
        "6. OUTPUT INTEGRITY: Each claim MUST include at least one elements item, and full_text must be a complete sentence.\n"
        "7. LANGUAGE: Output must be in professional Chinese.\n\n"
        "Return the fully revised set of claims as valid JSON matching the schema.\n\n"
        f"[DISCLOSURE_TEXT]\n{state['disclosure_text']}\n\n"
        f"[CURRENT_CLAIMS]\n{state['claims']}\n\n"
        f"[TRACEABILITY_REPORT]\n{state['claim_traceability']}"
    )
    result = agent.run_structured(prompt=prompt, output_model=ClaimsSetRevision)
    original_claims = state.get("claims")
    if not isinstance(original_claims, dict):
        raise ValueError("claims must be a dict before revision.")
    claims = _normalize_revised_claims(result, original_claims)
    current_revisions = int(state.get("claim_revision_count", 0)) + 1
    return {
        "claims": claims,
        "current_step": "revise_claims_node",
        "status": "running",
        "claim_revision_count": current_revisions,
        "node_latency_ms": _duration_ms(started_at),
    }


_ELEMENT_SPLIT_RE = re.compile(r"[；;。]\s*")


def _normalize_revised_claims(revised: ClaimsSetRevision, original_claims: dict[str, Any]) -> dict[str, Any]:
    original = ClaimsSet.model_validate(original_claims)
    revised_map = {int(item.claim_number): item for item in revised.claims}

    rows: list[dict[str, Any]] = []
    for orig in original.claims:
        cand = revised_map.get(int(orig.claim_number))

        preamble = str(cand.preamble).strip() if cand and str(cand.preamble).strip() else orig.preamble
        transition = str(cand.transition).strip() if cand and str(cand.transition).strip() else orig.transition
        full_text = str(cand.full_text).strip() if cand else ""
        elements = [str(item).strip() for item in (cand.elements if cand else []) if str(item).strip()]

        if not elements and full_text:
            parts = [seg.strip() for seg in _ELEMENT_SPLIT_RE.split(full_text) if seg.strip()]
            elements = [seg for seg in parts if len(seg) >= 6]
        if not elements:
            elements = list(orig.elements)
        if not transition:
            transition = orig.transition or "其特征在于，包括："
        if not preamble:
            preamble = orig.preamble or "一种方法"
        if len(full_text) < 20:
            full_text = f"{preamble}，{transition}{'；'.join(elements)}。"

        claim_type = str(cand.claim_type).strip().lower() if cand else str(orig.claim_type).strip().lower()
        depends_on = [int(dep) for dep in (cand.depends_on if cand else orig.depends_on) if int(dep) > 0]

        if int(orig.claim_number) == 1:
            claim_type = "independent"
            depends_on = []
        else:
            if claim_type not in {"independent", "dependent"}:
                claim_type = orig.claim_type
            if not depends_on:
                depends_on = list(orig.depends_on) or [1]

        rows.append(
            {
                "claim_number": int(orig.claim_number),
                "claim_type": "independent" if claim_type == "independent" else "dependent",
                "depends_on": depends_on,
                "preamble": preamble,
                "transition": transition,
                "elements": elements,
                "full_text": full_text,
            }
        )

    claims_payload = {"claims": rows}
    return ClaimsSet.model_validate(claims_payload).model_dump()


def human_review_node(state: DraftingState) -> dict[str, Any]:
    """
    HITL interrupt node prototype.
    Workflow should pause here and wait for approved claims from external API.
    """
    return {
        "current_step": "human_review_node",
        "status": "waiting_human",
    }


def write_spec_node(
    state: DraftingState,
    agent: BaseStructuredAgent[Specification],
) -> dict[str, Any]:
    """Generate specification sections from approved claims + disclosure text + optional drawing map."""
    started_at = time.perf_counter()
    if state.get("claims") is None:
        raise ValueError("claims is required before writing specification.")
    drawing_map = state.get("drawing_map")
    prompt = build_write_spec_prompt(
        disclosure_text=state["disclosure_text"],
        tech_summary=state.get("tech_summary"),
        claims=state["claims"],
        drawing_map=drawing_map,
    )
    result = agent.run_structured(prompt=prompt, output_model=Specification)
    return {
        "specification": result.model_dump(),
        "current_step": "write_spec_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def targeted_revise_spec_node(
    state: DraftingState,
    agent: BaseStructuredAgent[Specification],
) -> dict[str, Any]:
    """
    Apply minimal spec edits only for review issues, preserving unaffected sections.
    """
    started_at = time.perf_counter()
    if state.get("claims") is None:
        raise ValueError("claims is required before targeted spec revision.")
    if state.get("specification") is None:
        raise ValueError("specification is required before targeted spec revision.")

    revision_instruction = str(state.get("revision_instruction") or "").strip()
    prompt = (
        "You are an Expert Patent Specification Editor. Your mission is to fix the missing support issues in the current specification.\n\n"
        "### REVISION RULES (CRITICAL):\n"
        "1. THE CLAIMS ARE THE ABSOLUTE TRUTH: The [APPROVED_CLAIMS] have already been finalized. "
        "Your sole job is to ensure the [CURRENT_SPECIFICATION] provides 100% literal and technical support for them.\n"
        "2. EXECUTE THE PATCH: Strictly follow the instructions in [REVIEW_ISSUES]. If the issue says to add "
        "a specific paragraph about a feature, YOU MUST ADD IT.\n"
        "3. PRESERVE THE REST: Make surgical, minimal edits. Keep sections with no issues exactly as they were.\n"
        "4. Output the COMPLETE revised Specification as valid JSON matching the schema.\n\n"
        "Return valid JSON only.\n\n"
        f"[APPROVED_CLAIMS]\n{state['claims']}\n\n"
        f"[CURRENT_SPECIFICATION]\n{state['specification']}\n\n"
        f"[REVIEW_ISSUES (YOUR REVISION DIRECTIVE)]\n{state.get('review_issues', [])}\n\n"
        f"[REVISION_INSTRUCTION]\n{revision_instruction}\n\n"
        f"[DISCLOSURE_TEXT]\n{state['disclosure_text']}"
    )
    result = agent.run_structured(prompt=prompt, output_model=Specification)
    return {
        "specification": result.model_dump(),
        "current_step": "targeted_revise_spec_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def logic_review_node(
    state: DraftingState,
    agent: BaseStructuredAgent[ReviewReport],
) -> dict[str, Any]:
    """
    LLM-based logic review node.
    The model performs semantic support checks between claims and specification.
    """
    started_at = time.perf_counter()
    if state.get("claims") is None:
        raise ValueError("claims is required before logic review.")
    if state.get("specification") is None:
        raise ValueError("specification is required before logic review.")

    prompt = (
        "You are an expert Patent Examiner. Compare the drafted [CLAIMS] against the generated [SPECIFICATION].\n"
        "Your task is to identify if any technical features in the claims are COMPLETELY MISSING or UNSUPPORTED by the specification.\n\n"
        "### RULES:\n"
        "1. Ignore minor wording differences (e.g., 'processor' vs 'processing module').\n"
        "2. If you find an unsupported feature, you MUST provide a precise `patch_instruction`. "
        "Tell the downstream editor EXACTLY what sentences to add and WHERE to insert them in the Specification to fix the gap.\n"
        "3. Set issue_type to 'UNSUPPORTED_CLAIM_FEATURE' for unsupported findings.\n"
        "4. If the specification fully supports all claims, return an empty issues list [].\n\n"
        "5. LANGUAGE MANDATE: All natural-language fields MUST be Simplified Chinese, including "
        "`claim_reference`, `patch_instruction`, `description`, and `suggestion`. "
        "Do NOT output English in these fields.\n\n"
        "Return the issues as JSON matching the output schema.\n\n"
        f"[CLAIMS]\n{state['claims']}\n\n"
        f"[SPECIFICATION]\n{state['specification']}"
    )
    result = agent.run_structured(prompt=prompt, output_model=ReviewReport)
    issues = [issue.model_dump() for issue in result.issues]

    passed = len(issues) == 0
    return {
        "review_issues": issues,
        "current_step": "logic_review_node",
        "status": "completed" if passed else "running",
        "node_latency_ms": _duration_ms(started_at),
    }
