from __future__ import annotations

import time
from typing import Any, TypedDict

from agents.base_agent import BaseStructuredAgent
from models.compare_schemas import (
    AmendmentSuggestionReport,
    DraftBaselineReport,
    FeatureCollisionMatrix,
    PriorArtProfileSet,
    RiskAssessmentReport,
)
from tools.rag_search import RAGSearchService


class CompareState(TypedDict, total=False):
    session_id: str
    trace_id: str
    status: str
    current_step: str
    comparison_goal: str
    original_claims: dict[str, Any]
    application_specification: dict[str, Any]
    prior_arts_paths: list[str]
    application_images: list[dict[str, Any]]
    prior_art_images: list[dict[str, Any]]
    vision_warnings: list[dict[str, Any]]
    draft_baseline: dict[str, Any] | None
    prior_art_profiles: dict[str, Any] | None
    feature_collision_matrix: dict[str, Any] | None
    collision_matrix: dict[str, Any] | None
    risk_report: dict[str, Any] | None
    risk_assessment_report: dict[str, Any] | None
    amendment_suggestions: dict[str, Any] | None
    final_compare_report: dict[str, Any] | None
    prior_art_targeted_report: dict[str, Any] | None
    targeted_reading_audit: dict[str, Any] | None
    retrieved_contexts: list[dict[str, Any]]
    error_count: int
    tool_error_count: int
    last_error: dict[str, Any] | None
    max_reflections: int
    node_latency_ms: int


def _duration_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _allowed_doc_ids_from_paths(paths: list[str]) -> list[str]:
    if not paths:
        return ["D1"]
    return [f"D{i+1}" for i in range(len(paths))]


def _normalize_doc_id(raw: Any, allowed_doc_ids: list[str]) -> str:
    if not allowed_doc_ids:
        return "D1"
    if isinstance(raw, str):
        text = raw.strip().upper()
        if text in allowed_doc_ids:
            return text
        if text.startswith("D") and text[1:].isdigit():
            idx = int(text[1:])
            if 1 <= idx <= len(allowed_doc_ids):
                return f"D{idx}"
    return allowed_doc_ids[0]


def _sanitize_prior_art_profiles_payload(payload: dict[str, Any], allowed_doc_ids: list[str]) -> dict[str, Any]:
    profiles = payload.get("prior_art_profiles")
    if isinstance(profiles, list):
        for item in profiles:
            if isinstance(item, dict):
                item["prior_art_id"] = _normalize_doc_id(item.get("prior_art_id") or item.get("doc_id"), allowed_doc_ids)
    return payload


def _sanitize_matrix_payload(payload: dict[str, Any], allowed_doc_ids: list[str]) -> dict[str, Any]:
    reports = payload.get("prior_art_targeted_report")
    if isinstance(reports, list):
        for report in reports:
            if not isinstance(report, dict):
                continue
            collisions = report.get("feature_collisions")
            if not isinstance(collisions, list):
                continue
            for cell in collisions:
                if isinstance(cell, dict):
                    cell["prior_art_id"] = _normalize_doc_id(cell.get("prior_art_id"), allowed_doc_ids)
    return payload


def _compact_retrieved_contexts(items: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items[:limit]:
        snippet = str(item.get("text", ""))
        compact.append(
            {
                "source_path": item.get("source_path", ""),
                "score": item.get("score", 0),
                "text": snippet[:500],
            }
        )
    return compact


def multimodal_draft_parser_node(
    state: CompareState,
    agent: BaseStructuredAgent[DraftBaselineReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    claims_payload = state.get("original_claims") or {}
    spec_payload = state.get("application_specification") or {}
    claims_text = str(claims_payload.get("text", "")) if isinstance(claims_payload, dict) else str(claims_payload)
    specification_text = str(spec_payload.get("text", "")) if isinstance(spec_payload, dict) else str(spec_payload)

    app_paths = [item.get("source_path", "") for item in state.get("application_images", []) if item.get("source_path")]
    app_mimes = [item.get("mime_type", "image/png") for item in state.get("application_images", []) if item.get("source_path")]
    drawings_context = state.get("application_images", [])

    prompt = (
        "You are an Expert Chinese Patent Attorney and Multimodal Parsing Specialist.\n"
        "Your mission is to build an exhaustive, high-resolution `Application Baseline` from the provided claims, specification, and applicant drawings. You must act as a precise reverse-engineer.\n\n"
        "### PARSING RULES:\n"
        "1. TOPOLOGICAL DISSECTION OVER BOM:\n"
        "   - When breaking down claims into `atomic_features`, DO NOT just extract a Bill of Materials or a list of nouns.\n"
        "   - An un-connected part is a useless part. For EVERY feature, you MUST rigorously populate `connection_and_synergy`.\n"
        "   - Extract verbs, prepositions and logic: How A connects to B, spatial relationships, and the collaborative mechanism formed.\n"
        "2. VISUAL BINDING:\n"
        "   - For EVERY feature, extract `visual_anchor`.\n"
        "   - `visual_morphology` must vividly describe physical shape and spatial assembly state as seen in drawings.\n"
        "3. EXHAUSTIVE FALLBACK MINING:\n"
        "   - Scan the specification for embodiments with unique structural connections/operating mechanisms not in broad claims.\n"
        "   - Extract them into `fallback_feature_index` using verbatim quotes to ensure Article 33 compliance.\n"
        "4. ZERO HALLUCINATION:\n"
        "   - Keep terms and relationships strictly faithful to source text. Do not invent functionalities.\n\n"
        "Content in JSON must be in professional Chinese.\n"
        "Return valid JSON strictly matching the schema.\n\n"
        f"[CLAIMS (权利要求书文本)]\n{claims_text}\n\n"
        f"[SPECIFICATION (说明书全文)]\n{specification_text}\n\n"
        f"[DRAWINGS_CONTEXT (附图说明及视觉输入上下文)]\n{drawings_context}"
    )
    result = agent.run_structured(
        prompt=prompt,
        output_model=DraftBaselineReport,
        context={"image_paths": app_paths[:8], "image_mime_types": app_mimes[:8]},
    )
    return {
        "draft_baseline": result.model_dump(),
        "application_baseline": result.model_dump(),
        "current_step": "multimodal_draft_parser_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def multimodal_prior_art_node(
    state: CompareState,
    agent: BaseStructuredAgent[PriorArtProfileSet],
    rag_service: RAGSearchService,
    *,
    top_k: int = 12,
    min_score: float = 0.42,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    paths = state.get("prior_arts_paths", [])
    rag_service.build_index_from_paths(paths)
    retrieved = rag_service.retrieve(query="核心技术方案 结构 连接 附图 图", top_k=top_k, min_score=min_score)
    retrieved_contexts = [item.model_dump() for item in retrieved]
    allowed_doc_ids = _allowed_doc_ids_from_paths(paths)

    prior_paths = [item.get("source_path", "") for item in state.get("prior_art_images", []) if item.get("source_path")]
    prior_mimes = [item.get("mime_type", "image/png") for item in state.get("prior_art_images", []) if item.get("source_path")]

    context_paths = prior_paths[:12]
    context_mimes = prior_mimes[:12]
    current_prior_art_id = "、".join(allowed_doc_ids)
    prior_art_text = _compact_retrieved_contexts(retrieved_contexts, limit=12)
    prior_art_drawings_context = state.get("prior_art_images", [])

    prompt = (
        "You are an Expert Multimodal Prior-Art Reverse Engineer for Chinese patentability comparison.\n"
        "Your mission is to dissect the provided prior art and build `Prior-Art Profile` that captures not only WHAT parts are, but HOW THEY CONNECT AND OPERATE.\n\n"
        "### REVERSE ENGINEERING RULES:\n"
        "1. KINEMATICS OVER BOM:\n"
        "   - A pile of isolated parts is not a technical solution.\n"
        "   - For each PriorArtComponent, rigorously fill `structural_connections_and_mechanisms` with verbs, positions, and synergy logic, with paragraph coordinates.\n"
        "2. VISUAL TOPOLOGY:\n"
        "   - Analyze drawings figure-by-figure.\n"
        "   - In `FigureConnection`, fill `kinematic_relationship` to describe physical interactions (threaded, coaxial sleeve, spring preload, hinge, sliding clearance).\n"
        "3. TARGETED READING AUDIT:\n"
        "   - Count input images and analyzed images in reading_audit to prevent omission.\n"
        "4. ZERO HALLUCINATION:\n"
        "   - If connection is not explicit in text/drawing, state '未明确记载连接方式'.\n\n"
        "Content in JSON must be in professional Chinese.\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[PRIOR_ART_ID (对比文件编号)]\n{current_prior_art_id}\n\n"
        f"[PRIOR_ART_TEXT (现有技术说明书及权利要求文本)]\n{prior_art_text}\n\n"
        f"[PRIOR_ART_IMAGES_CONTEXT (现有技术附图总数及视觉输入)]\n{prior_art_drawings_context}"
    )
    result = agent.run_structured(
        prompt=prompt,
        output_model=PriorArtProfileSet,
        context={"image_paths": context_paths, "image_mime_types": context_mimes},
    )
    payload = _sanitize_prior_art_profiles_payload(result.model_dump(), allowed_doc_ids)
    aggregated_targeted_audit = {
        "input_image_count": len(prior_paths),
        "actually_used_image_count": 0,
        "omission_warning": "合规",
    }
    profiles = payload.get("prior_art_profiles")
    if isinstance(profiles, list):
        used = 0
        warnings: list[str] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            audit = profile.get("reading_audit")
            if not isinstance(audit, dict):
                continue
            used += _safe_int(audit.get("actually_used_image_count", 0), 0)
            warning_text = str(audit.get("omission_warning", "")).strip()
            if warning_text and warning_text != "合规":
                warnings.append(warning_text)
        aggregated_targeted_audit["actually_used_image_count"] = used
        if warnings:
            aggregated_targeted_audit["omission_warning"] = "；".join(warnings)
        elif used < len(prior_paths):
            aggregated_targeted_audit["omission_warning"] = (
                f"传入{len(prior_paths)}张图，仅解析{used}张，存在漏看风险"
            )
    return {
        "prior_art_profiles": payload,
        "targeted_reading_audit": aggregated_targeted_audit,
        "retrieved_contexts": retrieved_contexts,
        "current_step": "multimodal_prior_art_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def multimodal_matrix_comparison_node(
    state: CompareState,
    agent: BaseStructuredAgent[FeatureCollisionMatrix],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if state.get("draft_baseline") is None:
        raise ValueError("draft_baseline is required before matrix comparison.")
    if state.get("prior_art_profiles") is None:
        raise ValueError("prior_art_profiles is required before matrix comparison.")
    allowed_doc_ids = _allowed_doc_ids_from_paths(state.get("prior_arts_paths", []))

    app_paths = [item.get("source_path", "") for item in state.get("application_images", []) if item.get("source_path")]
    app_mimes = [item.get("mime_type", "image/png") for item in state.get("application_images", []) if item.get("source_path")]
    prior_paths = [item.get("source_path", "") for item in state.get("prior_art_images", []) if item.get("source_path")]
    prior_mimes = [item.get("mime_type", "image/png") for item in state.get("prior_art_images", []) if item.get("source_path")]

    context_paths = (app_paths[:4] + prior_paths[:10])[:14]
    context_mimes = (app_mimes[:4] + prior_mimes[:10])[:14]

    raw_contexts = state.get("retrieved_contexts", [])
    compact_contexts = _compact_retrieved_contexts(raw_contexts if isinstance(raw_contexts, list) else [])
    application_baseline = state.get("application_baseline") or state.get("draft_baseline")
    prompt = (
        "You are the core Multimodal Feature Matrix Collision Agent for Chinese patentability.\n"
        "Your mission is to perform a clinical N x M comparison between the applicant's Application Baseline and Prior-Art Profiles. You must evaluate both textual and visual evidence.\n\n"
        "### COLLISION RULES:\n"
        "1. DUAL-AXIS EVALUATION:\n"
        "   - You must separately evaluate physical parts (`component_match_status`) AND interactions (`relationship_match_status`).\n"
        "2. RELATIONSHIPS ARE VETO FACTORS:\n"
        "   - Compare applicant `connection_and_synergy` with prior-art kinematic/connection evidence.\n"
        "   - If relationship is inconsistent, disclosure_status MUST be NOT_DISCLOSED even if nouns match.\n"
        "3. DISCLOSURE STATUS DEFINITIONS:\n"
        "   - EXPLICIT: both components and connections explicitly described in text.\n"
        "   - IMPLICIT_VISUAL: text silent, but drawings clearly show identical components AND identical physical connections/synergy.\n"
        "   - NOT_DISCLOSED: components missing OR relationship/synergy fundamentally different.\n\n"
        "### ENGINEERING SAFETY RULES:\n"
        "1. JSON SAFETY: Output a valid JSON object ONLY. NO markdown fences. NO extra text.\n"
        "2. NO RAW QUOTES: Do NOT include raw double quotes (\") or multiline copied quotations in any string value.\n"
        "3. STRICT CONCISENESS: Keep EVERY string value <=120 Chinese characters.\n"
        "4. NO HALLUCINATION: only cite evidence from provided context.\n\n"
        "Content in JSON must be in formal Chinese.\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[ALLOWED_DOC_IDS]\n{allowed_doc_ids}\n\n"
        f"[APPLICATION_BASELINE (本案拓扑特征)]\n{application_baseline}\n\n"
        f"[PRIOR_ART_PROFILES]\n{state.get('prior_art_profiles')}\n\n"
        f"[RETRIEVED_CONTEXTS]\n{compact_contexts}"
    )
    result = agent.run_structured(
        prompt=prompt,
        output_model=FeatureCollisionMatrix,
        context={"image_paths": context_paths, "image_mime_types": context_mimes},
    )
    payload = _sanitize_matrix_payload(result.model_dump(), allowed_doc_ids)
    return {
        "collision_matrix": payload,
        "feature_collision_matrix": payload,
        "prior_art_targeted_report": payload.get("prior_art_targeted_report"),
        "current_step": "multimodal_matrix_comparison_agent",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def risk_assessment_node(
    state: CompareState,
    agent: BaseStructuredAgent[RiskAssessmentReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    max_reflections = _safe_int(state.get("max_reflections", 3), 3) or 3
    collision_matrix = state.get("collision_matrix") or state.get("feature_collision_matrix")
    if collision_matrix is None:
        raise ValueError("feature_collision_matrix is required before risk assessment.")

    prompt = (
        "You are an Expert Patentability Risk Assessment Agent (Senior Legal Partner) for Chinese Patents.\n"
        "Your duty is to analyze the Dual-Axis `Collision Matrix` (Node 3) and evaluate Novelty (A22.2) and Inventiveness (A22.3) risks based on TOPOLOGICAL and SYNERGISTIC differences.\n\n"
        "### CRITICAL LEGAL RULES:\n"
        "1. NOVELTY REQUIRES STRICT TOPOLOGICAL IDENTITY:\n"
        "   - Novelty is 'FATAL' only if a SINGLE prior-art document discloses all identical components and identical connections/synergies.\n"
        "   - If any feature has inconsistent `relationship_match_status`, novelty should be 'SAFE'.\n"
        "2. INVENTIVENESS AND SYNERGY:\n"
        "   - Focus on survived features (`NOT_DISCLOSED`) due to relationship differences.\n"
        "   - 'HIGH': trivial replacement/routine connection without unexpected effect.\n"
        "   - 'LOW'/'MEDIUM': changed connection/mechanism forms synergistic effect solving specific technical problem.\n"
        "3. IDENTIFY BREAKTHROUGH POINT:\n"
        "   - For each claim, identify strongest topological/mechanistic difference and output robust distinguishing features.\n"
        "4. STRATEGIC DIRECTIVE:\n"
        "   - In `strategic_amendment_direction`, provide actionable amendment direction, specifying mechanisms/relationships to emphasize.\n\n"
        f"MAX_REFLECTION_ROUNDS: {max_reflections}\n"
        "Before final risk grading, internally reflect for up to MAX_REFLECTION_ROUNDS rounds.\n\n"
        "Content in JSON must be formal Chinese.\n"
        "Return valid JSON strictly matching output schema; keep string values concise.\n\n"
        f"[COLLISION MATRIX (Node 3)]\n{collision_matrix}\n\n"
        f"[APPLICATION_BASELINE (Node 1)]\n{state.get('application_baseline') or state.get('draft_baseline')}"
    )
    result = agent.run_structured(prompt=prompt, output_model=RiskAssessmentReport)
    payload = result.model_dump()
    return {
        "risk_report": payload,
        "risk_assessment_report": payload,
        "current_step": "risk_assessment_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def amendment_suggestion_node(
    state: CompareState,
    agent: BaseStructuredAgent[AmendmentSuggestionReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    risk_payload = state.get("risk_assessment_report") or state.get("risk_report")
    if risk_payload is None:
        raise ValueError("risk_report is required before amendment suggestion.")
    baseline_payload = state.get("application_baseline") or state.get("draft_baseline")
    if baseline_payload is None:
        raise ValueError("draft_baseline is required before amendment suggestion.")

    draft_baseline = baseline_payload if isinstance(baseline_payload, dict) else {}
    fallback_index = draft_baseline.get("fallback_feature_index")
    if not fallback_index:
        fallback_index = draft_baseline.get("spec_feature_index", [])

    prompt = (
        "You are an Expert Patent Amendment Agent (The Chief Surgeon) specializing in patentability rescue.\n"
        "Your mission is to execute Node 4 `strategic_amendment_direction`, using ONLY Node 1 ammunition (fallback features and claims tree).\n\n"
        "### AMENDMENT RULES:\n"
        "1. ARTICLE 33 ABSOLUTE COMPLIANCE:\n"
        "   - Do NOT invent new components, connections, or synergies.\n"
        "   - Every amendment must come directly from fallback_feature_index or claims_tree, and provide `verbatim_addition`.\n"
        "2. SYNERGY OVER NOUNS:\n"
        "   - Prioritize unique topological connection / dynamic mechanism features over isolated nouns.\n"
        "3. EXECUTE NODE-4 DIRECTIVE:\n"
        "   - Align amendments strictly with risk diagnosis and strategic direction.\n"
        "4. DRAFT FULL CLAIM TEXT:\n"
        "   - In `draft_amended_claim_text`, provide fluent, full claim draft conforming to Chinese claim style.\n\n"
        "Content in JSON must be formal Chinese.\n"
        "Return valid JSON strictly matching schema. Keep reasoning concise.\n\n"
        f"[RISK REPORT & DIRECTIVE (Node 4)]\n{risk_payload}\n\n"
        f"[APPLICATION BASELINE (Node 1)]\n{draft_baseline}\n\n"
        f"[FALLBACK FEATURE INDEX]\n{fallback_index}"
    )
    result = agent.run_structured(prompt=prompt, output_model=AmendmentSuggestionReport)
    payload = result.model_dump()
    suggestions = payload.get("concrete_amendments", [])
    if not suggestions and isinstance(payload.get("suggestions"), list):
        suggestions = payload.get("suggestions", [])
    risk_payload = risk_payload or {}
    overall_risk_level = str(risk_payload.get("overall_risk_level", "")).strip()
    if not overall_risk_level:
        assessments = risk_payload.get("claim_assessments", [])
        novelty = []
        inventiveness = []
        if isinstance(assessments, list):
            for item in assessments:
                if not isinstance(item, dict):
                    continue
                novelty.append(str(item.get("novelty_risk", "")).upper())
                inventiveness.append(str(item.get("inventiveness_risk", "")).upper())
        if "FATAL" in novelty:
            overall_risk_level = "FATAL"
        elif "HIGH" in inventiveness:
            overall_risk_level = "HIGH"
        elif "MEDIUM" in inventiveness:
            overall_risk_level = "MEDIUM"
        else:
            overall_risk_level = "LOW"
    final_report = {
        "comparison_goal": state.get("comparison_goal", "patentability"),
        "overall_risk_level": overall_risk_level,
        "matrix_summary": (state.get("feature_collision_matrix") or {}).get("global_conclusion", ""),
        "suggestion_count": len(suggestions) if isinstance(suggestions, list) else 0,
        "final_recommendation": payload.get("overall_rescue_strategy", payload.get("final_recommendation", "")),
    }
    return {
        "amendment_suggestions": payload,
        "amendment_strategy_report": payload,
        "final_compare_report": final_report,
        "current_step": "amendment_suggestion_node",
        "status": "completed",
        "node_latency_ms": _duration_ms(started_at),
    }
