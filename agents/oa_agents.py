from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from models.image_schemas import ImageAsset, PriorArtVisualDiff, PriorArtVisualReport
from models.oa_schemas import (
    ApplicationBaselineReport,
    ArgumentDraft,
    ClaimAmendmentResult,
    ComparisonResult,
    ConcessionGapReport,
    DebateStrategy,
    FallbackFeatureMiningReport,
    OADefectList,
    PriorArtStressTestReport,
    PriorArtTargetedReadingReport,
    RebuttalActionPlan,
    ResponseTraceabilityReport,
    SpecUpdateNote,
    StrategyDecision,
)

from .base_agent import BaseStructuredAgent
from .oa_visual_analyzer_agent import run_prior_art_visual_analyzer
from tools.rag_search import RAGSearchService


class OAState(TypedDict, total=False):
    session_id: str
    trace_id: str
    status: str
    current_step: str
    oa_text: str
    original_claims: dict[str, Any]
    application_specification: dict[str, Any]
    prior_arts_paths: list[str]
    parsed_defects: dict[str, Any] | None
    retrieved_contexts: list[dict[str, Any]]
    prior_art_targeted_report: dict[str, Any] | None
    image_recognition_report: dict[str, Any] | None
    targeted_reading_audit: dict[str, Any] | None
    application_baseline: dict[str, Any] | None
    concession_gap_report: dict[str, Any] | None
    mined_fallback_features: dict[str, Any] | None
    stress_test_report: dict[str, Any] | None
    strategy_decision: dict[str, Any] | None
    rebuttal_plan: dict[str, Any] | None
    amended_claims: dict[str, Any] | None
    argument_draft: dict[str, Any] | None
    spec_update_note: dict[str, Any] | None
    response_traceability: dict[str, Any] | None
    comparison_result: dict[str, Any] | None
    visual_report: dict[str, Any] | None
    application_images: list[dict[str, Any]]
    prior_art_images: list[dict[str, Any]]
    vision_warnings: list[dict[str, Any]]
    final_strategy: dict[str, Any] | None
    final_reply_text: str | None
    error_count: int
    tool_error_count: int
    last_error: dict[str, Any] | None
    node_latency_ms: int
    max_reflections: int


def _duration_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def oa_parser_node(
    state: OAState,
    agent: BaseStructuredAgent[OADefectList],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    baseline = state.get("application_baseline") or {}
    claims_tree = baseline.get("claims_tree") if isinstance(baseline, dict) else {}
    if not claims_tree:
        raise ValueError("claims_tree is required before OA parsing.")
    prompt = (
        "You are an Expert Chinese Patent Examiner Assistant. Your task is to deep-parse the provided Office Action "
        "(OA) notice into highly structured, machine-actionable defect items.\n\n"
        "PARSING RULES:\n"
        "1. CLAIM & FEATURE RESOLUTION:\n"
        "   - Use the provided [CLAIMS_TREE] to map examiner arguments to specific claims and atomic features.\n"
        "   - Do NOT only summarize overall rejection; break down reasoning feature-by-feature.\n"
        "2. EXACT COORDINATES:\n"
        "   - For each attacked feature, extract exact prior-art doc id, cited paragraphs, and cited figures.\n"
        "   - Do not hallucinate coordinates.\n"
        "3. EXAMINER'S LOGIC:\n"
        "   - Capture how examiner maps prior art to application features.\n"
        "4. COMBINATION MOTIVATION:\n"
        "   - For combined references, extract stated motivation for combination; otherwise use 无.\n"
        "5. LANGUAGE: All output content in JSON MUST be Chinese.\n\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[CLAIMS_TREE (Application Baseline)]\n{claims_tree}\n\n"
        f"[OFFICE_ACTION_TEXT]\n{state['oa_text']}"
    )
    defects = agent.run_structured(prompt=prompt, output_model=OADefectList)
    return {
        "parsed_defects": defects.model_dump(),
        "current_step": "oa_parser_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def multimodal_prior_art_agent_node(
    state: OAState,
    agent: BaseStructuredAgent[PriorArtTargetedReadingReport],
    rag_service: RAGSearchService,
    *,
    top_k: int = 8,
    min_score: float = 0.55,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if state.get("parsed_defects") is None:
        raise ValueError("parsed_defects is required before multimodal prior-art reading.")
    if state.get("application_baseline") is None:
        raise ValueError("application_baseline is required before multimodal prior-art reading.")

    prior_arts_paths = state.get("prior_arts_paths", [])
    rag_service.build_index_from_paths(prior_arts_paths)
    cue_query = str(state["parsed_defects"])
    hits = rag_service.retrieve(query=cue_query, top_k=top_k, min_score=min_score)
    retrieved_contexts = [hit.model_dump() for hit in hits]

    prior_art_paths = [item.get("source_path", "") for item in state.get("prior_art_images", []) if item.get("source_path")]
    prior_art_mimes = [item.get("mime_type", "image/png") for item in state.get("prior_art_images", []) if item.get("source_path")]
    app_paths = [item.get("source_path", "") for item in state.get("application_images", []) if item.get("source_path")]
    app_mimes = [item.get("mime_type", "image/png") for item in state.get("application_images", []) if item.get("source_path")]

    context_paths = prior_art_paths[:4] + app_paths[:2]
    context_mimes = prior_art_mimes[:4] + app_mimes[:2]
    warnings = list(state.get("vision_warnings", []))
    if len(context_paths) == 0:
        warnings.append(
            {
                "code": "VISION_WARNING",
                "message": "Targeted visual verification skipped because no images were provided.",
                "node": "multimodal_prior_art_agent",
            }
        )

    prompt = (
        "You are an Expert Multimodal Prior-Art Analyst for Chinese patents.\n"
        "Read ONLY OA-cited evidence and figures, then produce a highly granular, targeted verification report.\n\n"
        "CORE DIRECTIVES:\n"
        "1. DEFAULT CONCESSION: Default assumption is that examiner conclusion is generally correct.\n"
        "   Prioritize confirming support to map the minefield for future amendments.\n"
        "2. STRICT SCOPE: Use [OA_DEFECTS], [APPLICATION_BASELINE], and [RETRIEVED_SNIPPETS] as hard scope.\n"
        "   Do NOT summarize full documents.\n"
        "3. ANCHORING: Treat [APPLICATION_BASELINE] as semantic anchor and application drawings as visual anchor.\n"
        "4. NO HALLUCINATION IN DISPUTES: List disputable_items only when prior-art evidence is truly insufficient or visually contradictory.\n\n"
        "GRANULARITY REQUIREMENT:\n"
        "- For each SupportingItem, prior_art_visual_disclosure MUST be highly detailed:\n"
        "  include physical shape, structural connection, and relative positions from drawings.\n"
        "- Your output is a minefield map for downstream claim amendment. Missing structural detail may cause repeated rejection.\n\n"
        "Content in JSON must be Chinese (Simplified Chinese).\n"
        "Return valid JSON only matching the schema.\n\n"
        f"[APPLICATION_BASELINE]\n{state.get('application_baseline')}\n\n"
        f"[OA_DEFECTS]\n{state['parsed_defects']}\n\n"
        f"[RETRIEVED_SNIPPETS (Text & Image Context)]\n{retrieved_contexts}"
    )
    targeted_report = agent.run_structured(
        prompt=prompt,
        output_model=PriorArtTargetedReadingReport,
        context={"image_paths": context_paths, "image_mime_types": context_mimes},
    )
    if targeted_report.examiner_conclusion_supported is None:
        targeted_report.examiner_conclusion_supported = True
    defects = state.get("parsed_defects") or {}
    defect_items = defects.get("defects") if isinstance(defects, dict) else []

    figure_refs: list[str] = []
    diff_items: list[PriorArtVisualDiff] = []
    for item in targeted_report.supporting_items:
        point = item.prior_art_visual_disclosure
        diff_items.append(
            PriorArtVisualDiff(
                feature_name=item.target_feature,
                application_evidence="Visual cross-check based on applicant drawings and OA context.",
                prior_art_evidence=point,
                difference_assessment=item.amendment_avoidance_warning,
            )
        )
    for disputable in targeted_report.disputable_items:
        point = disputable.multimodal_reality_check
        diff_items.append(
            PriorArtVisualDiff(
                feature_name=disputable.target_feature,
                application_evidence=disputable.examiner_assertion,
                prior_art_evidence=point,
                difference_assessment=disputable.rebuttal_angle,
            )
        )

    import re
    pattern = re.compile(r"图\s*\d+")
    for item in targeted_report.supporting_items:
        figure_refs.extend(pattern.findall(item.prior_art_visual_disclosure))
    for disputable in targeted_report.disputable_items:
        figure_refs.extend(pattern.findall(disputable.multimodal_reality_check))
    if not figure_refs:
        # Fallback to OA coordinates when report text doesn't carry explicit figure labels.
        for item in defect_items if isinstance(defect_items, list) else []:
            if not isinstance(item, dict):
                continue
            for mapping in item.get("feature_mappings", []) if isinstance(item.get("feature_mappings"), list) else []:
                if isinstance(mapping, dict):
                    fig = str(mapping.get("cited_figures", "")).strip()
                    if fig and fig != "无":
                        figure_refs.append(fig)

    visual_report = PriorArtVisualReport(
        cited_figure_refs=sorted(set(figure_refs)),
        diffs=diff_items,
        conclusion=targeted_report.overall_conclusion,
    )

    app_image_ids = [str(item.get("image_id", "")) for item in state.get("application_images", []) if item.get("image_id")]
    prior_image_ids = [str(item.get("image_id", "")) for item in state.get("prior_art_images", []) if item.get("image_id")]
    used_app_image_ids = app_image_ids[:2]
    used_prior_image_ids = prior_image_ids[:4]
    image_recognition_report = {
        "application_extracted_total": len(app_image_ids),
        "prior_art_extracted_total": len(prior_image_ids),
        "application_used_for_vision": len(used_app_image_ids),
        "prior_art_used_for_vision": len(used_prior_image_ids),
        "application_image_ids": app_image_ids,
        "prior_art_image_ids": prior_image_ids,
        "used_application_image_ids": used_app_image_ids,
        "used_prior_art_image_ids": used_prior_image_ids,
        "application_truncated_for_vision": len(app_image_ids) > len(used_app_image_ids),
        "prior_art_truncated_for_vision": len(prior_image_ids) > len(used_prior_image_ids),
    }

    cited_docs: list[str] = []
    cited_figures: list[str] = []
    if isinstance(defect_items, list):
        for item in defect_items:
            if not isinstance(item, dict):
                continue
            docs = item.get("main_cited_docs") or item.get("cited_docs")
            if isinstance(docs, list):
                cited_docs.extend(str(v) for v in docs if isinstance(v, str) and v.strip())
            feature_mappings = item.get("feature_mappings")
            if isinstance(feature_mappings, list):
                for mapping in feature_mappings:
                    if not isinstance(mapping, dict):
                        continue
                    doc = str(mapping.get("prior_art_doc", "")).strip()
                    fig = str(mapping.get("cited_figures", "")).strip()
                    if doc and doc != "无":
                        cited_docs.append(doc)
                    if fig and fig != "无":
                        cited_figures.append(fig)
            figs = item.get("cited_figure_refs")
            if isinstance(figs, list):
                cited_figures.extend(str(v) for v in figs if isinstance(v, str) and v.strip())
    cited_docs = sorted(set(cited_docs))
    cited_figures = sorted(set(cited_figures))

    targeted_docs: list[str] = []
    for ctx in retrieved_contexts:
        if not isinstance(ctx, dict):
            continue
        for key in ("doc_id", "source_doc", "source_name", "source_path"):
            val = str(ctx.get(key, "")).upper()
            for m in re.findall(r"D\d+", val):
                targeted_docs.append(m)
    targeted_docs = sorted(set(targeted_docs))

    targeted_figures: list[str] = []
    for fig in visual_report.cited_figure_refs:
        if isinstance(fig, str) and fig.strip():
            targeted_figures.append(fig.strip())
    targeted_figures = sorted(set(targeted_figures))
    targeted_reading_audit = {
        "cited_docs_from_oa": cited_docs,
        "cited_figures_from_oa": cited_figures,
        "targeted_docs_read": targeted_docs,
        "targeted_figures_read": targeted_figures,
        "retrieved_snippets_count": len(retrieved_contexts),
        "all_cited_docs_covered": all(doc in targeted_docs for doc in cited_docs) if cited_docs else True,
        "all_cited_figures_covered": all(fig in targeted_figures for fig in cited_figures) if cited_figures else True,
    }

    return {
        "retrieved_contexts": retrieved_contexts,
        "prior_art_targeted_report": targeted_report.model_dump(),
        "image_recognition_report": image_recognition_report,
        "targeted_reading_audit": targeted_reading_audit,
        "visual_report": visual_report.model_dump(),
        "vision_warnings": warnings,
        "current_step": "multimodal_prior_art_agent",
        "status": "running",
        "last_error": None,
        "node_latency_ms": _duration_ms(started_at),
    }


def application_baseline_node(
    state: OAState,
    agent: BaseStructuredAgent[ApplicationBaselineReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    claims_payload = state.get("original_claims", {})
    spec_payload = state.get("application_specification", {})
    prompt = (
        "You are an Expert Chinese Patent Attorney. Your task is to deconstruct the provided patent application "
        "(claims and specification) to establish a highly granular, structured 'Application Baseline'.\n"
        "This baseline will serve as the ground truth for downstream Office Action defense and claim amendments.\n\n"
        "EXTRACTION RULES:\n"
        "1. ATOMIC CLAIM BREAKDOWN:\n"
        "   - Parse the claims into a dependency tree (`claims_tree`).\n"
        "   - For each claim, break down the text into distinct, atomic technical features (`features`). "
        "Do not lump multiple hardware components or distinct method steps into a single feature.\n"
        "2. EXHAUSTIVE SPECIFICATION MINING:\n"
        "   - Scan the specification text and build a comprehensive feature dictionary (`spec_feature_index`).\n"
        "   - Extract EVERY physical component, logical module, or method step described, even if it is NOT currently in the claims.\n"
        "   - CRITICAL: You must explicitly extract Reference Numerals and specific structural relationships/working principles. "
        "This is the fallback goldmine for future amendments.\n"
        "3. NO HALLUCINATION: Extract ONLY what is explicitly written in the provided text. "
        "Do not summarize too broadly; retain the original technical terminology.\n"
        "4. LANGUAGE: All output content within the JSON MUST be in professional Chinese.\n\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[ORIGINAL_CLAIMS]\n{claims_payload}\n\n"
        f"[SPECIFICATION_TEXT]\n{spec_payload}"
    )
    baseline = agent.run_structured(prompt=prompt, output_model=ApplicationBaselineReport)
    baseline_payload = baseline.model_dump()
    if not baseline_payload.get("spec_feature_index"):
        baseline_payload["spec_feature_index"] = []
    if not baseline_payload.get("claims_tree"):
        baseline_payload["claims_tree"] = []
    return {
        "application_baseline": baseline_payload,
        "current_step": "application_baseline_agent",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def concession_and_gap_node(
    state: OAState,
    agent: BaseStructuredAgent[ConcessionGapReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    max_reflections = _safe_int(state.get("max_reflections", 3), 3) or 3
    prompt = (
        "You are an Expert Chinese Patent OA Strategy Analyst.\n"
        "Your task is to produce a Concession & Gap Report for downstream claim amendment.\n"
        "You MUST follow the practical strategy priority: merge first, then mine specification fallback features.\n\n"
        "STRATEGY PRIORITY:\n"
        "1. MERGE-FIRST:\n"
        "   - When independent claim (e.g., Claim 1) is defeated, first check dependent claims.\n"
        "   - Identify which dependent claims remain defensible and should be merged into Claim 1.\n"
        "   - Put these claim numbers into `recommended_merges`.\n"
        "2. MINE-SECOND:\n"
        "   - Only when merges are insufficient or no merge candidate exists, issue detailed `mining_directives`.\n"
        "   - Mining directives must target concrete components/steps and clearly define the technical gap to fill.\n"
        "   - Include strict avoidance warnings based on prior-art minefields.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "1. CLAIM ASSESSMENT:\n"
        "   - For each key claim, output `claim_number`, `status` (DEFEATED / MERGE_CANDIDATE / UNCERTAIN), and `reasoning`.\n"
        "2. MERGE RECOMMENDATION:\n"
        "   - `recommended_merges` must contain claim numbers to merge into independent claim.\n"
        "3. MINING DIRECTIVES:\n"
        "   - For each directive, output `target_component_or_step`, `technical_gap_to_fill`, `avoidance_warning`.\n"
        "4. LANGUAGE:\n"
        "   - All JSON content must be in Chinese.\n"
        "   - Return valid JSON strictly matching the output schema.\n\n"
        f"MAX_REFLECTION_ROUNDS: {max_reflections}\n"
        "Before finalizing, internally self-review your conclusions for up to MAX_REFLECTION_ROUNDS rounds.\n\n"
        f"[OA_DEFECTS]\n{state.get('parsed_defects')}\n\n"
        f"[TARGETED_PRIOR_ART_REPORT]\n{state.get('prior_art_targeted_report')}\n\n"
        f"[APPLICATION_BASELINE]\n{state.get('application_baseline')}"
    )
    report = agent.run_structured(prompt=prompt, output_model=ConcessionGapReport)
    payload = report.model_dump()
    # Backward-compatible derivation for downstream nodes still using legacy fields.
    if not payload.get("failed_claims"):
        payload["failed_claims"] = [
            _safe_int(item.get("claim_number"), 0)
            for item in payload.get("claim_assessments", [])
            if isinstance(item, dict) and str(item.get("status", "")).upper() == "DEFEATED"
        ]
    if not payload.get("confirmed_points"):
        payload["confirmed_points"] = [
            str(item.get("reasoning", "")).strip()
            for item in payload.get("claim_assessments", [])
            if isinstance(item, dict) and str(item.get("status", "")).upper() in {"DEFEATED", "MERGE_CANDIDATE"}
        ]
    if not payload.get("gap_targets"):
        payload["gap_targets"] = [
            str(item.get("technical_gap_to_fill", "")).strip()
            for item in payload.get("mining_directives", [])
            if isinstance(item, dict) and str(item.get("technical_gap_to_fill", "")).strip()
        ]
    if not payload.get("rationale"):
        payload["rationale"] = str(payload.get("overall_strategy_summary", "")).strip() or "基于当前战况执行先合并后引入策略。"
    return {
        "concession_gap_report": payload,
        "current_step": "concession_and_gap_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def fallback_feature_miner_node(
    state: OAState,
    agent: BaseStructuredAgent[FallbackFeatureMiningReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    application_baseline = state.get("application_baseline")
    concession_gap = state.get("concession_gap_report") or {}
    mining_directives = concession_gap.get("mining_directives", []) if isinstance(concession_gap, dict) else []
    original_claims = state.get("original_claims")
    prompt = (
        "You are an Expert Fallback-Feature Mining Specialist for Chinese patents.\n"
        "Your crucial task is to salvage the patent application by finding specific, undisclosed technical features "
        "from the specification to overcome the examiner's rejection.\n\n"
        "### CORE MINING RULES:\n"
        "1. STRICT OBEDIENCE TO DIRECTIVES:\n"
        "   - You MUST strictly follow `mining_directives` from Node 4. Do not mine random features.\n"
        "2. MINEFIELD EVASION:\n"
        "   - Read `avoidance_warning` carefully. If a feature falls into warned category, discard it and keep searching.\n"
        "3. NOVELTY WITHIN APPLICATION:\n"
        "   - The feature must NOT already exist in the currently rejected independent claims.\n"
        "4. VERBATIM EXTRACTION - ZERO HALLUCINATION:\n"
        "   - For each candidate, extract exact `verbatim_quote` from APPLICATION_BASELINE spec_feature_index.\n"
        "   - Paraphrasing or inventing terms is forbidden.\n"
        "   - Record exact `source_location`.\n\n"
        "Content in JSON must be professional Chinese.\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[APPLICATION_BASELINE (Specification Feature Goldmine)]\n{application_baseline}\n\n"
        f"[MINING_DIRECTIVES (Targets & Avoidance Warnings from Node 4)]\n{mining_directives}\n\n"
        f"[CURRENT_CLAIMS (Do not extract these again)]\n{original_claims}"
    )
    mining = agent.run_structured(prompt=prompt, output_model=FallbackFeatureMiningReport)
    payload = mining.model_dump()
    # Compatibility normalization for legacy model outputs.
    for item in payload.get("candidates", []):
        if not isinstance(item, dict):
            continue
        if not str(item.get("feature_name", "")).strip():
            item["feature_name"] = str(item.get("feature_text", "")).strip() or "未命名特征"
        if not str(item.get("verbatim_quote", "")).strip():
            item["verbatim_quote"] = str(item.get("source_quote", "")).strip()
        if not str(item.get("addressed_directive", "")).strip():
            item["addressed_directive"] = "未标注指令"
        if not str(item.get("reference_numeral", "")).strip():
            item["reference_numeral"] = "无"
        if not str(item.get("gap_filling_rationale", "")).strip():
            item["gap_filling_rationale"] = str(item.get("commercial_relevance", "")).strip() or "用于补足技术缺口。"
    if not payload.get("mining_status"):
        payload["mining_status"] = "SUCCESS" if payload.get("candidates") else "EXHAUSTED"
    return {
        "mined_fallback_features": payload,
        "current_step": "fallback_feature_miner_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def prior_art_stress_tester_node(
    state: OAState,
    agent: BaseStructuredAgent[PriorArtStressTestReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    max_reflections = _safe_int(state.get("max_reflections", 3), 3) or 3
    prior_art_paths = [item.get("source_path", "") for item in state.get("prior_art_images", []) if item.get("source_path")]
    prior_art_mimes = [item.get("mime_type", "image/png") for item in state.get("prior_art_images", []) if item.get("source_path")]
    context_paths = prior_art_paths[:6]
    context_mimes = prior_art_mimes[:6]
    prompt = (
        "You are an Expert Prior-Art Red-Team Stress Tester for Chinese patents.\n"
        "Your mission is to rigorously interrogate the newly mined fallback candidates against cited prior art.\n\n"
        "### STRESS TEST PROTOCOL:\n"
        "1. MULTIMODAL LETHALITY:\n"
        "   - Test each candidate against BOTH textual snippets and visual figures.\n"
        "   - If drawings strongly imply the structure, mark ELIMINATED even without explicit text.\n"
        "2. COMBINATION RISK:\n"
        "   - Evaluate D1+D2 combination motivation and obviousness risk.\n"
        "   - If easy combination exists, mark ELIMINATED or UNCERTAIN.\n"
        "3. SURVIVOR'S REBUTTAL:\n"
        "   - For SURVIVED candidates, provide robust rebuttal_foundation for downstream argument drafting.\n"
        "4. LANGUAGE:\n"
        "   - All output content within JSON MUST be in professional Chinese.\n\n"
        f"MAX_REFLECTION_ROUNDS: {max_reflections}\n"
        "Before final verdicts, perform up to MAX_REFLECTION_ROUNDS internal red-team reflection passes.\n\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[FALLBACK_CANDIDATES (From Node 5)]\n{state.get('mined_fallback_features')}\n\n"
        f"[PRIOR_ART_VERIFIED_REPORT (Known Minefields from Node 3)]\n{state.get('prior_art_targeted_report')}\n\n"
        f"[PRIOR_ART_CONTEXTS (Text & Images for verification)]\n{state.get('retrieved_contexts', [])}"
    )
    report = agent.run_structured(
        prompt=prompt,
        output_model=PriorArtStressTestReport,
        context={"image_paths": context_paths, "image_mime_types": context_mimes},
    )
    payload = report.model_dump()
    tested = payload.get("tested_features", [])
    if not tested and payload.get("results"):
        tested = payload.get("results", [])
        payload["tested_features"] = tested
    if not payload.get("overall_survival_rate"):
        surv = [x for x in tested if isinstance(x, dict) and str(x.get("test_verdict", x.get("verdict", ""))).upper() == "SURVIVED"]
        payload["overall_survival_rate"] = f"压测了{len(tested)}个特征，最终{len(surv)}个存活。"

    normalized_ids: list[str] = []
    for item in tested:
        if not isinstance(item, dict):
            continue
        if not str(item.get("feature_name", "")).strip():
            item["feature_name"] = "未命名特征"
        verdict = str(item.get("test_verdict", "")).strip().upper()
        if not verdict:
            legacy = str(item.get("verdict", "")).strip().lower()
            mapping = {"survive": "SURVIVED", "eliminated": "ELIMINATED", "uncertain": "UNCERTAIN"}
            verdict = mapping.get(legacy, "UNCERTAIN")
            item["test_verdict"] = verdict
        if not str(item.get("prior_art_hit_location", "")).strip():
            item["prior_art_hit_location"] = str(item.get("textual_evidence", "")).strip() or "无"
        if not str(item.get("red_team_reasoning", "")).strip():
            item["red_team_reasoning"] = str(item.get("risk_reason", "")).strip() or "基于图文对比结果形成该结论。"
        if not str(item.get("rebuttal_foundation", "")).strip():
            item["rebuttal_foundation"] = "无" if verdict != "SURVIVED" else "该特征在D1/D2中未被公开，且具备独特技术效果。"
        if verdict == "SURVIVED":
            cid = str(item.get("candidate_id", "")).strip()
            if cid:
                normalized_ids.append(cid)
    if not payload.get("survived_candidate_ids"):
        payload["survived_candidate_ids"] = normalized_ids
    return {
        "stress_test_report": payload,
        "current_step": "prior_art_stress_tester_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def strategy_decision_node(
    state: OAState,
    agent: BaseStructuredAgent[StrategyDecision],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompt = (
        "You are the Final OA Strategy Decision Commander for Chinese patents.\n"
        "Your mission is to synthesize node-3/4/6 intelligence and issue deterministic execution blueprints for Node 8 and Node 9.\n\n"
        "### CRITICAL DIRECTIVES:\n"
        "1. NO MANUAL ESCALATION: You MUST output a fully automated resolution path.\n"
        "2. DECISION ROUTING:\n"
        "   - Use AMEND_AND_ARGUE as default preferred path.\n"
        "   - Use ARGUE_ONLY only if node-3 proves examiner mapping is completely factually wrong and current claims are fully safe.\n"
        "3. AMENDMENT TACTIC HIERARCHY (STRICT):\n"
        "   - PRIORITY 1 MERGE: Read recommended_merges from CONCESSION_GAP_REPORT and merge safe dependent claims first.\n"
        "   - PRIORITY 2 INTRODUCE: If merge unavailable/unsafe, use survived_candidate_ids from STRESS_TEST_REPORT.\n"
        "4. BLUEPRINT GENERATION:\n"
        "   - amendment_plan must be exact and executable for node 8.\n"
        "   - rebuttal_plan must leverage rebuttal_foundation from stress test for node 9.\n\n"
        "Content in JSON must be professional Chinese.\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[CONCESSION_GAP_REPORT (Node 4 - Merge Recommendations)]\n{state.get('concession_gap_report')}\n\n"
        f"[STRESS_TEST_REPORT (Node 6 - Survived Features)]\n{state.get('stress_test_report')}\n\n"
        f"[PRIOR_ART_REPORT (Node 3 - Fact Checks)]\n{state.get('prior_art_targeted_report')}"
    )
    decision = agent.run_structured(prompt=prompt, output_model=StrategyDecision)
    decision_payload = decision.model_dump()

    stress_report = state.get("stress_test_report") or {}
    survived_ids = list(stress_report.get("survived_candidate_ids", [])) if isinstance(stress_report, dict) else []
    concession_report = state.get("concession_gap_report") or {}
    recommended_merges = list(concession_report.get("recommended_merges", [])) if isinstance(concession_report, dict) else []
    mined = state.get("mined_fallback_features") or {}
    candidates = mined.get("candidates", []) if isinstance(mined, dict) else []
    candidate_text_by_id: dict[str, str] = {}
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict):
                cid = str(item.get("candidate_id", "")).strip()
                ftxt = str(item.get("feature_name", "")).strip() or str(item.get("feature_text", "")).strip()
                if cid and ftxt:
                    candidate_text_by_id[cid] = ftxt

    # Hard guardrail: fully automated route only; default AMEND_AND_ARGUE.
    global_decision = str(decision_payload.get("global_decision", decision_payload.get("action", ""))).upper()
    if global_decision not in {"AMEND_AND_ARGUE", "ARGUE_ONLY"}:
        global_decision = "AMEND_AND_ARGUE"
    if global_decision != "AMEND_AND_ARGUE":
        # allow ARGUE_ONLY only under strict condition
        prior_report = state.get("prior_art_targeted_report") or {}
        fully_wrong = bool(isinstance(prior_report, dict) and prior_report.get("examiner_conclusion_supported") is False)
        if not fully_wrong:
            global_decision = "AMEND_AND_ARGUE"
    decision_payload["global_decision"] = global_decision
    decision_payload["action"] = global_decision

    amendment_plan = decision_payload.get("amendment_plan")
    if global_decision == "AMEND_AND_ARGUE":
        if not isinstance(amendment_plan, dict):
            amendment_plan = {}
        target_claim = _safe_int(amendment_plan.get("target_independent_claim", 1), 1) or 1
        tactic = str(amendment_plan.get("amendment_tactic", "")).upper()
        if recommended_merges:
            tactic = "MERGE_DEPENDENT"
            amendment_plan["source_dependent_claims"] = recommended_merges
            amendment_plan["survived_candidate_ids"] = []
            amendment_plan["amendment_guidance"] = (
                f"请将从属权利要求{recommended_merges}的限定特征并入权利要求{target_claim}，并调整从属引用关系。"
            )
        else:
            tactic = "INTRODUCE_SPEC_FEATURE"
            selected_ids = amendment_plan.get("survived_candidate_ids")
            if not isinstance(selected_ids, list) or len(selected_ids) == 0:
                selected_ids = survived_ids
            amendment_plan["source_dependent_claims"] = []
            amendment_plan["survived_candidate_ids"] = selected_ids
            selected_texts = [candidate_text_by_id.get(str(cid), str(cid)) for cid in selected_ids]
            amendment_plan["amendment_guidance"] = (
                f"请将候选特征{selected_texts}按原话插入权利要求{target_claim}的对应技术特征之后，并保证术语前后一致。"
            )
        amendment_plan["target_independent_claim"] = target_claim
        amendment_plan["amendment_tactic"] = tactic
        decision_payload["amendment_plan"] = amendment_plan
        decision_payload["selected_candidate_ids"] = list(amendment_plan.get("survived_candidate_ids", []))
        decision_payload["amendment_instruction"] = str(amendment_plan.get("amendment_guidance", "")).strip()
    else:
        decision_payload["amendment_plan"] = None

    if not str(decision_payload.get("strategy_rationale", "")).strip():
        if recommended_merges:
            decision_payload["strategy_rationale"] = f"权利要求1被击穿，优先并入从属权利要求{recommended_merges}以稳妥收窄保护范围。"
        elif survived_ids:
            decision_payload["strategy_rationale"] = "无可安全合并从权，改为引入压测存活特征以形成新的区别点。"
        else:
            decision_payload["strategy_rationale"] = "在可得信息下执行自动修订路径，优先保证可执行性与可答辩性。"

    if not str(decision_payload.get("argument_logic", "")).strip():
        decision_payload["argument_logic"] = str(decision_payload.get("strategy_rationale", "")).strip()

    rebuttal_items = decision_payload.get("rebuttal_plan")
    if not isinstance(rebuttal_items, list) or len(rebuttal_items) == 0:
        stress_results: list[dict[str, Any]] = []
        if isinstance(stress_report, dict):
            raw_stress_results = stress_report.get("tested_features") or stress_report.get("results") or []
            if isinstance(raw_stress_results, list):
                stress_results = [item for item in raw_stress_results if isinstance(item, dict)]
        foundation = "结合节点3与节点6结果，修改后特征与D1/D2存在明确结构差异并产生独特技术效果。"
        if isinstance(stress_results, list):
            for item in stress_results:
                if isinstance(item, dict) and str(item.get("test_verdict", item.get("verdict", ""))).upper() == "SURVIVED":
                    foundation = str(item.get("rebuttal_foundation", "")).strip() or foundation
                    break
        rebuttal_items = [
            {
                "target_claim": _safe_int((decision_payload.get("amendment_plan") or {}).get("target_independent_claim", 1), 1) or 1,
                "core_argument_logic": foundation,
                "evidence_support": "依据节点3图文核验与节点6红队压测存活结论。",
            }
        ]
        decision_payload["rebuttal_plan"] = rebuttal_items

    decision = StrategyDecision.model_validate(decision_payload)

    action_map: dict[str, Literal["Argue", "Amend"]] = {"AMEND_AND_ARGUE": "Amend", "ARGUE_ONLY": "Argue"}
    first_rebuttal = rebuttal_items[0] if isinstance(rebuttal_items, list) and rebuttal_items else {}
    rebuttal_plan = RebuttalActionPlan(
        action=action_map.get(decision_payload.get("global_decision", "AMEND_AND_ARGUE"), "Argue"),
        target_claims=[_safe_int(first_rebuttal.get("target_claim", 1), 1)],
        rationale=str(decision_payload.get("strategy_rationale", "")),
        argument_points=[str(first_rebuttal.get("core_argument_logic", ""))],
        amendment_instructions=[str((decision_payload.get("amendment_plan") or {}).get("amendment_guidance", "")).strip()]
        if decision_payload.get("amendment_plan")
        else [],
    )
    comparison_result = ComparisonResult(
        feature_diffs=[str(first_rebuttal.get("core_argument_logic", ""))],
        supporting_evidence=[],
        visual_report=PriorArtVisualReport.model_validate(
            state.get("visual_report") or {"cited_figure_refs": [], "diffs": [], "conclusion": "No visual verification conclusion."}
        ),
    )
    return {
        "strategy_decision": decision_payload,
        "rebuttal_plan": rebuttal_plan.model_dump(),
        "comparison_result": comparison_result.model_dump(),
        "current_step": "strategy_decision_node",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def claim_amendment_node(
    state: OAState,
    agent: BaseStructuredAgent[ClaimAmendmentResult],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    decision = state.get("strategy_decision") or {}
    decision_flag = str(decision.get("global_decision", decision.get("action", "")))
    if decision_flag != "AMEND_AND_ARGUE":
        original_claims = state.get("original_claims")
        fallback_text = str(original_claims) if original_claims is not None else "无"
        return {
            "amended_claims": {
                "is_amended": False,
                "amendment_basis_statement": "本轮策略为仅答辩（ARGUE_ONLY），权利要求书未作实质性修改。",
                "claim_mappings": [],
                "final_claims_text": fallback_text,
                "amended_claims": original_claims or {},
                "amendment_log": ["ARGUE_ONLY：未修改权利要求文本。"],
            },
            "current_step": "claim_amendment_agent",
            "status": "running",
            "node_latency_ms": _duration_ms(started_at),
        }

    prompt = (
        "You are an Expert Chinese Patent Claim Amendment Specialist (Surgeon).\n"
        "Your sole duty is to flawlessly execute the `amendment_plan` provided by the Strategist (Node 7) on the ORIGINAL_CLAIMS.\n\n"
        "### SURGICAL RULES (RED LINES):\n"
        "1. ABSOLUTE OBEDIENCE: Apply ONLY instructions in amendment_plan. Do not play strategist.\n"
        "2. NO NEW MATTER (Article 33):\n"
        "   - MERGE_DEPENDENT: copy source dependent claim text into independent claim, then delete merged claims.\n"
        "   - INTRODUCE_SPEC_FEATURE: use exact verbatim_quote from MINED_FALLBACK_FEATURES by survived_candidate_ids.\n"
        "     Only connective words may be added. No paraphrasing or invented technical nouns.\n"
        "3. COHERENCE & RENUMBERING:\n"
        "   - After deletion/merge, renumber remaining claims and update dependency references accordingly.\n"
        "4. AMENDMENT BASIS:\n"
        "   - amendment_basis_statement must explicitly state source of new limitations for downstream legal drafting.\n\n"
        "Content in JSON must be professional Chinese.\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[STRATEGY_DECISION (Node 7 - The Blueprint)]\n{state.get('strategy_decision')}\n\n"
        f"[MINED_FALLBACK_FEATURES (Node 5 - Verbatim Text for Insertion)]\n{state.get('mined_fallback_features')}\n\n"
        f"[ORIGINAL_CLAIMS (The Patient to be Operated On)]\n{state.get('original_claims')}"
    )
    amended = agent.run_structured(prompt=prompt, output_model=ClaimAmendmentResult)
    payload = amended.model_dump()
    if not payload.get("amended_claims"):
        payload["amended_claims"] = state.get("original_claims") or {}
    if not payload.get("final_claims_text"):
        payload["final_claims_text"] = str(payload.get("amended_claims"))
    if not payload.get("amendment_log"):
        payload["amendment_log"] = [str(payload.get("amendment_basis_statement", "已按施工图完成修改。"))]
    return {
        "amended_claims": payload,
        "current_step": "claim_amendment_agent",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def argument_writer_node(
    state: OAState,
    agent: BaseStructuredAgent[ArgumentDraft],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompt = (
        "You are a Senior Chinese Patent Attorney drafting a highly detailed and persuasive Office Action Response.\n"
        "Your goal is to generate a comprehensive response that directly addresses examiner rejections and thoroughly analyzes technical differences. "
        "You have freedom to structure final text dynamically for maximum persuasive impact.\n\n"
        "### DRAFTING PROTOCOL:\n"
        "1. TARGETED REFUTATION:\n"
        "   - Summarize examiner viewpoint, then forcefully counter by pointing out fundamental differences in overall technical solution.\n"
        "2. EXTREME TECHNICAL DETAIL:\n"
        "   - Deeply analyze distinguishing features based on AMENDED_CLAIMS.\n"
        "   - Do NOT just list names; explain mechanisms and synergies.\n"
        "3. REBUTTING OBVIOUSNESS:\n"
        "   - Argue the features form an indivisible cooperative system and exceed routine combination logic.\n"
        "4. TANGIBLE EFFECTS:\n"
        "   - Link structural differences to concrete unexpected technical effects.\n"
        "5. DYNAMIC DOCUMENT ASSEMBLY:\n"
        "   - Do NOT rigidly force hardcoded section numbering.\n"
        "   - Start politely and end with Article 22.3 creativity conclusion and grant request.\n\n"
        "Return valid JSON strictly matching the output schema. Chinese must be formal, legally precise, and structurally rich.\n\n"
        f"[EXAMINER'S REJECTION (审查员驳回意见)]\n{state.get('parsed_defects')}\n\n"
        f"[STRATEGY_DECISION (答辩与修改策略)]\n{state.get('strategy_decision')}\n\n"
        f"[AMENDED_CLAIMS (修改后权利要求)]\n{state.get('amended_claims')}"
    )
    draft = agent.run_structured(prompt=prompt, output_model=ArgumentDraft)
    payload = draft.model_dump()
    # Flexible fallback: natural paragraph assembly instead of rigid fixed headings.
    if not str(payload.get("final_reply_text", "")).strip():
        tech_diffs = "\n\n".join(
            str(x).strip() for x in payload.get("detailed_technical_differences", []) if str(x).strip()
        )
        effects = "\n\n".join(str(x).strip() for x in payload.get("unexpected_effects", []) if str(x).strip())
        parts = [
            "尊敬的审查员：\n\n您好！感谢您对本申请的认真审查。结合申请文件修改情况，现提交意见陈述如下：",
            str(payload.get("amendment_statement", "")).strip(),
            str(payload.get("examiner_logic_refutation", "")).strip(),
            f"具体的技术机制与结构差异如下：\n{tech_diffs}" if tech_diffs else "",
            str(payload.get("non_obviousness_argument", "")).strip(),
            f"基于上述区别机制，本申请取得如下技术效果：\n{effects}" if effects else "",
            "综上所述，修改后的权利要求具备《专利法》第22条第3款规定的创造性，恳请审查员予以认可并授予专利权。",
        ]
        payload["final_reply_text"] = "\n\n".join([p for p in parts if p.strip()])

    # Legacy mirrors for existing consumers.
    if not payload.get("argument_text"):
        payload["argument_text"] = payload.get("final_reply_text", "")
    if not payload.get("key_points"):
        payload["key_points"] = [
            str(payload.get("examiner_logic_refutation", "")).strip(),
            str(payload.get("non_obviousness_argument", "")).strip(),
        ]
        payload["key_points"] = [x for x in payload["key_points"] if x]
    return {
        "argument_draft": payload,
        "final_reply_text": payload.get("final_reply_text"),
        "current_step": "argument_writer_agent",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def spec_update_node(
    state: OAState,
    agent: BaseStructuredAgent[SpecUpdateNote],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompt = (
        "You are a Senior Patent Drafting QA Assistant for Chinese patents.\n"
        "Your critical task is to review the newly AMENDED_CLAIMS against the original APPLICATION_BASELINE and perform "
        "MINIMAL terminology harmonization.\n\n"
        "### ADAPTIVE AMENDMENT RULES:\n"
        "1. TERMINOLOGY HARMONIZATION ONLY:\n"
        "   - Only align terminology between amended claims and specification.\n"
        "2. NEVER REWRITE FULL SPECIFICATION:\n"
        "   - Output ONLY targeted Find-and-Replace instructions.\n"
        "3. ARTICLE 33 STRICT COMPLIANCE:\n"
        "   - Do NOT add any new technical features/effects/embodiments.\n"
        "4. NO UNNECESSARY CHANGES:\n"
        "   - If already consistent or claims not amended, set requires_spec_update=false and amendment_items=[].\n\n"
        "Content in JSON must be formal professional Chinese.\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[AMENDED_CLAIMS (Node 8 - The New Standard)]\n{state.get('amended_claims')}\n\n"
        f"[APPLICATION_BASELINE (Node 1 - Original Specification Features)]\n{state.get('application_baseline')}"
    )
    note = agent.run_structured(prompt=prompt, output_model=SpecUpdateNote)
    payload = note.model_dump()
    if "requires_spec_update" not in payload:
        payload["requires_spec_update"] = bool(payload.get("applied", False))
    if not payload.get("article_33_declaration"):
        payload["article_33_declaration"] = "上述说明书修改仅为术语统一和适应性调整，未引入新的技术内容，符合《专利法》第33条规定。"
    if not payload.get("changes"):
        payload["changes"] = [
            f"{item.get('target_paragraph', '')}: {item.get('original_text_snippet', '')} -> {item.get('amended_text_snippet', '')}"
            for item in payload.get("amendment_items", [])
            if isinstance(item, dict)
        ]
    payload["applied"] = bool(payload.get("requires_spec_update", False))
    return {
        "spec_update_note": payload,
        "current_step": "spec_update_agent",
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }


def response_traceability_node(
    state: OAState,
    agent: BaseStructuredAgent[ResponseTraceabilityReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompt = (
        "You are the Final OA Response Compliance Reviewer (QA Partner) for a top-tier Chinese patent firm.\n"
        "Your solemn duty is to audit the entire generated OA response package before filing.\n\n"
        "### CRITICAL AUDIT PROTOCOLS:\n"
        "1. ARTICLE 33 & 26.4 TRACEABILITY:\n"
        "   - Compare AMENDED_CLAIMS against APPLICATION_BASELINE.\n"
        "   - Trace every newly added feature to original specification; unsupported/hallucinated terms must be flagged.\n"
        "2. LOGICAL CONSISTENCY:\n"
        "   - Ensure argued distinguishing features exactly match amended claims.\n"
        "3. NO HARMFUL ADMISSIONS:\n"
        "   - Flag unnecessary concessions and estoppel-risk statements.\n"
        "4. ISSUING THE RULING:\n"
        "   - If ANY FATAL flag exists, global_go_no_go MUST be NO_GO.\n\n"
        "Content in JSON must be formal professional Chinese.\n"
        "Return valid JSON strictly matching the output schema.\n\n"
        f"[APPLICATION_BASELINE (The Ground Truth)]\n{state.get('application_baseline')}\n\n"
        f"[AMENDED_CLAIMS (The Final Output to Audit)]\n{state.get('amended_claims')}\n\n"
        f"[ARGUMENT_DRAFT (The Legal Response to Audit)]\n{state.get('argument_draft')}\n\n"
        f"[STRATEGY_DECISION & STRESS TEST (For Context)]\n{state.get('strategy_decision')}\n{state.get('stress_test_report')}"
    )
    report = agent.run_structured(prompt=prompt, output_model=ResponseTraceabilityReport)
    payload = report.model_dump()
    if not payload.get("support_basis_audit") and payload.get("findings"):
        payload["support_basis_audit"] = payload.get("findings", [])
    if "logic_consistency_audit" not in payload:
        payload["logic_consistency_audit"] = []
    if "harmful_admission_audit" not in payload:
        payload["harmful_admission_audit"] = []
    if not payload.get("final_strategy_summary"):
        payload["final_strategy_summary"] = str(payload.get("final_risk_summary", "")).strip() or "已完成终审风控审查。"

    def _has_fatal(items: list[dict[str, Any]]) -> bool:
        for it in items:
            if isinstance(it, dict) and str(it.get("severity", "")).upper() == "FATAL":
                return True
        return False

    fatal_exists = _has_fatal(payload.get("support_basis_audit", [])) or _has_fatal(
        payload.get("logic_consistency_audit", [])
    ) or _has_fatal(payload.get("harmful_admission_audit", []))
    if not payload.get("global_go_no_go"):
        payload["global_go_no_go"] = "NO_GO" if fatal_exists else "GO"
    elif fatal_exists:
        payload["global_go_no_go"] = "NO_GO"

    decision = state.get("strategy_decision") or {}
    decision_flag = str(decision.get("global_decision", decision.get("action", "")))
    final_decision: Literal["amend_or_argument", "argument_only"] = (
        "amend_or_argument" if decision_flag == "AMEND_AND_ARGUE" else "argument_only"
    )
    amendment_plan = decision.get("amendment_plan", {}) if isinstance(decision, dict) else {}
    rebuttal_items = decision.get("rebuttal_plan", []) if isinstance(decision, dict) else []
    rebuttal_points = []
    if isinstance(rebuttal_items, list):
        for item in rebuttal_items:
            if isinstance(item, dict):
                logic = str(item.get("core_argument_logic", "")).strip()
                if logic:
                    rebuttal_points.append(logic)
    final_strategy = DebateStrategy(
        decision=final_decision,
        amendment_plan=[str(amendment_plan.get("amendment_guidance", "")).strip()] if isinstance(amendment_plan, dict) and amendment_plan.get("amendment_guidance") else [],
        rebuttal_points=rebuttal_points or ([str(decision.get("argument_logic", "")).strip()] if decision.get("argument_logic") else []),
    )
    return {
        "response_traceability": payload,
        "final_strategy": final_strategy.model_dump(),
        "current_step": "response_traceability_node",
        "status": "completed",
        "node_latency_ms": _duration_ms(started_at),
    }


def parse_oa_node(state: OAState, agent: BaseStructuredAgent[OADefectList]) -> dict[str, Any]:
    return oa_parser_node(state, agent)


def analyze_prior_art_visual_node(
    state: OAState,
    agent: BaseStructuredAgent[PriorArtVisualReport],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    app_images = [ImageAsset.model_validate(item) for item in state.get("application_images", [])]
    prior_images = [ImageAsset.model_validate(item) for item in state.get("prior_art_images", [])]
    visual_report = run_prior_art_visual_analyzer(
        examiner_reasoning=state.get("oa_text", ""),
        application_images=app_images,
        prior_art_images=prior_images,
        agent=agent,
    )
    warnings = list(state.get("vision_warnings", []))
    if len(app_images) == 0 or len(prior_images) == 0:
        warnings.append(
            {
                "code": "VISION_WARNING",
                "message": "Visual comparison skipped because required images are missing.",
                "node": "analyze_prior_art_visual_node",
            }
        )
    return {
        "visual_report": visual_report.model_dump(),
        "vision_warnings": warnings,
        "status": "running",
        "node_latency_ms": _duration_ms(started_at),
    }
