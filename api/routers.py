from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from agents.base_agent import BaseStructuredAgent, RetryPolicy
from api.errors import ApiError
from models.common import ApiEnvelope, SessionStatus
from models.compare_schemas import (
    AmendmentSuggestionReport,
    DraftBaselineReport,
    FeatureCollisionMatrix,
    PriorArtProfileSet,
    RiskAssessmentReport,
)
from models.draft_schemas import ClaimTraceabilityReport, ClaimsSet, ClaimsSetRevision, Specification, TechSummary
from models.image_schemas import DrawingMap
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
from models.review_schemas import ReviewReport
from models.polish_schemas import (
    AdversarialReviewReport,
    AmplifiedSpecification,
    ClaimArchitecturePlan,
    DiagnosticReport,
    SynergyVault,
)
from services.checkpoint import CheckpointManager
from services.file_store import InMemoryFileStore
from services.llm_factory import build_llm_callable
from services.session_store import InMemorySessionStore, SessionRecord
from services.word_exporter import build_export_docx
from tools.doc_parser import DocumentParser
from tools.rag_search import RAGSearchService
from workflows.draft_workflow import build_draft_workflow, cancel_draft_workflow, resume_draft_workflow
from workflows.draft_workflow import DraftingState as WorkflowDraftState
from workflows.draft_workflow import DraftAgentBundle
from workflows.oa_workflow import OAAgentBundle, OAState as WorkflowOAState, build_oa_workflow
from workflows.compare_workflow import CompareAgentBundle, CompareState as WorkflowCompareState, build_compare_workflow
from workflows.polish_workflow import PolishAgentBundle, PolishState as WorkflowPolishState, build_polish_workflow


logger = logging.getLogger("api.routers")
router = APIRouter(prefix="/api/v1", tags=["draft", "oa", "compare", "polish"])
_SESSION_STORE = InMemorySessionStore()
_CHECKPOINT_MANAGER = CheckpointManager()
_FILE_STORE = InMemoryFileStore(root_dir=os.getenv("UPLOAD_ROOT_DIR", ".runtime/uploads"))
_DOC_PARSER = DocumentParser()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _redact_text(text: str) -> str:
    """Basic content redaction for logs: no full patent text is logged."""
    preview = text[:32].replace("\n", " ")
    return f"<len={len(text)} preview='{preview}...'>"


def _structured_log(event: str, **fields: Any) -> None:
    """Emit structured logs with common observability fields."""
    parts = [f"event={event}"] + [f"{k}={v}" for k, v in fields.items()]
    logger.info(" ".join(parts))


def _append_event(
    *,
    session_id: str,
    event_type: str,
    payload: dict[str, Any],
    store: InMemorySessionStore,
    ) -> None:
    store.append_event(
        session_id,
        {
            "timestamp": _utc_now(),
            "session_id": session_id,
            "type": event_type,
            "payload": payload,
        },
    )


def _resolve_runtime_image_path(image_id: str) -> Path | None:
    """
    Resolve extracted runtime image file by image_id.
    Expected location: <UPLOAD_ROOT_DIR>/images/{image_id}.{ext}
    """
    safe_id = (image_id or "").strip()
    if not safe_id:
        return None
    images_root = (Path(os.getenv("UPLOAD_ROOT_DIR", ".runtime/uploads")).resolve() / "images").resolve()
    if not images_root.exists():
        return None

    for candidate in images_root.glob(f"{safe_id}.*"):
        if candidate.is_file():
            return candidate
    return None


def _extract_examiner_opinion_text(notice_text: str, *, notice_pages: list[str] | None = None) -> tuple[str, bool]:
    """
    Extract the substantive examiner-opinion section from an OA notice.
    Returns (extracted_text, used_extraction).
    """
    import re

    text = (notice_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", False

    signature_markers = (
        "审查员姓名",
        "审查员代码",
        "审查员：",
        "审查员:",
    )
    retrieval_markers = (
        "检索报告",
        "检索式",
        "引用文件",
    )
    opinion_heading_markers = (
        "审查员具体意见",
        "审查员认为",
        "具体意见",
    )

    def _cut_at_earliest(source: str, markers: tuple[str, ...], *, from_pos: int = 0) -> int:
        end_pos = -1
        for marker in markers:
            pos = source.find(marker, from_pos)
            if pos >= 0 and (end_pos < 0 or pos < end_pos):
                end_pos = pos
        return end_pos

    # Strategy 1: page-level fallback (from page 3 onward), then cut signature lines.
    if notice_pages and len(notice_pages) >= 3:
        body = "\n\n".join((page or "").strip() for page in notice_pages[2:]).strip()
        sig_pos = _cut_at_earliest(body, signature_markers)
        candidate = body[:sig_pos].strip() if sig_pos >= 0 else body
        if len(candidate) >= 20:
            return candidate, True

    # Strategy 2: start from the second OA notice title block if present.
    issue_title_pattern = re.compile(r"第\s*\d+\s*次审查意见通知书", flags=re.IGNORECASE)
    issue_matches = list(issue_title_pattern.finditer(text))
    if len(issue_matches) >= 2:
        start_pos = issue_matches[1].end()
        end_pos = _cut_at_earliest(text, retrieval_markers + signature_markers, from_pos=start_pos)
        candidate = text[start_pos:end_pos].strip() if end_pos >= 0 else text[start_pos:].strip()
        if len(candidate) >= 20:
            return candidate, True

    # Strategy 3: explicit examiner-opinion heading.
    start_pos = -1
    for marker in opinion_heading_markers:
        pos = text.find(marker)
        if pos >= 0 and (start_pos < 0 or pos < start_pos):
            start_pos = pos
    if start_pos >= 0:
        end_pos = _cut_at_earliest(text, retrieval_markers + signature_markers, from_pos=start_pos)
        candidate = text[start_pos:end_pos].strip() if end_pos >= 0 else text[start_pos:].strip()
        if len(candidate) >= 20:
            return candidate, True

    return text, False


def _is_doc_image_dependency_error(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "failed to parse doc images" in msg
        and ("microsoft word" in msg or "libreoffice" in msg or "soffice" in msg or "pywin32" in msg)
    )


def _extract_original_claims_text(application_text: str) -> tuple[str, bool, str]:
    """
    Extract the "Claims" section from uploaded application text.
    Returns (claims_text, extracted, strategy).
    """
    text = (application_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", False, "empty_input"

    import re

    # Strategy 1: explicit section heading range.
    start_pattern = re.compile(r"(?:^|\n)\s*\u6743\u5229\u8981\u6c42\u4e66\s*(?:\n|$)")
    start_match = start_pattern.search(text)
    if start_match is not None:
        start_pos = start_match.end()
        # Typical following sections after claims.
        end_pattern = re.compile(
            r"(?:^|\n)\s*(\u8bf4\u660e\u4e66(?:\u6458\u8981)?|\u6458\u8981|"
            r"\u6280\u672f\u9886\u57df|\u80cc\u666f\u6280\u672f|"
            r"\u53d1\u660e\u5185\u5bb9|\u9644\u56fe\u8bf4\u660e|"
            r"\u5177\u4f53\u5b9e\u65bd\u65b9\u5f0f)\s*(?:\n|$)"
        )
        end_match = end_pattern.search(text, start_pos)
        claims_body = text[start_pos : end_match.start() if end_match else len(text)].strip()
        if len(claims_body) >= 20:
            return claims_body, True, "heading_range"

    # Strategy 2: claim-like numbered lines.
    lines = [line.strip() for line in text.split("\n")]
    claim_line_pattern = re.compile(r"^(\u6743\u5229\u8981\u6c42\s*\d+|[1-9]\d{0,2}[\u3001.\uff0e])")
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        if claim_line_pattern.match(line):
            start_idx = idx
            break
    if start_idx is not None:
        collected: list[str] = []
        stop_headings = {
            "\u8bf4\u660e\u4e66",
            "\u8bf4\u660e\u4e66\u6458\u8981",
            "\u6458\u8981",
            "\u6280\u672f\u9886\u57df",
            "\u80cc\u666f\u6280\u672f",
            "\u53d1\u660e\u5185\u5bb9",
            "\u9644\u56fe\u8bf4\u660e",
            "\u5177\u4f53\u5b9e\u65bd\u65b9\u5f0f",
        }
        for line in lines[start_idx:]:
            if line in stop_headings:
                break
            collected.append(line)
        candidate = "\n".join(collected).strip()
        if len(candidate) >= 20:
            return candidate, True, "claim_line_scan"

    return text, False, "fallback_full_text"


def _extract_application_specification_text(application_text: str) -> tuple[str, bool, str]:
    """
    Extract the "Specification" section from uploaded application text.
    Returns (spec_text, extracted, strategy).
    """
    text = (application_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", False, "empty_input"

    import re

    # Strategy 1: explicit specification heading to end.
    start_pattern = re.compile(r"(?:^|\n)\s*\u8bf4\u660e\u4e66\s*(?:\n|$)")
    start_match = start_pattern.search(text)
    if start_match is not None:
        candidate = text[start_match.end() :].strip()
        if len(candidate) >= 40:
            return candidate, True, "heading_range"

    # Strategy 2: start from first specification chapter heading.
    start_pattern_2 = re.compile(
        r"(?:^|\n)\s*(\u6280\u672f\u9886\u57df|\u80cc\u666f\u6280\u672f|"
        r"\u53d1\u660e\u5185\u5bb9|\u9644\u56fe\u8bf4\u660e|"
        r"\u5177\u4f53\u5b9e\u65bd\u65b9\u5f0f)\s*(?:\n|$)"
    )
    start_match_2 = start_pattern_2.search(text)
    if start_match_2 is not None:
        candidate = text[start_match_2.start() :].strip()
        if len(candidate) >= 40:
            return candidate, True, "chapter_heading_range"

    # Strategy 3: if claims heading exists, use content after claims section.
    claims_heading = re.search(r"(?:^|\n)\s*\u6743\u5229\u8981\u6c42\u4e66\s*(?:\n|$)", text)
    if claims_heading is not None:
        candidate = text[claims_heading.end() :].strip()
        if len(candidate) >= 40:
            return candidate, True, "after_claims_heading"

    return text, False, "fallback_full_text"


def get_session_store() -> InMemorySessionStore:
    return _SESSION_STORE


def get_file_store() -> InMemoryFileStore:
    return _FILE_STORE


def _make_stub_llm_callable(payload: Any):
    """MVP stub callable to keep workflow executable before real model integration."""

    def _call(_: str, __: dict[str, Any]) -> Any:
        return payload

    return _call


_DRAFT_STUBS = {'extract_tech': {'source_quotes': ['????', '????', '????'],
                  'background_and_core_problems': ['????', '????', '????'],
                  'core_solution_overview': '????',
                  'detailed_features': [{'feature_name': '????',
                                         'detailed_structure_or_step': '????',
                                         'solved_sub_problem': '????',
                                         'specific_effect': '????'},
                                        {'feature_name': '????',
                                         'detailed_structure_or_step': '????',
                                         'solved_sub_problem': '????',
                                         'specific_effect': '????'},
                                        {'feature_name': '????',
                                         'detailed_structure_or_step': '????',
                                         'solved_sub_problem': '????',
                                         'specific_effect': '????'}],
                  'overall_advantages': ['????', '????', '????']},
 'draft_claims': {'claims': [{'claim_number': 1,
                              'claim_type': 'independent',
                              'depends_on': [],
                              'preamble': '????',
                              'transition': '????',
                              'elements': ['????', '????', '????'],
                              'full_text': '????'},
                             {'claim_number': 2,
                              'claim_type': 'dependent',
                              'depends_on': [1],
                              'preamble': '????',
                              'transition': '????',
                              'elements': ['????', '????'],
                              'full_text': '????'}]},
 'revise_claims': {'claims': [{'claim_number': 1,
                               'claim_type': 'independent',
                               'depends_on': [],
                               'preamble': '????',
                               'transition': '????',
                               'elements': ['????', '????'],
                               'full_text': '????'}]},
 'write_spec': {'title': '????',
                'technical_field': '????',
                'background_art': '????',
                'invention_content': {'technical_problem': '????',
                                      'technical_solution': '????',
                                      'beneficial_effects': '????'},
                'drawings_description': '????',
                'detailed_implementation': {'introductory_boilerplate': '????',
                                            'overall_architecture': '????',
                                            'component_details': [{'feature_name': '????',
                                                                   'structure_and_connection': '????',
                                                                   'working_principle': '????'},
                                                                  {'feature_name': '????',
                                                                   'structure_and_connection': '????',
                                                                   'working_principle': '????'},
                                                                  {'feature_name': '????',
                                                                   'structure_and_connection': '????',
                                                                   'working_principle': '????'}],
                                            'workflow_description': '????',
                                            'alternative_embodiments': '????'}},
 'traceability': {'reports': [{'claim_number': 1,
                               'elements_evidence': [{'feature_text': '????',
                                                      'verbatim_quote': '????',
                                                      'support_level': 'Explicit',
                                                      'reasoning': '????'},
                                                     {'feature_text': '????',
                                                      'verbatim_quote': '????',
                                                      'support_level': 'Explicit',
                                                      'reasoning': '????'},
                                                     {'feature_text': '????',
                                                      'verbatim_quote': '????',
                                                      'support_level': 'Explicit',
                                                      'reasoning': '????'}],
                               'is_fully_supported': True}],
                  'overall_risk_assessment': '????'},
 'logic_review': {'issues': []},
 'drawing_map': {'figures': [],
                 'overall_notes': 'No drawing analysis available in stub mode.',
                 'warnings': ['stub_mode_no_vision']}}
_OA_STUBS = {'oa_parser': {'defects': [{'defect_type': '????',
                            'rejected_claims': [1],
                            'main_cited_docs': ['D1', 'D2'],
                            'feature_mappings': [{'target_feature': '????',
                                                  'prior_art_doc': 'D1',
                                                  'cited_paragraphs': '[0032]',
                                                  'cited_figures': '????',
                                                  'examiner_logic': '????'}],
                            'combination_motivation': '????'}],
               'overall_summary': '????'},
 'multimodal_prior_art': {'supporting_items': [{'target_feature': '????',
                                                'prior_art_text_disclosure': '????',
                                                'prior_art_visual_disclosure': '????',
                                                'amendment_avoidance_warning': '????'}],
                          'disputable_items': [{'target_feature': '????',
                                                'examiner_assertion': '????',
                                                'multimodal_reality_check': '????',
                                                'rebuttal_angle': '????'}],
                          'examiner_conclusion_supported': True,
                          'confidence': 'High',
                          'overall_conclusion': '????'},
 'application_baseline': {'claims_tree': [{'claim_number': 1,
                                           'claim_type': 'independent',
                                           'depends_on': [],
                                           'features': [{'feature_id': '1A',
                                                         'feature_text': '????'},
                                                        {'feature_id': '1B',
                                                         'feature_text': '????'},
                                                        {'feature_id': '1C',
                                                         'feature_text': '????'}]},
                                          {'claim_number': 2,
                                           'claim_type': 'dependent',
                                           'depends_on': [1],
                                           'features': [{'feature_id': '2A',
                                                         'feature_text': '????'}]}],
                          'spec_feature_index': [{'component_or_step_name': '????',
                                                  'reference_numeral': '10',
                                                  'detailed_description': '????',
                                                  'alternative_embodiments': '????',
                                                  'source_paragraph': '????'},
                                                 {'component_or_step_name': '????',
                                                  'reference_numeral': '????',
                                                  'detailed_description': '????',
                                                  'alternative_embodiments': '????',
                                                  'source_paragraph': '????'}],
                          'claim_tree_overview': '????',
                          'normalized_claim_features': ['????', '????', '????'],
                          'fallback_features': ['????', '????'],
                          'specification_feature_index': ['????', '????']},
 'concession_gap': {'overall_strategy_summary': '????',
                    'claim_assessments': [{'claim_number': 1,
                                           'status': 'DEFEATED',
                                           'reasoning': '????'},
                                          {'claim_number': 3,
                                           'status': 'MERGE_CANDIDATE',
                                           'reasoning': '????'}],
                    'recommended_merges': [3],
                    'mining_directives': [{'target_component_or_step': '????',
                                           'technical_gap_to_fill': '????',
                                           'avoidance_warning': '????'}],
                    'failed_claims': [1],
                    'confirmed_points': ['????'],
                    'gap_targets': ['????', '????', '????'],
                    'rationale': '????'},
 'fallback_feature_mining': {'mining_status': 'SUCCESS',
                             'candidates': [{'candidate_id': 'Candidate_A',
                                             'addressed_directive': '????',
                                             'feature_name': '????',
                                             'reference_numeral': '15',
                                             'verbatim_quote': '????',
                                             'source_location': '????',
                                             'gap_filling_rationale': '????'},
                                            {'candidate_id': 'Candidate_B',
                                             'addressed_directive': '????',
                                             'feature_name': '????',
                                             'reference_numeral': '????',
                                             'verbatim_quote': '????',
                                             'source_location': '????',
                                             'gap_filling_rationale': '????'},
                                            {'candidate_id': 'Candidate_C',
                                             'addressed_directive': '????',
                                             'feature_name': '????',
                                             'reference_numeral': '20',
                                             'verbatim_quote': '????',
                                             'source_location': '????',
                                             'gap_filling_rationale': '????'}],
                             'mining_summary': '????'},
 'prior_art_stress_test': {'overall_survival_rate': '????',
                           'tested_features': [{'candidate_id': 'Candidate_A',
                                                'feature_name': '????',
                                                'test_verdict': 'ELIMINATED',
                                                'prior_art_hit_location': '????',
                                                'red_team_reasoning': '????',
                                                'rebuttal_foundation': '????'},
                                               {'candidate_id': 'Candidate_B',
                                                'feature_name': '????',
                                                'test_verdict': 'ELIMINATED',
                                                'prior_art_hit_location': '????',
                                                'red_team_reasoning': '????',
                                                'rebuttal_foundation': '????'},
                                               {'candidate_id': 'Candidate_C',
                                                'feature_name': '????',
                                                'test_verdict': 'SURVIVED',
                                                'prior_art_hit_location': '????',
                                                'red_team_reasoning': '????',
                                                'rebuttal_foundation': '????'}],
                           'survived_candidate_ids': ['Candidate_C'],
                           'summary': '????'},
 'strategy_decision': {'global_decision': 'AMEND_AND_ARGUE',
                       'strategy_rationale': '????',
                       'amendment_plan': {'target_independent_claim': 1,
                                          'amendment_tactic': 'INTRODUCE_SPEC_FEATURE',
                                          'source_dependent_claims': [],
                                          'survived_candidate_ids': ['Candidate_C'],
                                          'amendment_guidance': '????'},
                       'rebuttal_plan': [{'target_claim': 1,
                                          'core_argument_logic': '????',
                                          'evidence_support': '????'}],
                       'action': 'AMEND_AND_ARGUE',
                       'amendment_instruction': '????',
                       'argument_logic': '????',
                       'selected_candidate_ids': ['Candidate_C']},
 'claim_amendment': {'is_amended': True,
                     'amendment_basis_statement': '????',
                     'claim_mappings': [{'original_claim_number': '????',
                                         'new_claim_number': '????',
                                         'amendment_type': 'MODIFIED_WITH_NEW_FEATURE',
                                         'amended_text': '????'}],
                     'final_claims_text': '????',
                     'amended_claims': {'claims': [{'id': 1, 'text': '????'}]},
                     'amendment_log': ['????']},
 'argument_writer': {'amendment_statement': '????',
                     'arguments_by_claim': [{'target_claim': 1,
                                             'closest_prior_art': '????',
                                             'distinguishing_features': '????',
                                             'technical_problem_solved': '????',
                                             'non_obviousness_logic': '????',
                                             'legal_conclusion': '????'}],
                     'final_reply_text': '????',
                     'argument_text': '????',
                     'key_points': ['????', '????', '????']},
 'spec_update': {'requires_spec_update': True,
                 'amendment_items': [{'target_paragraph': '????',
                                      'original_text_snippet': '????',
                                      'amended_text_snippet': '????',
                                      'amendment_reason': '????'}],
                 'article_33_declaration': '????',
                 'applied': True,
                 'changes': ['????'],
                 'updated_excerpt': ''},
 'response_traceability': {'global_go_no_go': 'GO',
                           'support_basis_audit': [{'severity': 'PASS',
                                                    'risk_category': 'A26.4_UNSUPPORTED',
                                                    'problematic_text': '????',
                                                    'audit_reasoning': '????',
                                                    'suggested_remedy': '????'}],
                           'logic_consistency_audit': [{'severity': 'PASS',
                                                        'risk_category': 'LOGIC_INCONSISTENCY',
                                                        'problematic_text': '????',
                                                        'audit_reasoning': '????',
                                                        'suggested_remedy': '????'}],
                           'harmful_admission_audit': [{'severity': 'WARNING',
                                                        'risk_category': 'HARMFUL_ADMISSION',
                                                        'problematic_text': '????',
                                                        'audit_reasoning': '????',
                                                        'suggested_remedy': '????'}],
                           'final_strategy_summary': '????',
                           'claim_support_ok': True,
                           'logic_consistency_ok': True,
                           'findings': [],
                           'final_risk_summary': '????'}}
_COMPARE_STUBS = {'draft_parser': {'claims_tree': [{'claim_number': 1,
                                   'is_independent': True,
                                   'dependency': [],
                                   'atomic_features': [{'feature_id': 'F1.1',
                                                        'verbatim_text': '????',
                                                        'entity_components': ['????'],
                                                        'connection_and_synergy': '????',
                                                        'visual_anchor': {'reference_numeral': '10',
                                                                          'figure_labels': ['????'],
                                                                          'visual_morphology': '????'}},
                                                       {'feature_id': 'F1.2',
                                                        'verbatim_text': '????',
                                                        'entity_components': ['????', '????'],
                                                        'connection_and_synergy': '????',
                                                        'visual_anchor': {'reference_numeral': '20',
                                                                          'figure_labels': ['????',
                                                                                            '????'],
                                                                          'visual_morphology': '????'}}]}],
                  'fallback_feature_index': [{'feature_name': '????',
                                              'verbatim_quote': '????',
                                              'connection_and_synergy': '????',
                                              'source_location': '????',
                                              'visual_anchor': {'reference_numeral': '10',
                                                                'figure_labels': ['????'],
                                                                'visual_morphology': '????'}}]},
 'prior_art_parser': {'comparison_goal': 'patentability',
                      'prior_art_profiles': [{'prior_art_id': 'D1',
                                              'core_technical_problem_solved': '????',
                                              'component_index': [{'component_name': '????',
                                                                   'reference_numeral': '10',
                                                                   'structural_connections_and_mechanisms': '????',
                                                                   'visual_appearance': '????'}],
                                              'figure_library': [{'figure_label': '????',
                                                                  'observed_components': ['10',
                                                                                          '20'],
                                                                  'visual_connections': [{'source_component': '????',
                                                                                          'target_component': '????',
                                                                                          'kinematic_relationship': '????'}]}],
                                              'reading_audit': {'input_image_count': 6,
                                                                'actually_used_image_count': 5,
                                                                'omission_warning': '????'}}],
                      'overall_summary': '????'},
 'matrix_comparison': {'global_conclusion': '????',
                       'prior_art_targeted_report': [{'claim_number': 1,
                                                      'feature_collisions': [{'feature_id': 'F1.1',
                                                                              'prior_art_id': 'D1',
                                                                              'text_evidence': '????',
                                                                              'visual_evidence': '????',
                                                                              'component_match_status': '????',
                                                                              'relationship_match_status': '????',
                                                                              'disclosure_status': 'EXPLICIT',
                                                                              'collision_reasoning': '????'},
                                                                             {'feature_id': 'F1.2',
                                                                              'prior_art_id': 'D1',
                                                                              'text_evidence': '????',
                                                                              'visual_evidence': '????',
                                                                              'component_match_status': '????',
                                                                              'relationship_match_status': '????',
                                                                              'disclosure_status': 'EXPLICIT',
                                                                              'collision_reasoning': '????'}],
                                                      'claim_safety_status': 'DESTROYED'}]},
 'risk_assessment': {'global_risk_summary': '????',
                     'claim_assessments': [{'claim_number': 1,
                                            'novelty_risk': 'FATAL',
                                            'inventiveness_risk': 'HIGH',
                                            'topology_difference_analysis': '????',
                                            'breakthrough_point': '????',
                                            'robust_distinguishing_features': ['????']}],
                     'strategic_amendment_direction': '????'},
 'amendment_suggestion': {'overall_rescue_strategy': '????',
                          'concrete_amendments': [{'target_claim_number': 1,
                                                   'amendment_type': 'INTRODUCE_SPEC_FEATURE',
                                                   'source_feature_name': '????',
                                                   'source_location': '????',
                                                   'verbatim_addition': '????',
                                                   'synergy_and_mechanism_focus': '????',
                                                   'draft_amended_claim_text': '????',
                                                   'expected_overcoming_effect': '????'}],
                          'article_33_compliance_statement': '????'}}
_POLISH_STUBS = {'diagnostic_analyzer': {'overview': '????',
                         'wide_scope_issues': [{'issue_type': '????',
                                                'severity': 'high',
                                                'claim_scope': '????',
                                                'evidence': '????',
                                                'recommendation': '????'}],
                         'dependent_gap_issues': [{'issue_type': '????',
                                                   'severity': 'medium',
                                                   'claim_scope': '????',
                                                   'evidence': '????',
                                                   'recommendation': '????'}],
                         'effect_gap_issues': [{'issue_type': '????',
                                                'severity': 'medium',
                                                'claim_scope': '????',
                                                'evidence': '????',
                                                'recommendation': '????'}],
                         'key_repair_targets': ['????', '????', '????']},
 'synergy_miner': {'vault_summary': '????',
                   'high_value_features': [{'feature_name': '????',
                                            'source_location': '????',
                                            'verbatim_quote': '????',
                                            'connection_and_synergy': '????',
                                            'value_score': 'high',
                                            'recommended_usage': '????'},
                                           {'feature_name': '????',
                                            'source_location': '????',
                                            'verbatim_quote': '????',
                                            'connection_and_synergy': '????',
                                            'value_score': 'medium',
                                            'recommended_usage': '????'}]},
 'claim_architect': {'architecture_summary': '????',
                     'rebuilt_claims': [{'claim_number': 1,
                                         'claim_type': 'independent',
                                         'depends_on': [],
                                         'draft_text': '????',
                                         'added_mechanisms': ['????'],
                                         'source_basis': ['????', '????']},
                                        {'claim_number': 2,
                                         'claim_type': 'dependent',
                                         'depends_on': [1],
                                         'draft_text': '????',
                                         'added_mechanisms': ['????'],
                                         'source_basis': ['????', '????']}],
                     'optimized_claims_text': '????',
                     'article_33_basis': '????'},
 'specification_amplifier': {'amplification_summary': '????',
                             'mechanism_effect_map': [{'feature_name': '????',
                                                       'mechanism_explanation': '????',
                                                       'technical_effect': '????'}],
                             'optimized_specification_text': '????'},
 'adversarial_reviewer': {'pass_gate': True,
                          'issues': [],
                          'return_instruction': '????',
                          'final_judgement': '????'}}

def _minimal_specification_stub() -> dict[str, Any]:
    write_spec = _DRAFT_STUBS.get("write_spec")
    if isinstance(write_spec, dict):
        return cast(dict[str, Any], write_spec)
    return {
        "title": "????",
        "technical_field": "????",
        "background_art": "????",
        "invention_content": {
            "technical_problem": "????",
            "technical_solution": "????",
            "beneficial_effects": "????",
        },
        "drawings_description": "????",
        "detailed_implementation": {
            "introductory_boilerplate": "????",
            "overall_architecture": "????",
            "component_details": [],
            "workflow_description": "????",
            "alternative_embodiments": "????",
        },
    }


def _build_draft_graph_for_runtime(*, llm_runtime: dict[str, Any], llm_api_key: str | None):
    llm_callable = build_llm_callable(
        provider=llm_runtime.get("provider"),
        model=llm_runtime.get("model"),
        vision_model=llm_runtime.get("vision_model"),
        base_url=llm_runtime.get("base_url"),
        api_key=llm_api_key,
        temperature=llm_runtime.get("temperature"),
    )
    bundle = DraftAgentBundle(
        extract_tech_agent=BaseStructuredAgent[TechSummary](
            name="extract_tech_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_DRAFT_STUBS["extract_tech"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        draft_claims_agent=BaseStructuredAgent[ClaimsSet](
            name="draft_claims_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_DRAFT_STUBS["draft_claims"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        revise_claims_agent=BaseStructuredAgent[ClaimsSetRevision](
            name="revise_claims_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_DRAFT_STUBS["revise_claims"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        drawing_analyzer_agent=BaseStructuredAgent[DrawingMap](
            name="drawing_analyzer_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_DRAFT_STUBS["drawing_map"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        traceability_agent=BaseStructuredAgent[ClaimTraceabilityReport](
            name="traceability_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_DRAFT_STUBS["traceability"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        logic_review_agent=BaseStructuredAgent[ReviewReport](
            name="logic_review_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_DRAFT_STUBS["logic_review"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        write_spec_agent=BaseStructuredAgent[Specification](
            name="write_spec_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_minimal_specification_stub()),
            retry_policy=RetryPolicy(max_retries=3),
        ),
    )
    return build_draft_workflow(bundle, checkpointer=_CHECKPOINT_MANAGER.checkpointer), llm_callable is not None


def _build_oa_graph_for_runtime(*, llm_runtime: dict[str, Any], llm_api_key: str | None):
    llm_callable = build_llm_callable(
        provider=llm_runtime.get("provider"),
        model=llm_runtime.get("model"),
        vision_model=llm_runtime.get("vision_model"),
        base_url=llm_runtime.get("base_url"),
        api_key=llm_api_key,
        temperature=llm_runtime.get("temperature"),
    )
    bundle = OAAgentBundle(
        oa_parser_agent=BaseStructuredAgent[OADefectList](
            name="oa_parser_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["oa_parser"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        multimodal_prior_art_agent=BaseStructuredAgent[PriorArtTargetedReadingReport](
            name="multimodal_prior_art_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["multimodal_prior_art"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        application_baseline_agent=BaseStructuredAgent[ApplicationBaselineReport](
            name="application_baseline_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["application_baseline"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        concession_gap_agent=BaseStructuredAgent[ConcessionGapReport](
            name="concession_and_gap_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["concession_gap"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        fallback_feature_miner_agent=BaseStructuredAgent[FallbackFeatureMiningReport](
            name="fallback_feature_miner_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["fallback_feature_mining"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        prior_art_stress_tester_agent=BaseStructuredAgent[PriorArtStressTestReport](
            name="prior_art_stress_tester_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["prior_art_stress_test"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        strategy_decision_agent=BaseStructuredAgent[StrategyDecision](
            name="strategy_decision_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["strategy_decision"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        claim_amendment_agent=BaseStructuredAgent[ClaimAmendmentResult](
            name="claim_amendment_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["claim_amendment"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        argument_writer_agent=BaseStructuredAgent[ArgumentDraft](
            name="argument_writer_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["argument_writer"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        spec_update_agent=BaseStructuredAgent[SpecUpdateNote](
            name="spec_update_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["spec_update"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        response_traceability_agent=BaseStructuredAgent[ResponseTraceabilityReport](
            name="response_traceability_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_OA_STUBS["response_traceability"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        rag_service=RAGSearchService(),
    )
    return build_oa_workflow(bundle, checkpointer=_CHECKPOINT_MANAGER.checkpointer), llm_callable is not None


def _build_compare_graph_for_runtime(*, llm_runtime: dict[str, Any], llm_api_key: str | None):
    llm_callable = build_llm_callable(
        provider=llm_runtime.get("provider"),
        model=llm_runtime.get("model"),
        vision_model=llm_runtime.get("vision_model"),
        base_url=llm_runtime.get("base_url"),
        api_key=llm_api_key,
        temperature=llm_runtime.get("temperature"),
    )
    bundle = CompareAgentBundle(
        draft_parser_agent=BaseStructuredAgent[DraftBaselineReport](
            name="compare_draft_parser_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_COMPARE_STUBS["draft_parser"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        prior_art_parser_agent=BaseStructuredAgent[PriorArtProfileSet](
            name="compare_prior_art_parser_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_COMPARE_STUBS["prior_art_parser"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        matrix_comparison_agent=BaseStructuredAgent[FeatureCollisionMatrix](
            name="compare_matrix_comparison_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_COMPARE_STUBS["matrix_comparison"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        risk_assessment_agent=BaseStructuredAgent[RiskAssessmentReport](
            name="compare_risk_assessment_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_COMPARE_STUBS["risk_assessment"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        amendment_suggestion_agent=BaseStructuredAgent[AmendmentSuggestionReport](
            name="compare_amendment_suggestion_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_COMPARE_STUBS["amendment_suggestion"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        rag_service=RAGSearchService(),
    )
    return build_compare_workflow(bundle, checkpointer=_CHECKPOINT_MANAGER.checkpointer), llm_callable is not None


def _build_polish_graph_for_runtime(*, llm_runtime: dict[str, Any], llm_api_key: str | None):
    llm_callable = build_llm_callable(
        provider=llm_runtime.get("provider"),
        model=llm_runtime.get("model"),
        vision_model=llm_runtime.get("vision_model"),
        base_url=llm_runtime.get("base_url"),
        api_key=llm_api_key,
        temperature=llm_runtime.get("temperature"),
    )
    bundle = PolishAgentBundle(
        diagnostic_agent=BaseStructuredAgent[DiagnosticReport](
            name="polish_diagnostic_analyzer_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_POLISH_STUBS["diagnostic_analyzer"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        synergy_miner_agent=BaseStructuredAgent[SynergyVault](
            name="polish_synergy_miner_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_POLISH_STUBS["synergy_miner"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        claim_architect_agent=BaseStructuredAgent[ClaimArchitecturePlan](
            name="polish_claim_architect_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_POLISH_STUBS["claim_architect"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        specification_amplifier_agent=BaseStructuredAgent[AmplifiedSpecification](
            name="polish_specification_amplifier_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_POLISH_STUBS["specification_amplifier"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
        adversarial_reviewer_agent=BaseStructuredAgent[AdversarialReviewReport](
            name="polish_adversarial_reviewer_agent",
            llm_callable=llm_callable or _make_stub_llm_callable(_POLISH_STUBS["adversarial_reviewer"]),
            retry_policy=RetryPolicy(max_retries=3),
        ),
    )
    return build_polish_workflow(bundle, checkpointer=_CHECKPOINT_MANAGER.checkpointer), llm_callable is not None


_DEFAULT_DRAFT_GRAPH, _ = _build_draft_graph_for_runtime(llm_runtime={}, llm_api_key=None)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    Optional API-key protection:
    - If APP_API_KEY is empty, auth is disabled for local development.
    - If set, every request must provide matching X-API-Key.
    """
    expected = os.getenv("APP_API_KEY", "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise ApiError(
            http_status=401,
            code="E401_UNAUTHORIZED",
            message="Invalid API key.",
            session_id="unknown",
            retryable=False,
            details={},
        )


class DraftStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str = Field(..., min_length=1)
    disclosure_text: str | None = Field(default=None, min_length=1)
    disclosure_file_id: str | None = Field(default=None, min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DraftContinueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1)
    approved_claims: ClaimsSet | None = None
    apply_auto_claim_revision: bool = False
    approved_specification: Specification | None = None
    apply_targeted_revision: bool = True
    revision_instruction: str | None = None


class OAStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str = Field(..., min_length=1)
    oa_text: str | None = Field(default=None, min_length=1)
    oa_notice_file_id: str | None = Field(default=None, min_length=1)
    application_file_id: str | None = Field(default=None, min_length=1)
    prior_art_file_ids: list[str] = Field(default_factory=list)
    original_claims: dict[str, Any] = Field(default_factory=dict)
    prior_arts_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompareStartRequest(BaseModel):
    # Compatibility-first for desktop clients with stale payload shapes.
    model_config = ConfigDict(extra="ignore")

    idempotency_key: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    comparison_goal: str = "patentability"
    application_file_id: str | None = Field(default=None, min_length=1)
    prior_art_file_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("prior_art_file_ids", "prior_arts_file_ids"),
    )
    original_claims: dict[str, Any] = Field(default_factory=dict)
    application_specification: dict[str, Any] = Field(default_factory=dict)
    prior_arts_paths: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("prior_arts_paths", "prior_art_paths"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("comparison_goal", mode="before")
    @classmethod
    def _normalize_goal(cls, value: Any) -> str:
        if not isinstance(value, str):
            return "patentability"
        norm = value.strip().lower()
        return "patentability" if norm != "patentability" else norm

    @field_validator("application_file_id", mode="before")
    @classmethod
    def _normalize_application_file_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return None

    @field_validator("prior_art_file_ids", mode="before")
    @classmethod
    def _normalize_prior_art_file_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        out.append(text)
            return out
        return []

    @field_validator("prior_arts_paths", mode="before")
    @classmethod
    def _normalize_prior_paths(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        out.append(text)
            return out
        return []


class PolishStartRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    idempotency_key: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    application_file_id: str | None = Field(default=None, min_length=1)
    original_claims: dict[str, Any] = Field(default_factory=dict)
    application_specification: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("original_claims", "application_specification", mode="before")
    @classmethod
    def _normalize_dict_fields(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}


class FilePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: Literal["draft", "oa", "compare", "polish"]
    disclosure_file_id: str | None = Field(default=None, min_length=1)
    oa_notice_file_id: str | None = Field(default=None, min_length=1)
    application_file_id: str | None = Field(default=None, min_length=1)
    prior_art_file_ids: list[str] = Field(default_factory=list)


def _normalize_provider_name(provider: str | None) -> str | None:
    raw = (provider or "").strip().lower()
    if not raw:
        return None
    if raw == "anthropic":
        return "claude"
    return raw


def _parse_runtime_temperature(raw: str | None) -> float | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(2.0, value))


def _parse_runtime_max_reflections(raw: str | None) -> int | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return max(1, min(10, value))


def _read_llm_runtime_from_headers(
    *,
    x_llm_provider: str | None,
    x_llm_model: str | None,
    x_llm_vision_model: str | None,
    x_llm_base_url: str | None,
    x_llm_api_key: str | None,
    x_llm_temperature: str | None = None,
    x_agent_max_reflections: str | None = None,
    x_agent_context_window_limit: str | None = None,
    x_legal_jurisdiction_baseline: str | None = None,
    x_legal_claim_formatting: str | None = None,
) -> dict[str, Any]:
    """
    Extract runtime LLM settings from headers.
    API key is represented only as presence flag to honor "API key not persisted".
    """
    provider = _normalize_provider_name(x_llm_provider)
    model = (x_llm_model or "").strip() or None
    vision_model = (x_llm_vision_model or "").strip() or None
    base_url = (x_llm_base_url or "").strip() or None
    temperature = _parse_runtime_temperature(x_llm_temperature)
    max_reflections = _parse_runtime_max_reflections(x_agent_max_reflections)
    context_window_limit = (x_agent_context_window_limit or "").strip() or None
    jurisdiction_baseline = (x_legal_jurisdiction_baseline or "").strip() or None
    claim_formatting = (x_legal_claim_formatting or "").strip() or None

    # Fallback to env defaults so local desktop flow can use real LLM without explicit headers.
    if provider is None:
        env_provider = _normalize_provider_name(os.getenv("LLM_PROVIDER", "").strip())
        if env_provider:
            provider = env_provider
        elif os.getenv("OPENAI_API_KEY", "").strip():
            provider = "openai"

    if model is None:
        if provider == "openai":
            model = os.getenv("OPENAI_MODEL", "").strip() or None
        else:
            model = os.getenv("LLM_MODEL", "").strip() or None

    if vision_model is None:
        if provider == "openai":
            vision_model = os.getenv("OPENAI_VISION_MODEL", "").strip() or None
        else:
            vision_model = os.getenv("LLM_VISION_MODEL", "").strip() or None

    if base_url is None:
        if provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        else:
            base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    api_key_configured = bool((x_llm_api_key or "").strip())
    if not api_key_configured:
        provider_env_map = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "doubao": "ARK_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "glm": "ZHIPUAI_API_KEY",
        }
        api_env = provider_env_map.get((provider or "").strip().lower())
        api_key_configured = bool(api_env and os.getenv(api_env, "").strip())

    return {
        "provider": provider,
        "model": model,
        "vision_model": vision_model,
        "base_url": base_url,
        "temperature": temperature,
        "max_reflections": max_reflections,
        "context_window_limit": context_window_limit,
        "jurisdiction_baseline": jurisdiction_baseline,
        "claim_formatting": claim_formatting,
        "api_key_configured": api_key_configured,
    }


def _merge_llm_runtime(
    *,
    header_runtime: dict[str, Any],
    session_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = session_runtime or {}
    return {
        "provider": header_runtime.get("provider") or base.get("provider"),
        "model": header_runtime.get("model") or base.get("model"),
        "vision_model": header_runtime.get("vision_model") or base.get("vision_model"),
        "base_url": header_runtime.get("base_url") or base.get("base_url"),
        "temperature": header_runtime.get("temperature")
        if header_runtime.get("temperature") is not None
        else base.get("temperature"),
        "max_reflections": header_runtime.get("max_reflections")
        if header_runtime.get("max_reflections") is not None
        else base.get("max_reflections"),
        "context_window_limit": header_runtime.get("context_window_limit") or base.get("context_window_limit"),
        "jurisdiction_baseline": header_runtime.get("jurisdiction_baseline") or base.get("jurisdiction_baseline"),
        "claim_formatting": header_runtime.get("claim_formatting") or base.get("claim_formatting"),
        "api_key_configured": bool(header_runtime.get("api_key_configured") or base.get("api_key_configured")),
    }


def _llm_requested(runtime: dict[str, Any]) -> bool:
    return bool(runtime.get("provider") or runtime.get("model") or runtime.get("vision_model") or runtime.get("base_url"))


def _build_doc_parser_for_runtime(*, llm_runtime: dict[str, Any], llm_api_key: str | None) -> DocumentParser:
    if not _llm_requested(llm_runtime):
        return _DOC_PARSER
    return DocumentParser(llm_runtime=llm_runtime, llm_api_key=llm_api_key)


def _parse_uploaded_file(
    file_id: str,
    store: InMemoryFileStore,
    *,
    parser: DocumentParser | None = None,
) -> tuple[str, str, str]:
    record = store.get(file_id)
    if record is None:
        raise ApiError(
            http_status=404,
            code="E404_FILE_NOT_FOUND",
            message="Uploaded file not found.",
            session_id="unknown",
            details={"file_id": file_id},
        )
    try:
        parsed = (parser or _DOC_PARSER).parse_file(record.path)
    except FileNotFoundError as exc:
        raise ApiError(
            http_status=404,
            code="E404_FILE_NOT_FOUND",
            message=str(exc),
            session_id="unknown",
            details={"file_id": file_id, "filename": record.filename},
        ) from exc
    except ValueError as exc:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message=str(exc),
            session_id="unknown",
            details={"file_id": file_id, "filename": record.filename},
            retryable=False,
        ) from exc
    return parsed.text, parsed.source_path, record.filename


def _parse_uploaded_pdf_pages(file_id: str, store: InMemoryFileStore) -> list[str]:
    """Parse uploaded PDF into per-page text. Returns [] for non-PDF files."""
    record = store.get(file_id)
    if record is None:
        raise ApiError(
            http_status=404,
            code="E404_FILE_NOT_FOUND",
            message="Uploaded file not found.",
            session_id="unknown",
            details={"file_id": file_id},
        )

    file_path = Path(record.path)
    if file_path.suffix.lower() != ".pdf":
        return []

    pages: list[str] = []
    try:
        with fitz.open(record.path) as doc:
            for page in doc:
                page_text = page.get_text("text") or ""
                pages.append(page_text.replace("\r\n", "\n").replace("\r", "\n").strip())
    except FileNotFoundError as exc:
        raise ApiError(
            http_status=404,
            code="E404_FILE_NOT_FOUND",
            message=str(exc),
            session_id="unknown",
            details={"file_id": file_id, "filename": record.filename},
        ) from exc
    except Exception as exc:  # pragma: no cover - parser backend failures
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message=f"Failed to parse PDF pages: {exc}",
            session_id="unknown",
            details={"file_id": file_id, "filename": record.filename},
            retryable=False,
        ) from exc

    return pages


def _extract_uploaded_images(file_id: str, store: InMemoryFileStore) -> list[dict[str, Any]]:
    record = store.get(file_id)
    if record is None:
        raise ApiError(
            http_status=404,
            code="E404_FILE_NOT_FOUND",
            message="Uploaded file not found.",
            session_id="unknown",
            details={"file_id": file_id},
        )
    try:
        images = _DOC_PARSER.extract_images(record.path, source_file_id=record.file_id)
    except FileNotFoundError as exc:
        raise ApiError(
            http_status=404,
            code="E404_FILE_NOT_FOUND",
            message=str(exc),
            session_id="unknown",
            details={"file_id": file_id, "filename": record.filename},
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if _is_doc_image_dependency_error(message):
            # Docker/Linux environments may not have Word/LibreOffice for legacy DOC image extraction.
            # Degrade to text-only flow instead of hard failing the upload/workflow.
            _structured_log(
                "doc_image_extract_degraded",
                file_id=file_id,
                filename=record.filename,
                reason=message,
            )
            return []
        exceeds_limit = "exceeds limit" in message.lower()
        raise ApiError(
            http_status=413 if exceeds_limit else 400,
            code="E413_IMAGE_LIMIT_EXCEEDED" if exceeds_limit else "E400_INVALID_INPUT",
            message=message,
            session_id="unknown",
            details={"file_id": file_id, "filename": record.filename},
            retryable=False,
        ) from exc
    return [item.model_dump() for item in images]


@router.post("/files/upload", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
async def upload_file(
    file: UploadFile = File(...),
    purpose: Literal["draft_disclosure", "oa_notice", "application", "prior_art"] | None = Form(default=None),
    store: InMemoryFileStore = Depends(get_file_store),
) -> ApiEnvelope:
    request_id = str(uuid4())
    max_size_bytes = int(os.getenv("UPLOAD_MAX_SIZE_BYTES", str(20 * 1024 * 1024)))
    payload = await file.read()
    if not file.filename:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="Filename is required.",
            session_id="unknown",
            details={},
            retryable=False,
        )
    if len(payload) > max_size_bytes:
        raise ApiError(
            http_status=413,
            code="E413_FILE_TOO_LARGE",
            message="Uploaded file exceeds size limit.",
            session_id="unknown",
            details={"max_size_bytes": max_size_bytes, "size_bytes": len(payload)},
            retryable=False,
        )
    try:
        record = store.save_file(
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            data=payload,
            purpose=purpose,
        )
    except ValueError as exc:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message=str(exc),
            session_id="unknown",
            details={"filename": file.filename},
            retryable=False,
        ) from exc

    # Pre-extract embedded images to enforce media limits early and return summary metadata.
    try:
        extracted_images = _DOC_PARSER.extract_images(record.path, source_file_id=record.file_id)
    except (FileNotFoundError, ValueError) as exc:
        message = str(exc)
        if isinstance(exc, ValueError) and _is_doc_image_dependency_error(message):
            # Allow upload to proceed as text-only when DOC image extraction dependencies are unavailable.
            _structured_log(
                "upload_doc_image_extract_degraded",
                file_id=record.file_id,
                filename=record.filename,
                reason=message,
            )
            extracted_images = []
        else:
            store.delete(record.file_id)
            exceeds_limit = "exceeds limit" in message.lower()
            raise ApiError(
                http_status=413 if exceeds_limit else 400,
                code="E413_IMAGE_LIMIT_EXCEEDED" if exceeds_limit else "E400_INVALID_INPUT",
                message=message,
                session_id="unknown",
                details={"filename": record.filename},
                retryable=False,
            ) from exc

    file_kind = "mixed" if len(extracted_images) > 0 else "text"

    return ApiEnvelope(
        request_id=request_id,
        session_id=record.file_id,
        status=SessionStatus.completed,
        data={
            "file_id": record.file_id,
            "filename": record.filename,
            "content_type": record.content_type,
            "size_bytes": record.size_bytes,
            "purpose": record.purpose,
            "file_kind": file_kind,
            "image_count": len(extracted_images),
        },
        error=None,
    )


def _infer_status(output: dict[str, Any]) -> SessionStatus:
    if "__interrupt__" in output:
        return SessionStatus.waiting_human
    raw_status = str(output.get("status", "")).strip()
    if raw_status == SessionStatus.completed.value:
        return SessionStatus.completed
    if raw_status == SessionStatus.failed.value:
        return SessionStatus.failed
    if raw_status == SessionStatus.cancelled.value:
        return SessionStatus.cancelled
    if raw_status == SessionStatus.waiting_human.value:
        return SessionStatus.waiting_human
    return SessionStatus.running


def _infer_waiting_step(output: dict[str, Any]) -> str:
    fallback_step = str(output.get("current_step", "human_review_node")).strip() or "human_review_node"
    interrupts = output.get("__interrupt__")
    if not isinstance(interrupts, list) or len(interrupts) == 0:
        return fallback_step
    first_interrupt = interrupts[0]
    value = getattr(first_interrupt, "value", None)
    if not isinstance(value, dict):
        return fallback_step
    event = str(value.get("event", "")).strip()
    event_to_step = {
        "hitl_required": "human_review_node",
        "spec_review_required": "spec_review_node",
        "claims_revision_required": "claims_revise_review_node",
    }
    return event_to_step.get(event, fallback_step)


@router.post("/draft/start", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def start_draft(
    request: DraftStartRequest,
    store: InMemorySessionStore = Depends(get_session_store),
    file_store: InMemoryFileStore = Depends(get_file_store),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
    x_llm_vision_model: str | None = Header(default=None, alias="X-LLM-Vision-Model"),
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_temperature: str | None = Header(default=None, alias="X-LLM-Temperature"),
    x_agent_max_reflections: str | None = Header(default=None, alias="X-Agent-Max-Reflections"),
    x_agent_context_window_limit: str | None = Header(default=None, alias="X-Agent-Context-Window-Limit"),
    x_legal_jurisdiction_baseline: str | None = Header(default=None, alias="X-Legal-Jurisdiction-Baseline"),
    x_legal_claim_formatting: str | None = Header(default=None, alias="X-Legal-Claim-Formatting"),
) -> ApiEnvelope:
    session_id = str(uuid4())
    request_id = str(uuid4())
    trace_id = request_id
    llm_runtime = _read_llm_runtime_from_headers(
        x_llm_provider=x_llm_provider,
        x_llm_model=x_llm_model,
        x_llm_vision_model=x_llm_vision_model,
        x_llm_base_url=x_llm_base_url,
        x_llm_api_key=x_llm_api_key,
        x_llm_temperature=x_llm_temperature,
        x_agent_max_reflections=x_agent_max_reflections,
        x_agent_context_window_limit=x_agent_context_window_limit,
        x_legal_jurisdiction_baseline=x_legal_jurisdiction_baseline,
        x_legal_claim_formatting=x_legal_claim_formatting,
    )
    draft_graph, is_real_llm = _build_draft_graph_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)
    if _llm_requested(llm_runtime) and not is_real_llm:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="LLM is configured but API key is missing or invalid for this provider.",
            session_id=session_id,
            details={"llm_runtime": llm_runtime},
            retryable=False,
        )
    parser = _build_doc_parser_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)
    disclosure_text = request.disclosure_text or ""
    disclosure_source = "inline"
    disclosure_images: list[dict[str, Any]] = []
    if request.disclosure_file_id:
        disclosure_text, _, disclosure_filename = _parse_uploaded_file(
            request.disclosure_file_id,
            file_store,
            parser=parser,
        )
        disclosure_images = _extract_uploaded_images(request.disclosure_file_id, file_store)
        disclosure_source = disclosure_filename
    if not disclosure_text.strip():
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="Either disclosure_text or disclosure_file_id is required.",
            session_id=session_id,
            details={},
            retryable=False,
        )

    _structured_log(
        "draft_start_received",
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        disclosure=_redact_text(disclosure_text),
    )

    state: WorkflowDraftState = {
        "session_id": session_id,
        "trace_id": trace_id,
        "status": "running",
        "disclosure_text": disclosure_text,
        "disclosure_images": disclosure_images,
        "tech_summary": None,
        "claims": None,
        "drawing_map": None,
        "claim_traceability": None,
        "approved_claims": None,
        "specification": None,
        "vision_warnings": [],
        "review_issues": [],
        "current_step": "extract_tech_node",
        "error_count": 0,
        "claim_revision_count": 0,
        "apply_auto_claim_revision": None,
        "last_error": None,
        "model_name": llm_runtime.get("model") or "stub",
        "node_latency_ms": 0,
    }

    try:
        output = _CHECKPOINT_MANAGER.invoke(graph=draft_graph, state=cast(dict[str, Any], state), thread_id=session_id)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(
            http_status=500,
            code="E500_TOOL_FAILURE",
            message=str(exc),
            session_id=session_id,
            retryable=True,
            details={"step": "draft_start"},
        ) from exc

    status = _infer_status(output)
    waiting_step = _infer_waiting_step(output) if status == SessionStatus.waiting_human else None
    data = {
        "current_step": waiting_step or output.get("current_step", "extract_tech_node"),
        "disclosure_text": disclosure_text,
        "disclosure_images_meta": disclosure_images,
        "tech_summary": output.get("tech_summary"),
        "claims": output.get("claims"),
        "drawing_map": output.get("drawing_map"),
        "claim_traceability": output.get("claim_traceability"),
        "specification": output.get("specification"),
        "vision_warnings": output.get("vision_warnings", []),
        "review_issues": output.get("review_issues", []),
        "claim_revision_count": output.get("claim_revision_count", 0),
        "apply_auto_claim_revision": output.get("apply_auto_claim_revision"),
        "metadata": request.metadata,
        "disclosure_source": disclosure_source,
        "llm_runtime": llm_runtime,
        "llm_mode": "real" if is_real_llm else "stub",
        "vision_mode": "real" if is_real_llm else "stub",
    }
    record = SessionRecord(
        session_id=session_id,
        request_id=request_id,
        status=status,
        data=data,
    )
    store.upsert(record)

    _append_event(
        session_id=session_id,
        event_type="node_finished",
        payload={"node_name": "draft_claims_node"},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="image_extracted",
        payload={"workflow": "draft", "image_count": len(disclosure_images)},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="drawing_map_ready",
        payload={"workflow": "draft", "has_drawing_map": bool(output.get("drawing_map"))},
        store=store,
    )
    if output.get("vision_warnings"):
        _append_event(
            session_id=session_id,
            event_type="vision_fallback",
            payload={"workflow": "draft", "warnings": output.get("vision_warnings", [])},
            store=store,
        )
    if status == SessionStatus.waiting_human:
        _append_event(
            session_id=session_id,
            event_type="hitl_required",
            payload={
                "current_step": waiting_step or "human_review_node",
                "claims": output.get("claims"),
                "claim_traceability": output.get("claim_traceability"),
            },
            store=store,
        )

    return ApiEnvelope(
        request_id=request_id,
        session_id=session_id,
        status=status,
        data=data,
        error=None,
    )


@router.post("/draft/continue", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def continue_draft(
    request: DraftContinueRequest,
    store: InMemorySessionStore = Depends(get_session_store),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
    x_llm_vision_model: str | None = Header(default=None, alias="X-LLM-Vision-Model"),
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_temperature: str | None = Header(default=None, alias="X-LLM-Temperature"),
    x_agent_max_reflections: str | None = Header(default=None, alias="X-Agent-Max-Reflections"),
    x_agent_context_window_limit: str | None = Header(default=None, alias="X-Agent-Context-Window-Limit"),
    x_legal_jurisdiction_baseline: str | None = Header(default=None, alias="X-Legal-Jurisdiction-Baseline"),
    x_legal_claim_formatting: str | None = Header(default=None, alias="X-Legal-Claim-Formatting"),
) -> ApiEnvelope:
    session = store.get(request.session_id)
    if session is None:
        raise ApiError(
            http_status=404,
            code="E404_SESSION_NOT_FOUND",
            message="Session not found.",
            session_id=request.session_id,
            details={"session_id": request.session_id},
        )
    if session.status != SessionStatus.waiting_human:
        raise ApiError(
            http_status=409,
            code="E409_HITL_STATE_CONFLICT",
            message="Session is not waiting for human review.",
            session_id=request.session_id,
            details={"status": session.status.value},
        )

    _structured_log(
        "draft_continue_received",
        request_id=session.request_id,
        session_id=session.session_id,
        trace_id=session.request_id,
    )
    _append_event(
        session_id=session.session_id,
        event_type="node_started",
        payload={"node_name": "write_spec_node"},
        store=store,
    )
    header_llm_runtime = _read_llm_runtime_from_headers(
        x_llm_provider=x_llm_provider,
        x_llm_model=x_llm_model,
        x_llm_vision_model=x_llm_vision_model,
        x_llm_base_url=x_llm_base_url,
        x_llm_api_key=x_llm_api_key,
        x_llm_temperature=x_llm_temperature,
        x_agent_max_reflections=x_agent_max_reflections,
        x_agent_context_window_limit=x_agent_context_window_limit,
        x_legal_jurisdiction_baseline=x_legal_jurisdiction_baseline,
        x_legal_claim_formatting=x_legal_claim_formatting,
    )
    session_llm_runtime = (
        session.data.get("llm_runtime", {}) if isinstance(session.data.get("llm_runtime", {}), dict) else {}
    )
    llm_runtime = _merge_llm_runtime(header_runtime=header_llm_runtime, session_runtime=session_llm_runtime)
    draft_graph, is_real_llm = _build_draft_graph_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)
    if _llm_requested(llm_runtime) and not is_real_llm:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="LLM is configured for this session but API key is missing in continue request.",
            session_id=session.session_id,
            details={"llm_runtime": llm_runtime},
            retryable=False,
        )
    current_step = str(session.data.get("current_step", "")).strip()
    review_issues = session.data.get("review_issues", [])
    is_claims_revise_stage = current_step == "claims_revise_review_node"
    is_claims_review_stage = current_step == "human_review_node"
    is_spec_review_stage = current_step == "spec_review_node" or (
        isinstance(review_issues, list) and len(review_issues) > 0
    )
    if (
        is_spec_review_stage
        and request.apply_targeted_revision
        and request.approved_specification is None
        and not is_real_llm
    ):
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="\u5f53\u524d\u4e3a Stub \u6a21\u5f0f\uff08\u672a\u542f\u7528\u771f\u5b9e LLM\uff09\uff0c\u4e0d\u80fd\u81ea\u52a8\u901a\u8fc7\u8bf4\u660e\u4e66\u4fee\u8ba2\u3002\u8bf7\u63d0\u4ea4 approved_specification\u3002",
            session_id=session.session_id,
            details={
                "current_step": current_step or "unknown",
                "llm_mode": "stub",
            },
            retryable=False,
        )

    try:
        resume_payload: dict[str, Any] = {}
        if request.approved_claims is not None:
            resume_payload["approved_claims"] = request.approved_claims.model_dump()
        if request.apply_auto_claim_revision:
            resume_payload["apply_auto_claim_revision"] = True
        if is_claims_review_stage and "approved_claims" not in resume_payload:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="human_review_node \u9700\u8981\u63d0\u4ea4 approved_claims\u3002",
                session_id=session.session_id,
                details={"current_step": current_step},
                retryable=False,
            )
        if is_claims_revise_stage and not request.apply_auto_claim_revision and "approved_claims" not in resume_payload:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="claims_revise_review_node \u9700\u8981\u4e8c\u9009\u4e00\uff1aapply_auto_claim_revision=true \u6216\u63d0\u4ea4 approved_claims\u3002",
                session_id=session.session_id,
                details={"current_step": current_step},
                retryable=False,
            )
        resume_payload["apply_targeted_revision"] = request.apply_targeted_revision
        if request.approved_specification is not None:
            resume_payload["approved_specification"] = request.approved_specification.model_dump()
        if request.revision_instruction:
            resume_payload["revision_instruction"] = request.revision_instruction
        output = resume_draft_workflow(
            draft_graph,
            thread_id=session.session_id,
            resume_payload=resume_payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise ApiError(
            http_status=500,
            code="E500_TOOL_FAILURE",
            message=str(exc),
            session_id=session.session_id,
            retryable=True,
            details={"step": "draft_continue"},
        ) from exc

    final_status = _infer_status(output)
    waiting_step = _infer_waiting_step(output) if final_status == SessionStatus.waiting_human else None
    session.status = final_status
    session.data = {
        "current_step": waiting_step or output.get("current_step", "logic_review_node"),
        "disclosure_text": session.data.get("disclosure_text"),
        "disclosure_images_meta": session.data.get("disclosure_images_meta", []),
        "tech_summary": output.get("tech_summary"),
        "claims": output.get("claims"),
        "drawing_map": output.get("drawing_map"),
        "claim_traceability": output.get("claim_traceability"),
        "specification": output.get("specification", session.data.get("specification")),
        "vision_warnings": output.get("vision_warnings", session.data.get("vision_warnings", [])),
        "review_issues": output.get("review_issues", []),
        "claim_revision_count": output.get("claim_revision_count", session.data.get("claim_revision_count", 0)),
        "apply_auto_claim_revision": output.get(
            "apply_auto_claim_revision",
            session.data.get("apply_auto_claim_revision"),
        ),
        "llm_runtime": llm_runtime,
        "llm_mode": "real" if is_real_llm else "stub",
        "vision_mode": "real" if is_real_llm else "stub",
    }
    store.upsert(session)

    _append_event(
        session_id=session.session_id,
        event_type="node_finished",
        payload={"node_name": "logic_review_node"},
        store=store,
    )
    if output.get("vision_warnings"):
        _append_event(
            session_id=session.session_id,
            event_type="vision_fallback",
            payload={"workflow": "draft", "warnings": output.get("vision_warnings", [])},
            store=store,
        )
    if final_status == SessionStatus.waiting_human:
        _append_event(
            session_id=session.session_id,
            event_type="hitl_required",
            payload={
                "current_step": waiting_step or output.get("current_step"),
                "review_issues": output.get("review_issues", []),
                "claims": output.get("claims"),
                "claim_traceability": output.get("claim_traceability"),
            },
            store=store,
        )
    else:
        _append_event(
            session_id=session.session_id,
            event_type="session_completed" if final_status == SessionStatus.completed else "session_failed",
            payload={"status": final_status.value},
            store=store,
        )

    return ApiEnvelope(
        request_id=session.request_id,
        session_id=session.session_id,
        status=session.status,
        data=session.data,
        error=None,
    )


@router.post("/oa/start", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def start_oa(
    request: OAStartRequest,
    store: InMemorySessionStore = Depends(get_session_store),
    file_store: InMemoryFileStore = Depends(get_file_store),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
    x_llm_vision_model: str | None = Header(default=None, alias="X-LLM-Vision-Model"),
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_temperature: str | None = Header(default=None, alias="X-LLM-Temperature"),
    x_agent_max_reflections: str | None = Header(default=None, alias="X-Agent-Max-Reflections"),
    x_agent_context_window_limit: str | None = Header(default=None, alias="X-Agent-Context-Window-Limit"),
    x_legal_jurisdiction_baseline: str | None = Header(default=None, alias="X-Legal-Jurisdiction-Baseline"),
    x_legal_claim_formatting: str | None = Header(default=None, alias="X-Legal-Claim-Formatting"),
) -> ApiEnvelope:
    session_id = str(uuid4())
    request_id = str(uuid4())
    trace_id = request_id
    llm_runtime = _read_llm_runtime_from_headers(
        x_llm_provider=x_llm_provider,
        x_llm_model=x_llm_model,
        x_llm_vision_model=x_llm_vision_model,
        x_llm_base_url=x_llm_base_url,
        x_llm_api_key=x_llm_api_key,
        x_llm_temperature=x_llm_temperature,
        x_agent_max_reflections=x_agent_max_reflections,
        x_agent_context_window_limit=x_agent_context_window_limit,
        x_legal_jurisdiction_baseline=x_legal_jurisdiction_baseline,
        x_legal_claim_formatting=x_legal_claim_formatting,
    )
    oa_graph, is_real_llm = _build_oa_graph_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)
    if _llm_requested(llm_runtime) and not is_real_llm:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="LLM is configured but API key is missing or invalid for this provider.",
            session_id=session_id,
            details={"llm_runtime": llm_runtime},
            retryable=False,
        )
    parser = _build_doc_parser_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)
    oa_text = request.oa_text or ""
    oa_source_text = oa_text
    original_claims = dict(request.original_claims)
    original_claims_extracted = False
    original_claims_strategy = "request_payload"
    application_specification: dict[str, Any] = {}
    application_specification_extracted = False
    application_specification_strategy = "not_available"
    prior_arts_paths = list(request.prior_arts_paths)
    application_images: list[dict[str, Any]] = []
    prior_art_images: list[dict[str, Any]] = []
    if request.oa_notice_file_id or request.application_file_id or request.prior_art_file_ids:
        if not request.oa_notice_file_id or not request.application_file_id:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="oa_notice_file_id and application_file_id are required when using file upload mode.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        if len(request.prior_art_file_ids) < 1:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="At least one prior_art_file_id is required when using file upload mode.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        notice_text, _, notice_name = _parse_uploaded_file(
            request.oa_notice_file_id,
            file_store,
            parser=parser,
        )
        notice_pages = _parse_uploaded_pdf_pages(request.oa_notice_file_id, file_store)
        focused_notice_text, used_notice_extraction = _extract_examiner_opinion_text(
            notice_text,
            notice_pages=notice_pages,
        )
        application_text, _, application_name = _parse_uploaded_file(
            request.application_file_id,
            file_store,
            parser=parser,
        )
        claims_text, claims_extracted, claims_strategy = _extract_original_claims_text(application_text)
        original_claims_extracted = claims_extracted
        original_claims_strategy = claims_strategy
        spec_text, spec_extracted, spec_strategy = _extract_application_specification_text(application_text)
        application_specification_extracted = spec_extracted
        application_specification_strategy = spec_strategy
        application_specification = {
            "source": application_name,
            "text": spec_text,
            "extracted": spec_extracted,
            "strategy": spec_strategy,
        }
        application_images = _extract_uploaded_images(request.application_file_id, file_store)
        prior_parts: list[str] = []
        resolved_prior_paths: list[str] = []
        for index, file_id in enumerate(request.prior_art_file_ids, start=1):
            prior_text, prior_path, prior_name = _parse_uploaded_file(file_id, file_store, parser=parser)
            prior_art_images.extend(_extract_uploaded_images(file_id, file_store))
            prior_parts.append(f"D{index} ({prior_name}):\n{prior_text}")
            resolved_prior_paths.append(prior_path)
        oa_source_text = "\n".join(
            [
                "[\u5ba1\u67e5\u610f\u89c1\u901a\u77e5\u4e66]",
                notice_text,
                "",
                "[\u7533\u8bf7\u6587\u4ef6]",
                application_text,
                "",
                "[\u5bf9\u6bd4\u6587\u4ef6]",
                *prior_parts,
            ]
        )
        oa_text = "\n".join(
            [
                "[\u5ba1\u67e5\u610f\u89c1\u901a\u77e5\u4e66_\u5bf9\u6bd4\u805a\u7126\u6458\u8981]",
                focused_notice_text,
                "",
                "[\u7533\u8bf7\u6587\u4ef6]",
                application_text,
                "",
                "[\u5bf9\u6bd4\u6587\u4ef6]",
                *prior_parts,
            ]
        )
        provided_claims_text = str(original_claims.get("text", "")).strip() if isinstance(original_claims, dict) else ""
        if (not original_claims) or (not provided_claims_text):
            original_claims = {
                "source": application_name,
                "text": claims_text,
                "extracted": claims_extracted,
                "strategy": claims_strategy,
            }
        prior_arts_paths = resolved_prior_paths
        request.original_claims = original_claims
    if not oa_text.strip():
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="Either oa_text or file ids are required.",
            session_id=session_id,
            details={},
            retryable=False,
        )
    _structured_log(
        "oa_start_received",
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        oa_text=_redact_text(oa_text),
    )

    state: WorkflowOAState = {
        "session_id": session_id,
        "trace_id": trace_id,
        "status": "running",
        "current_step": "application_baseline_node",
        "oa_text": oa_text,
        "original_claims": original_claims,
        "application_specification": application_specification,
        "prior_arts_paths": prior_arts_paths,
        "parsed_defects": None,
        "retrieved_contexts": [],
        "prior_art_targeted_report": None,
        "image_recognition_report": None,
        "targeted_reading_audit": None,
        "application_baseline": None,
        "concession_gap_report": None,
        "mined_fallback_features": None,
        "stress_test_report": None,
        "strategy_decision": None,
        "rebuttal_plan": None,
        "amended_claims": None,
        "argument_draft": None,
        "spec_update_note": None,
        "response_traceability": None,
        "comparison_result": None,
        "visual_report": None,
        "application_images": application_images,
        "prior_art_images": prior_art_images,
        "vision_warnings": [],
        "max_reflections": int(llm_runtime.get("max_reflections") or 3),
        "final_strategy": None,
        "final_reply_text": None,
        "error_count": 0,
        "tool_error_count": 0,
        "last_error": None,
        "node_latency_ms": 0,
    }
    try:
        output = _CHECKPOINT_MANAGER.invoke(graph=oa_graph, state=cast(dict[str, Any], state), thread_id=session_id)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(
            http_status=500,
            code="E500_TOOL_FAILURE",
            message=str(exc),
            session_id=session_id,
            retryable=True,
            details={"step": "oa_start"},
        ) from exc

    status = _infer_status(output)
    data = {
        "current_step": output.get("current_step", "response_traceability_node"),
        "original_claims": original_claims,
        "original_claims_extracted": original_claims_extracted,
        "original_claims_strategy": original_claims_strategy,
        "application_specification": application_specification,
        "application_specification_extracted": application_specification_extracted,
        "application_specification_strategy": application_specification_strategy,
        "parsed_defects": output.get("parsed_defects"),
        "retrieved_contexts": output.get("retrieved_contexts", []),
        "prior_art_targeted_report": output.get("prior_art_targeted_report"),
        "image_recognition_report": output.get("image_recognition_report"),
        "targeted_reading_audit": output.get("targeted_reading_audit"),
        "application_baseline": output.get("application_baseline"),
        "concession_gap_report": output.get("concession_gap_report"),
        "mined_fallback_features": output.get("mined_fallback_features"),
        "stress_test_report": output.get("stress_test_report"),
        "strategy_decision": output.get("strategy_decision"),
        "rebuttal_plan": output.get("rebuttal_plan"),
        "amended_claims": output.get("amended_claims"),
        "argument_draft": output.get("argument_draft"),
        "spec_update_note": output.get("spec_update_note"),
        "response_traceability": output.get("response_traceability"),
        "comparison_result": output.get("comparison_result"),
        "visual_report": output.get("visual_report"),
        "application_images_meta": application_images,
        "prior_art_images_meta": prior_art_images,
        "vision_warnings": output.get("vision_warnings", []),
        "final_strategy": output.get("final_strategy"),
        "final_reply_text": output.get("final_reply_text"),
        "oa_source_text": oa_source_text,
        "oa_notice_focus_text": focused_notice_text if request.oa_notice_file_id else None,
        "oa_notice_focus_applied": used_notice_extraction if request.oa_notice_file_id else None,
        "metadata": request.metadata,
        "llm_runtime": llm_runtime,
        "llm_mode": "real" if is_real_llm else "stub",
        "vision_mode": "real" if is_real_llm else "stub",
    }
    session = SessionRecord(
        session_id=session_id,
        request_id=request_id,
        status=status,
        data=data,
    )
    store.upsert(session)
    _append_event(
        session_id=session_id,
        event_type="image_extracted",
        payload={
            "workflow": "oa",
            "application_image_count": len(application_images),
            "prior_art_image_count": len(prior_art_images),
        },
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="visual_compare_ready",
        payload={"workflow": "oa", "has_visual_report": bool(output.get("visual_report"))},
        store=store,
    )
    if output.get("vision_warnings"):
        _append_event(
            session_id=session_id,
            event_type="vision_fallback",
            payload={"workflow": "oa", "warnings": output.get("vision_warnings", [])},
            store=store,
        )
    _append_event(
        session_id=session_id,
        event_type="session_completed" if status == SessionStatus.completed else "session_failed",
        payload={"status": status.value, "workflow": "oa"},
        store=store,
    )
    return ApiEnvelope(
        request_id=request_id,
        session_id=session_id,
        status=status,
        data=data,
        error=None,
    )


@router.post("/files/preview", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def preview_files(
    request: FilePreviewRequest,
    file_store: InMemoryFileStore = Depends(get_file_store),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
    x_llm_vision_model: str | None = Header(default=None, alias="X-LLM-Vision-Model"),
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_temperature: str | None = Header(default=None, alias="X-LLM-Temperature"),
    x_agent_max_reflections: str | None = Header(default=None, alias="X-Agent-Max-Reflections"),
    x_agent_context_window_limit: str | None = Header(default=None, alias="X-Agent-Context-Window-Limit"),
    x_legal_jurisdiction_baseline: str | None = Header(default=None, alias="X-Legal-Jurisdiction-Baseline"),
    x_legal_claim_formatting: str | None = Header(default=None, alias="X-Legal-Claim-Formatting"),
) -> ApiEnvelope:
    request_id = str(uuid4())
    session_id = str(uuid4())
    llm_runtime = _read_llm_runtime_from_headers(
        x_llm_provider=x_llm_provider,
        x_llm_model=x_llm_model,
        x_llm_vision_model=x_llm_vision_model,
        x_llm_base_url=x_llm_base_url,
        x_llm_api_key=x_llm_api_key,
        x_llm_temperature=x_llm_temperature,
        x_agent_max_reflections=x_agent_max_reflections,
        x_agent_context_window_limit=x_agent_context_window_limit,
        x_legal_jurisdiction_baseline=x_legal_jurisdiction_baseline,
        x_legal_claim_formatting=x_legal_claim_formatting,
    )
    parser = _build_doc_parser_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)

    def _clip(text: str, limit: int = 12000) -> str:
        text = (text or "").strip()
        return text if len(text) <= limit else text[:limit]

    data: dict[str, Any] = {"workflow": request.workflow}

    if request.workflow == "draft":
        if not request.disclosure_file_id:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="disclosure_file_id is required for draft preview.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        disclosure_text, _, _ = _parse_uploaded_file(request.disclosure_file_id, file_store, parser=parser)
        data["disclosure_preview_text"] = _clip(disclosure_text)

    elif request.workflow == "oa":
        if not request.oa_notice_file_id or not request.application_file_id:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="oa_notice_file_id and application_file_id are required for oa preview.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        if len(request.prior_art_file_ids) < 1:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="At least one prior_art_file_id is required for oa preview.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        notice_text, _, _ = _parse_uploaded_file(request.oa_notice_file_id, file_store, parser=parser)
        notice_pages = _parse_uploaded_pdf_pages(request.oa_notice_file_id, file_store)
        focused_notice, _ = _extract_examiner_opinion_text(notice_text, notice_pages=notice_pages)
        application_text, _, _ = _parse_uploaded_file(request.application_file_id, file_store, parser=parser)
        claims_text, _, _ = _extract_original_claims_text(application_text)
        spec_text, _, _ = _extract_application_specification_text(application_text)
        prior_art_previews: list[dict[str, str]] = []
        for file_id in request.prior_art_file_ids:
            prior_text, _, prior_name = _parse_uploaded_file(file_id, file_store, parser=parser)
            prior_art_previews.append({"file_id": file_id, "filename": prior_name, "text": _clip(prior_text, 6000)})
        data.update(
            {
                "oa_notice_focus_text": _clip(focused_notice),
                "original_claims_text": _clip(claims_text),
                "application_specification_text": _clip(spec_text),
                "prior_art_previews": prior_art_previews,
            }
        )

    elif request.workflow == "compare":
        if not request.application_file_id:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="application_file_id is required for compare preview.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        if len(request.prior_art_file_ids) < 1:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="At least one prior_art_file_id is required for compare preview.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        application_text, _, _ = _parse_uploaded_file(request.application_file_id, file_store, parser=parser)
        claims_text, _, _ = _extract_original_claims_text(application_text)
        spec_text, _, _ = _extract_application_specification_text(application_text)
        prior_art_previews = []
        for file_id in request.prior_art_file_ids:
            prior_text, _, prior_name = _parse_uploaded_file(file_id, file_store, parser=parser)
            prior_art_previews.append({"file_id": file_id, "filename": prior_name, "text": _clip(prior_text, 6000)})
        data.update(
            {
                "original_claims_text": _clip(claims_text),
                "application_specification_text": _clip(spec_text),
                "prior_art_previews": prior_art_previews,
            }
        )

    else:  # polish
        if not request.application_file_id:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="application_file_id is required for polish preview.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        application_text, _, _ = _parse_uploaded_file(request.application_file_id, file_store, parser=parser)
        claims_text, _, _ = _extract_original_claims_text(application_text)
        spec_text, _, _ = _extract_application_specification_text(application_text)
        data.update(
            {
                "original_claims_text": _clip(claims_text),
                "application_specification_text": _clip(spec_text),
            }
        )

    return ApiEnvelope(
        request_id=request_id,
        session_id=session_id,
        status=SessionStatus.completed,
        data=data,
        error=None,
    )


@router.get("/images/{image_id}")
def get_runtime_image(image_id: str):
    resolved = _resolve_runtime_image_path(image_id)
    if resolved is None:
        raise ApiError(
            http_status=404,
            code="E404_IMAGE_NOT_FOUND",
            message="Image not found.",
            session_id="unknown",
            details={"image_id": image_id},
        )
    return FileResponse(path=str(resolved))


@router.post("/compare/start", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def start_compare(
    request: CompareStartRequest,
    store: InMemorySessionStore = Depends(get_session_store),
    file_store: InMemoryFileStore = Depends(get_file_store),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
    x_llm_vision_model: str | None = Header(default=None, alias="X-LLM-Vision-Model"),
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_temperature: str | None = Header(default=None, alias="X-LLM-Temperature"),
    x_agent_max_reflections: str | None = Header(default=None, alias="X-Agent-Max-Reflections"),
    x_agent_context_window_limit: str | None = Header(default=None, alias="X-Agent-Context-Window-Limit"),
    x_legal_jurisdiction_baseline: str | None = Header(default=None, alias="X-Legal-Jurisdiction-Baseline"),
    x_legal_claim_formatting: str | None = Header(default=None, alias="X-Legal-Claim-Formatting"),
) -> ApiEnvelope:
    session_id = str(uuid4())
    request_id = str(uuid4())
    trace_id = request_id
    llm_runtime = _read_llm_runtime_from_headers(
        x_llm_provider=x_llm_provider,
        x_llm_model=x_llm_model,
        x_llm_vision_model=x_llm_vision_model,
        x_llm_base_url=x_llm_base_url,
        x_llm_api_key=x_llm_api_key,
        x_llm_temperature=x_llm_temperature,
        x_agent_max_reflections=x_agent_max_reflections,
        x_agent_context_window_limit=x_agent_context_window_limit,
        x_legal_jurisdiction_baseline=x_legal_jurisdiction_baseline,
        x_legal_claim_formatting=x_legal_claim_formatting,
    )
    compare_graph, is_real_llm = _build_compare_graph_for_runtime(
        llm_runtime=llm_runtime, llm_api_key=x_llm_api_key
    )
    if _llm_requested(llm_runtime) and not is_real_llm:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="LLM is configured but API key is missing or invalid for this provider.",
            session_id=session_id,
            details={"llm_runtime": llm_runtime},
            retryable=False,
        )

    parser = _build_doc_parser_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)
    original_claims = dict(request.original_claims)
    application_specification = dict(request.application_specification)
    prior_arts_paths = list(request.prior_arts_paths)
    application_images: list[dict[str, Any]] = []
    prior_art_images: list[dict[str, Any]] = []
    compare_source_text = ""

    if request.application_file_id or request.prior_art_file_ids:
        if not request.application_file_id:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="application_file_id is required in file upload mode.",
                session_id=session_id,
                details={},
                retryable=False,
            )
        if len(request.prior_art_file_ids) < 1:
            raise ApiError(
                http_status=400,
                code="E400_INVALID_INPUT",
                message="At least one prior_art_file_id is required in file upload mode.",
                session_id=session_id,
                details={},
                retryable=False,
            )

        application_text, _, application_name = _parse_uploaded_file(
            request.application_file_id,
            file_store,
            parser=parser,
        )
        claims_text, claims_extracted, claims_strategy = _extract_original_claims_text(application_text)
        spec_text, spec_extracted, spec_strategy = _extract_application_specification_text(application_text)
        original_claims = {
            "source": application_name,
            "text": claims_text,
            "extracted": claims_extracted,
            "strategy": claims_strategy,
        }
        application_specification = {
            "source": application_name,
            "text": spec_text,
            "extracted": spec_extracted,
            "strategy": spec_strategy,
        }
        application_images = _extract_uploaded_images(request.application_file_id, file_store)

        prior_parts: list[str] = []
        resolved_prior_paths: list[str] = []
        for index, file_id in enumerate(request.prior_art_file_ids, start=1):
            prior_text, prior_path, prior_name = _parse_uploaded_file(file_id, file_store, parser=parser)
            prior_art_images.extend(_extract_uploaded_images(file_id, file_store))
            prior_parts.append(f"D{index} ({prior_name}):\n{prior_text}")
            resolved_prior_paths.append(prior_path)
        prior_arts_paths = resolved_prior_paths
        compare_source_text = "\n".join(
            [
                "[\u7533\u8bf7\u6587\u4ef6]",
                application_text,
                "",
                "[\u5bf9\u6bd4\u6587\u4ef6]",
                *prior_parts,
            ]
        )

    claims_text = str(original_claims.get("text", "")).strip()
    spec_text = str(application_specification.get("text", "")).strip()
    if not claims_text or not spec_text:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="Missing extractable claims/specification text for comparison.",
            session_id=session_id,
            details={},
            retryable=False,
        )

    state: WorkflowCompareState = {
        "session_id": session_id,
        "trace_id": trace_id,
        "status": "running",
        "current_step": "multimodal_draft_parser_node",
        "comparison_goal": request.comparison_goal,
        "original_claims": original_claims,
        "application_specification": application_specification,
        "prior_arts_paths": prior_arts_paths,
        "application_images": application_images,
        "prior_art_images": prior_art_images,
        "vision_warnings": [],
        "max_reflections": int(llm_runtime.get("max_reflections") or 3),
        "draft_baseline": None,
        "prior_art_profiles": None,
        "feature_collision_matrix": None,
        "collision_matrix": None,
        "risk_report": None,
        "risk_assessment_report": None,
        "amendment_suggestions": None,
        "final_compare_report": None,
        "prior_art_targeted_report": None,
        "targeted_reading_audit": None,
        "retrieved_contexts": [],
        "error_count": 0,
        "tool_error_count": 0,
        "last_error": None,
        "node_latency_ms": 0,
    }
    try:
        output = _CHECKPOINT_MANAGER.invoke(graph=compare_graph, state=cast(dict[str, Any], state), thread_id=session_id)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(
            http_status=500,
            code="E500_TOOL_FAILURE",
            message=str(exc),
            session_id=session_id,
            retryable=True,
            details={"step": "compare_start"},
        ) from exc

    status = _infer_status(output)
    data = {
        "current_step": output.get("current_step", "amendment_suggestion_node"),
        "comparison_goal": request.comparison_goal,
        "original_claims": original_claims,
        "application_specification": application_specification,
        "draft_baseline": output.get("draft_baseline"),
        "prior_art_profiles": output.get("prior_art_profiles"),
        "targeted_reading_audit": output.get("targeted_reading_audit"),
        "retrieved_contexts": output.get("retrieved_contexts", []),
        "feature_collision_matrix": output.get("feature_collision_matrix"),
        "collision_matrix": output.get("collision_matrix"),
        "prior_art_targeted_report": output.get("prior_art_targeted_report"),
        "risk_report": output.get("risk_report"),
        "risk_assessment_report": output.get("risk_assessment_report"),
        "amendment_suggestions": output.get("amendment_suggestions"),
        "final_compare_report": output.get("final_compare_report"),
        "application_images_meta": application_images,
        "prior_art_images_meta": prior_art_images,
        "vision_warnings": output.get("vision_warnings", []),
        "compare_source_text": compare_source_text,
        "metadata": request.metadata,
        "llm_runtime": llm_runtime,
        "llm_mode": "real" if is_real_llm else "stub",
        "vision_mode": "real" if is_real_llm else "stub",
    }
    session = SessionRecord(
        session_id=session_id,
        request_id=request_id,
        status=status,
        data=data,
    )
    store.upsert(session)
    _append_event(
        session_id=session_id,
        event_type="compare_started",
        payload={"workflow": "compare"},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="image_extracted",
        payload={
            "workflow": "compare",
            "application_image_count": len(application_images),
            "prior_art_image_count": len(prior_art_images),
        },
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="matrix_ready",
        payload={"workflow": "compare", "has_matrix": bool(output.get("feature_collision_matrix"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="risk_ready",
        payload={"workflow": "compare", "has_risk_report": bool(output.get("risk_report"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="amendment_ready",
        payload={"workflow": "compare", "has_suggestions": bool(output.get("amendment_suggestions"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="session_completed" if status == SessionStatus.completed else "session_failed",
        payload={"status": status.value, "workflow": "compare"},
        store=store,
    )
    return ApiEnvelope(
        request_id=request_id,
        session_id=session_id,
        status=status,
        data=data,
        error=None,
    )


@router.post("/polish/start", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def start_polish(
    request: PolishStartRequest,
    store: InMemorySessionStore = Depends(get_session_store),
    file_store: InMemoryFileStore = Depends(get_file_store),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    x_llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
    x_llm_vision_model: str | None = Header(default=None, alias="X-LLM-Vision-Model"),
    x_llm_base_url: str | None = Header(default=None, alias="X-LLM-Base-URL"),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
    x_llm_temperature: str | None = Header(default=None, alias="X-LLM-Temperature"),
    x_agent_max_reflections: str | None = Header(default=None, alias="X-Agent-Max-Reflections"),
    x_agent_context_window_limit: str | None = Header(default=None, alias="X-Agent-Context-Window-Limit"),
    x_legal_jurisdiction_baseline: str | None = Header(default=None, alias="X-Legal-Jurisdiction-Baseline"),
    x_legal_claim_formatting: str | None = Header(default=None, alias="X-Legal-Claim-Formatting"),
) -> ApiEnvelope:
    session_id = str(uuid4())
    request_id = str(uuid4())
    trace_id = request_id
    llm_runtime = _read_llm_runtime_from_headers(
        x_llm_provider=x_llm_provider,
        x_llm_model=x_llm_model,
        x_llm_vision_model=x_llm_vision_model,
        x_llm_base_url=x_llm_base_url,
        x_llm_api_key=x_llm_api_key,
        x_llm_temperature=x_llm_temperature,
        x_agent_max_reflections=x_agent_max_reflections,
        x_agent_context_window_limit=x_agent_context_window_limit,
        x_legal_jurisdiction_baseline=x_legal_jurisdiction_baseline,
        x_legal_claim_formatting=x_legal_claim_formatting,
    )
    polish_graph, is_real_llm = _build_polish_graph_for_runtime(
        llm_runtime=llm_runtime, llm_api_key=x_llm_api_key
    )
    if _llm_requested(llm_runtime) and not is_real_llm:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="LLM is configured but API key is missing or invalid for this provider.",
            session_id=session_id,
            details={"llm_runtime": llm_runtime},
            retryable=False,
        )

    parser = _build_doc_parser_for_runtime(llm_runtime=llm_runtime, llm_api_key=x_llm_api_key)
    original_claims = dict(request.original_claims or {})
    application_specification = dict(request.application_specification or {})
    polish_source_text = ""
    application_images: list[dict[str, Any]] = []

    if request.application_file_id:
        application_text, _, application_name = _parse_uploaded_file(
            request.application_file_id,
            file_store,
            parser=parser,
        )
        application_images = _extract_uploaded_images(request.application_file_id, file_store)
        claims_text, claims_extracted, claims_strategy = _extract_original_claims_text(application_text)
        spec_text, spec_extracted, spec_strategy = _extract_application_specification_text(application_text)
        original_claims = {
            "source": application_name,
            "text": claims_text,
            "extracted": claims_extracted,
            "strategy": claims_strategy,
        }
        application_specification = {
            "source": application_name,
            "text": spec_text,
            "extracted": spec_extracted,
            "strategy": spec_strategy,
        }
        polish_source_text = application_text

    claims_text = str(original_claims.get("text", "")).strip()
    spec_text = str(application_specification.get("text", "")).strip()
    if not claims_text or not spec_text:
        raise ApiError(
            http_status=400,
            code="E400_INVALID_INPUT",
            message="Missing extractable claims/specification text for polish.",
            session_id=session_id,
            details={},
            retryable=False,
        )

    state: WorkflowPolishState = {
        "session_id": session_id,
        "trace_id": trace_id,
        "status": "running",
        "current_step": "multimodal_diagnostic_analyzer_node",
        "original_claims": original_claims,
        "application_specification": application_specification,
        "application_images": application_images,
        "diagnostic_report": None,
        "synergy_feature_vault": None,
        "claim_architecture_plan": None,
        "optimized_claims_text": "",
        "amplified_specification": None,
        "optimized_specification_text": "",
        "adversarial_review_report": None,
        "polish_final_package": None,
        "polish_revision_count": 0,
        "error_count": 0,
        "tool_error_count": 0,
        "last_error": None,
        "node_latency_ms": 0,
    }
    try:
        output = _CHECKPOINT_MANAGER.invoke(graph=polish_graph, state=cast(dict[str, Any], state), thread_id=session_id)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(
            http_status=500,
            code="E500_TOOL_FAILURE",
            message=str(exc),
            session_id=session_id,
            retryable=True,
            details={"step": "polish_start"},
        ) from exc

    status = _infer_status(output)
    data = {
        "current_step": output.get("current_step", "multimodal_adversarial_reviewer_node"),
        "original_claims": original_claims,
        "application_specification": application_specification,
        "application_images_meta": application_images,
        "diagnostic_report": output.get("diagnostic_report"),
        "synergy_feature_vault": output.get("synergy_feature_vault"),
        "claim_architecture_plan": output.get("claim_architecture_plan"),
        "optimized_claims_text": output.get("optimized_claims_text"),
        "amplified_specification": output.get("amplified_specification"),
        "optimized_specification_text": output.get("optimized_specification_text"),
        "adversarial_review_report": output.get("adversarial_review_report"),
        "polish_final_package": output.get("polish_final_package"),
        "polish_revision_count": output.get("polish_revision_count", 0),
        "polish_source_text": polish_source_text,
        "metadata": request.metadata,
        "llm_runtime": llm_runtime,
        "llm_mode": "real" if is_real_llm else "stub",
        "vision_mode": "real" if is_real_llm else "stub",
    }
    session = SessionRecord(
        session_id=session_id,
        request_id=request_id,
        status=status,
        data=data,
    )
    store.upsert(session)
    _append_event(
        session_id=session_id,
        event_type="polish_started",
        payload={"workflow": "polish"},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="diagnosis_ready",
        payload={"workflow": "polish", "has_diagnosis": bool(output.get("diagnostic_report"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="vault_ready",
        payload={"workflow": "polish", "has_vault": bool(output.get("synergy_feature_vault"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="architecture_ready",
        payload={"workflow": "polish", "has_claim_architecture": bool(output.get("claim_architecture_plan"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="spec_amplified",
        payload={"workflow": "polish", "has_spec": bool(output.get("amplified_specification"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="adversarial_review_ready",
        payload={"workflow": "polish", "has_review": bool(output.get("adversarial_review_report"))},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="polish_completed",
        payload={"workflow": "polish", "status": status.value},
        store=store,
    )
    _append_event(
        session_id=session_id,
        event_type="session_completed" if status == SessionStatus.completed else "session_failed",
        payload={"status": status.value, "workflow": "polish"},
        store=store,
    )
    return ApiEnvelope(
        request_id=request_id,
        session_id=session_id,
        status=status,
        data=data,
        error=None,
    )


@router.get("/sessions/{session_id}", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def get_session_status(
    session_id: str,
    store: InMemorySessionStore = Depends(get_session_store),
) -> ApiEnvelope:
    session = store.get(session_id)
    if session is None:
        raise ApiError(
            http_status=404,
            code="E404_SESSION_NOT_FOUND",
            message="Session not found.",
            session_id=session_id,
            details={"session_id": session_id},
        )
    return ApiEnvelope(
        request_id=session.request_id,
        session_id=session.session_id,
        status=session.status,
        data=session.data,
        error=None,
    )


@router.get("/sessions/{session_id}/export-word", dependencies=[Depends(require_api_key)])
def export_session_word(
    session_id: str,
    mode: Literal["draft", "oa", "compare", "polish"] = Query(...),
    store: InMemorySessionStore = Depends(get_session_store),
) -> StreamingResponse:
    session = store.get(session_id)
    if session is None:
        raise ApiError(
            http_status=404,
            code="E404_SESSION_NOT_FOUND",
            message="Session not found.",
            session_id=session_id,
            details={"session_id": session_id},
        )

    payload = session.data if isinstance(session.data, dict) else {}
    content = build_export_docx(mode=mode, session_id=session_id, data=payload)
    filename = f"{mode}_{session_id}.docx"
    return StreamingResponse(
        iter([content]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@router.post("/sessions/{session_id}/cancel", response_model=ApiEnvelope, dependencies=[Depends(require_api_key)])
def cancel_session(
    session_id: str,
    store: InMemorySessionStore = Depends(get_session_store),
) -> ApiEnvelope:
    session = store.get(session_id)
    if session is None:
        raise ApiError(
            http_status=404,
            code="E404_SESSION_NOT_FOUND",
            message="Session not found.",
            session_id=session_id,
            details={"session_id": session_id},
        )
    try:
        cancel_draft_workflow(_DEFAULT_DRAFT_GRAPH, thread_id=session_id)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(
            http_status=500,
            code="E500_TOOL_FAILURE",
            message=str(exc),
            session_id=session_id,
            retryable=True,
            details={"step": "cancel"},
        ) from exc

    session = store.cancel(session_id)
    if session is None:
        raise ApiError(
            http_status=404,
            code="E404_SESSION_NOT_FOUND",
            message="Session not found.",
            session_id=session_id,
            details={"session_id": session_id},
        )

    _structured_log(
        "session_cancelled",
        request_id=session.request_id,
        session_id=session.session_id,
        trace_id=session.request_id,
    )
    return ApiEnvelope(
        request_id=session.request_id,
        session_id=session.session_id,
        status=SessionStatus.cancelled,
        data=session.data,
        error=None,
    )


@router.get("/sessions/{session_id}/events", dependencies=[Depends(require_api_key)])
async def stream_session_events(
    session_id: str,
    after_index: int = Query(default=0, ge=0),
    store: InMemorySessionStore = Depends(get_session_store),
) -> StreamingResponse:
    if store.get(session_id) is None:
        raise ApiError(
            http_status=404,
            code="E404_SESSION_NOT_FOUND",
            message="Session not found.",
            session_id=session_id,
            details={"session_id": session_id},
        )

    async def _event_generator():
        index = after_index
        # Keep polling for up to 5 minutes per connection; client can reconnect with after_index.
        for _ in range(300):
            events = store.get_events(session_id, index)
            if events is None:
                break
            if events:
                for offset, event in enumerate(events):
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"event: {event.get('type', 'message')}\ndata: {payload}\n\n"
                    index += 1
                    # Emit deterministic index marker to help client resume stream.
                    marker = json.dumps({"index": index, "session_id": session_id}, ensure_ascii=False)
                    yield f"event: stream_offset\ndata: {marker}\n\n"
            else:
                heartbeat = json.dumps({"timestamp": _utc_now(), "session_id": session_id}, ensure_ascii=False)
                yield f"event: heartbeat\ndata: {heartbeat}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
