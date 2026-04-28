import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { HitlActionPanel } from "@/components/hitl/HitlActionPanel";
import HomeView from "@/components/home/HomeView";
import SettingsView from "@/components/settings/SettingsView";
import { Button } from "@/components/ui/button";
import { BottomHorizontalTimeline } from "@/components/workflow/BottomHorizontalTimeline";
import { NodeStageRenderer } from "@/components/workflow/NodeStageRenderer";
import {
  WORKFLOW_OUTPUTS,
  WORKFLOW_STEPS,
  getWorkflowOutputValue,
  type GlobalTab,
} from "@/components/workflow/workflowConfig";
import { buildTimelineNodes, getAutoSelectedNodeId } from "@/lib/timelineState";
import { apiClient, ApiClientError } from "@/services/apiClient";
import { mapDocumentEnvelope, mapEventPayload, mapSessionEnvelope } from "@/services/mappers";
import { connectSessionEvents, type SessionSSEConnection } from "@/services/sseClient";
import { useDocumentStore } from "@/stores/documentStore";
import { useLlmSettingsStore } from "@/stores/llmSettingsStore";
import { useLogStore } from "@/stores/logStore";
import { useSessionStore, type SessionStatus } from "@/stores/sessionStore";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const WORKFLOW_SESSIONS_KEY = "mpa.workflowSessions";
const LEGACY_SESSION_CACHE_KEY = "mpa.activeSession";
const LEGACY_STREAM_OFFSET_KEY = "mpa.streamOffset";
const MODE_KEY = "mpa.mode";
const POLL_INTERVAL_MS = 2500;
const SUPPORTED_UPLOAD_EXTS = new Set([".pdf", ".doc", ".docx", ".txt"]);

type WorkMode = "draft" | "oa" | "compare" | "polish" | null;
type WorkflowMode = Exclude<WorkMode, null>;
type WorkflowSessionCache = Record<WorkflowMode, { sessionId: string; streamOffset: number } | null>;

type PriorPreview = { file_id: string; filename: string; text: string };
type DraftPreviewState = { fileId: string; filename: string; disclosureText: string } | null;
type OaPreviewState = {
  noticeFileId: string;
  applicationFileId: string;
  priorArtFileIds: string[];
  oaNoticeFocusText: string;
  originalClaimsText: string;
  applicationSpecificationText: string;
  priorArtPreviews: PriorPreview[];
} | null;
type ComparePreviewState = {
  applicationFileId: string;
  priorArtFileIds: string[];
  originalClaimsText: string;
  applicationSpecificationText: string;
  priorArtPreviews: PriorPreview[];
} | null;
type PolishPreviewState = {
  applicationFileId: string;
  originalClaimsText: string;
  applicationSpecificationText: string;
} | null;

function ext(name: string) {
  const i = name.lastIndexOf(".");
  return i < 0 ? "" : name.slice(i).toLowerCase();
}

function isSupportedUpload(file: File) {
  return SUPPORTED_UPLOAD_EXTS.has(ext(file.name));
}

function isTerminal(status: SessionStatus) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function isWorkflowMode(value: unknown): value is WorkflowMode {
  return value === "draft" || value === "oa" || value === "compare" || value === "polish";
}

function readWorkflowSessionCache(): WorkflowSessionCache {
  const empty: WorkflowSessionCache = {
    draft: null,
    oa: null,
    compare: null,
    polish: null,
  };
  try {
    const raw = localStorage.getItem(WORKFLOW_SESSIONS_KEY);
    if (!raw) {
      const legacyMode = localStorage.getItem(MODE_KEY);
      const legacySessionId = localStorage.getItem(LEGACY_SESSION_CACHE_KEY);
      const legacyOffsetRaw = Number(localStorage.getItem(LEGACY_STREAM_OFFSET_KEY) ?? "0");
      if (isWorkflowMode(legacyMode) && legacySessionId) {
        empty[legacyMode] = {
          sessionId: legacySessionId,
          streamOffset: Number.isFinite(legacyOffsetRaw) ? legacyOffsetRaw : 0,
        };
      }
      return empty;
    }
    const parsed = JSON.parse(raw) as Partial<WorkflowSessionCache>;
    for (const key of ["draft", "oa", "compare", "polish"] as const) {
      const item = parsed[key];
      if (
        item &&
        typeof item === "object" &&
        typeof (item as { sessionId?: unknown }).sessionId === "string"
      ) {
        const offsetRaw = (item as { streamOffset?: unknown }).streamOffset;
        const streamOffset = typeof offsetRaw === "number" && Number.isFinite(offsetRaw) ? offsetRaw : 0;
        empty[key] = { sessionId: (item as { sessionId: string }).sessionId, streamOffset };
      }
    }
  } catch {
    return empty;
  }
  return empty;
}

function writeWorkflowSessionCache(cache: WorkflowSessionCache) {
  localStorage.setItem(WORKFLOW_SESSIONS_KEY, JSON.stringify(cache));
}

function sanitizeHeaderValue(value: string): string {
  // Browser Headers only accept ISO-8859-1 visible range.
  // Strip non-latin1 / control chars to avoid "Failed to construct 'Headers'" runtime errors.
  return value.replace(/[^\t\x20-\x7E\xA0-\xFF]/g, "").trim();
}

function App() {
  const { sessionId, status, currentStep, sessionData, setSession, resetSession } = useSessionStore();
  const { appendEvent, clearEvents } = useLogStore();
  const {
    claims,
    specification,
    applicationImagesMeta,
    priorArtImagesMeta,
    setDisclosureText,
    setDisclosureImagesMeta,
    setClaims,
    setDrawingMap,
    setClaimTraceability,
    setSpecification,
    setVisualReport,
    setApplicationImagesMeta,
    setPriorArtImagesMeta,
    setVisionWarnings,
    setOaReply,
    setOaSourceText,
    setOaNoticeFocusText,
    setOaNoticeFocusApplied,
    reset,
  } = useDocumentStore();
  const llm = useLlmSettingsStore();

  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [mode, setMode] = useState<WorkMode>(null);
  const [streamOffset, setStreamOffset] = useState(0);
  const [activeTab, setActiveTab] = useState<GlobalTab>("home");
  const [selectedNodeId, setSelectedNodeId] = useState<string>("upload");
  const [showStartHint, setShowStartHint] = useState(false);

  const [draftDisclosureFile, setDraftDisclosureFile] = useState<File | null>(null);
  const [draftPreview, setDraftPreview] = useState<DraftPreviewState>(null);
  const [oaNoticeFile, setOaNoticeFile] = useState<File | null>(null);
  const [applicationFile, setApplicationFile] = useState<File | null>(null);
  const [comparisonFiles, setComparisonFiles] = useState<Array<File | null>>([null]);
  const [oaPreview, setOaPreview] = useState<OaPreviewState>(null);
  const [compareApplicationFile, setCompareApplicationFile] = useState<File | null>(null);
  const [comparePriorFiles, setComparePriorFiles] = useState<Array<File | null>>([null]);
  const [comparePreview, setComparePreview] = useState<ComparePreviewState>(null);
  const [polishApplicationFile, setPolishApplicationFile] = useState<File | null>(null);
  const [polishPreview, setPolishPreview] = useState<PolishPreviewState>(null);
  const [isMaximized, setIsMaximized] = useState(false);

  const sseRef = useRef<SessionSSEConnection | null>(null);
  const pollerRef = useRef<number | null>(null);
  const tauriWindowRef = useRef<unknown>(null);
  const isDesktopRuntime =
    typeof window !== "undefined" &&
    (("__TAURI_INTERNALS__" in window) || ("__TAURI__" in window));
  const isMacRuntime =
    typeof navigator !== "undefined" && /Mac|iPhone|iPad/i.test(navigator.userAgent);

  const hasSession = useMemo(() => !!sessionId, [sessionId]);
  const hasActiveWorkflowSession = useMemo(
    () => hasSession && mode !== null && mode === activeTab,
    [hasSession, mode, activeTab],
  );
  const claimsJsonForHitl = useMemo(() => (claims ? JSON.stringify(claims, null, 2) : ""), [claims]);
  const hitlStage = useMemo(() => {
    if (currentStep === "claims_revise_review_node") return "claims_revise_review";
    if (currentStep === "spec_review_node") return "spec_review";
    const issues = sessionData?.review_issues;
    if (Array.isArray(issues) && issues.length > 0) return "spec_review";
    return "claims_review";
  }, [currentStep, sessionData]);

  const compareOriginalClaimsText = useMemo(() => {
    const raw = sessionData?.original_claims;
    if (raw && typeof raw === "object") {
      const t = (raw as Record<string, unknown>).text;
      if (typeof t === "string") return t;
      return JSON.stringify(raw, null, 2);
    }
    return "";
  }, [sessionData]);
  const compareSpecificationText = useMemo(() => {
    const raw = sessionData?.application_specification;
    if (raw && typeof raw === "object") {
      const t = (raw as Record<string, unknown>).text;
      if (typeof t === "string") return t;
      return JSON.stringify(raw, null, 2);
    }
    return "";
  }, [sessionData]);

  const timelineNodes = useMemo(
    () =>
      buildTimelineNodes({
        mode: activeTab,
        status,
        currentStep,
        data: sessionData,
        hasSession: hasActiveWorkflowSession,
      }),
    [activeTab, status, currentStep, sessionData, hasActiveWorkflowSession],
  );

  const selectedTimelineNode = useMemo(
    () => timelineNodes.find((n) => n.id === selectedNodeId) ?? timelineNodes[0] ?? null,
    [timelineNodes, selectedNodeId],
  );

  const selectedOutputConfig = useMemo(() => {
    if (activeTab === "home" || activeTab === "settings") return null;
    return WORKFLOW_OUTPUTS[activeTab].find((item) => item.nodeId === selectedNodeId) ?? null;
  }, [activeTab, selectedNodeId]);

  const selectedNodeValue = useMemo(() => {
    if (!selectedOutputConfig || !sessionData) return null;
    if (activeTab === "home" || activeTab === "settings") return null;
    const primary = getWorkflowOutputValue(activeTab, selectedOutputConfig.valueKey, sessionData);
    if (primary !== null && primary !== undefined) return primary;
    if (activeTab === "draft" && selectedNodeId === "write_spec_node") {
      return (
        sessionData.specification ??
        sessionData.approved_specification ??
        sessionData.application_specification ??
        sessionData.final_specification ??
        sessionData.specification_text ??
        specification ??
        null
      );
    }
    return primary;
  }, [activeTab, selectedNodeId, selectedOutputConfig, sessionData, specification]);

  const currentStepLabel = useMemo(() => {
    if (activeTab === "home" || activeTab === "settings") return "";
    return WORKFLOW_STEPS[activeTab].find((item) => item.id === currentStep)?.label ?? "";
  }, [activeTab, currentStep]);

  const headers = () => {
    const out: Record<string, string> = {};
    const apiKey = sanitizeHeaderValue(llm.apiKey.trim());
    // If no runtime API key is provided from frontend settings,
    // do not send any LLM runtime headers so backend can fully fallback to .env defaults.
    if (!apiKey) return out;

    const provider = sanitizeHeaderValue(llm.provider ?? "");
    const model = sanitizeHeaderValue(llm.model.trim());
    const visionModel = sanitizeHeaderValue(llm.visionModel.trim());
    const baseUrl = sanitizeHeaderValue(llm.baseUrl.trim());
    const temperature = Number.isFinite(llm.temperature) ? String(llm.temperature) : "";
    const maxReflections = Number.isFinite(llm.maxReflections) ? String(llm.maxReflections) : "";
    const contextWindowLimit = sanitizeHeaderValue((llm.contextWindowLimit ?? "").trim());
    const jurisdictionBaseline = sanitizeHeaderValue((llm.jurisdictionBaseline ?? "").trim());
    const claimFormatting = sanitizeHeaderValue((llm.claimFormatting ?? "").trim());
    if (provider) out["X-LLM-Provider"] = provider;
    if (model) out["X-LLM-Model"] = model;
    if (visionModel) out["X-LLM-Vision-Model"] = visionModel;
    if (baseUrl) out["X-LLM-Base-URL"] = baseUrl;
    if (apiKey) out["X-LLM-API-Key"] = apiKey;
    if (temperature) out["X-LLM-Temperature"] = temperature;
    if (maxReflections) out["X-Agent-Max-Reflections"] = maxReflections;
    if (contextWindowLimit) out["X-Agent-Context-Window-Limit"] = contextWindowLimit;
    if (jurisdictionBaseline) out["X-Legal-Jurisdiction-Baseline"] = jurisdictionBaseline;
    if (claimFormatting) out["X-Legal-Claim-Formatting"] = claimFormatting;
    return out;
  };
  const metadata = () => ({
    provider: llm.provider,
    model: llm.model.trim() || null,
    vision_model: llm.visionModel.trim() || null,
    base_url: llm.baseUrl.trim() || null,
    temperature: llm.temperature,
    max_reflections: llm.maxReflections,
    context_window_limit: llm.contextWindowLimit || null,
    jurisdiction_baseline: llm.jurisdictionBaseline || null,
    claim_formatting: llm.claimFormatting || null,
    api_key_configured: llm.apiKey.trim().length > 0,
  });

  const stopPolling = () => {
    if (pollerRef.current !== null) {
      window.clearInterval(pollerRef.current);
      pollerRef.current = null;
    }
  };
  const closeSse = () => {
    sseRef.current?.close();
    sseRef.current = null;
  };
  const cacheSessionForMode = (workflowMode: WorkflowMode, id: string, offset: number) => {
    const cache = readWorkflowSessionCache();
    cache[workflowMode] = { sessionId: id, streamOffset: Number.isFinite(offset) ? offset : 0 };
    writeWorkflowSessionCache(cache);
  };
  const getCachedSessionForMode = (workflowMode: WorkflowMode) => readWorkflowSessionCache()[workflowMode];
  const clearCachedSessionForMode = (workflowMode: WorkflowMode) => {
    const cache = readWorkflowSessionCache();
    cache[workflowMode] = null;
    writeWorkflowSessionCache(cache);
  };
  const resetForNewUpload = () => {
    stopPolling();
    closeSse();
    if (isWorkflowMode(activeTab)) clearCachedSessionForMode(activeTab);
    resetSession();
    reset();
    clearEvents();
    setMode(null);
    setStreamOffset(0);
    setSelectedNodeId("upload");
    setShowStartHint(false);
  };
  const cacheSession = (id: string, m: WorkMode, offset: number) => {
    if (!m) return;
    cacheSessionForMode(m, id, offset);
    localStorage.setItem(MODE_KEY, m);
  };

  const applyEnvelope = (envelope: Awaited<ReturnType<typeof apiClient.getSession>>) => {
    const s = mapSessionEnvelope(envelope);
    const d = mapDocumentEnvelope(envelope);
    setSession({
      requestId: s.requestId,
      sessionId: s.sessionId,
      status: s.status,
      currentStep: s.currentStep,
      llmMode: s.llmMode,
      llmRuntime: s.llmRuntime,
      visionMode: s.visionMode,
      sessionData: s.sessionData,
    });

    if (d.claims) setClaims(d.claims);
    if (d.disclosureText) setDisclosureText(d.disclosureText);
    setDisclosureImagesMeta(d.disclosureImagesMeta);
    if (d.claimTraceability) setClaimTraceability(d.claimTraceability);
    setDrawingMap(d.drawingMap);
    if (d.specification) setSpecification(d.specification);
    setVisualReport(d.visualReport);
    setApplicationImagesMeta(d.applicationImagesMeta);
    setPriorArtImagesMeta(d.priorArtImagesMeta);
    setVisionWarnings(d.visionWarnings);
    if (d.oaReply) setOaReply(d.oaReply);
    setOaSourceText(d.oaSourceText);
    setOaNoticeFocusText(d.oaNoticeFocusText);
    setOaNoticeFocusApplied(d.oaNoticeFocusApplied);

    if (isTerminal(s.status)) {
      stopPolling();
      closeSse();
    }
  };

  const withApiGuard = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
      setMessage("");
    } catch (error) {
      if (error instanceof ApiClientError) {
        setMessage(`${error.code}: ${error.message}`);
        appendEvent({
          timestamp: new Date().toISOString(),
          type: "api_error",
          payload: { code: error.code, message: error.message, status: error.httpStatus },
        });
      } else {
        setMessage(error instanceof Error ? error.message : "未知错误");
      }
      setShowStartHint(false);
    } finally {
      setBusy(false);
    }
  };

  const refreshSessionStatus = async (id: string) => {
    const env = await apiClient.getSession(id);
    applyEnvelope(env);
    appendEvent({ timestamp: new Date().toISOString(), type: "session_refreshed", payload: { session_id: id, status: env.status } });
    return env;
  };

  const startPolling = (id: string) => {
    stopPolling();
    pollerRef.current = window.setInterval(async () => {
      try {
        const env = await apiClient.getSession(id);
        applyEnvelope(env);
        if (isTerminal(env.status) || env.status === "waiting_human") stopPolling();
      } catch {
        stopPolling();
      }
    }, POLL_INTERVAL_MS);
  };

  const openSse = (id: string, offset: number) => {
    closeSse();
    sseRef.current = connectSessionEvents({
      baseUrl: API_BASE_URL,
      sessionId: id,
      afterIndex: offset,
      onOpen: () => {},
      onError: () => setMessage("SSE disconnected, you can reconnect manually."),
      onMessage: async (eventType, payload) => {
        appendEvent(mapEventPayload(eventType, payload));
        if (eventType === "stream_offset" && typeof payload.index === "number") {
          setStreamOffset(payload.index);
          cacheSession(id, mode, payload.index);
        }
        if (eventType === "hitl_required") {
          const p = payload.payload && typeof payload.payload === "object"
            ? (payload.payload as Record<string, unknown>)
            : (payload as unknown as Record<string, unknown>);
          const hasIssues = Array.isArray(p.review_issues) && (p.review_issues as unknown[]).length > 0;
          setSession({ status: "waiting_human", currentStep: typeof p.current_step === "string" ? p.current_step : hasIssues ? "spec_review_node" : "human_review_node" });
          stopPolling();
        }
        if (eventType === "session_completed" || eventType === "session_failed" || eventType === "session_cancelled") {
          await refreshSessionStatus(id);
        }
      },
    });
  };

  const startDraftWithDisclosure = async (fileId: string, fileName: string) => {
    await withApiGuard(async () => {
      clearEvents();
      reset();
      setMode("draft");
      setActiveTab("draft");
      setStreamOffset(0);
      const env = await apiClient.startDraft({
        idempotency_key: crypto.randomUUID(),
        disclosure_file_id: fileId,
        metadata: { source: "desktop-client", disclosure_file: fileName, upload_mode: "server_parse", llm: metadata() },
      }, headers());
      applyEnvelope(env);
      cacheSession(env.session_id, "draft", 0);
      openSse(env.session_id, 0);
      if (env.status === "running") startPolling(env.session_id);
    });
  };

  const startOaWithFiles = async (noticeId: string, appId: string, priorIds: string[]) => {
    await withApiGuard(async () => {
      clearEvents();
      reset();
      setMode("oa");
      setActiveTab("oa");
      setStreamOffset(0);
      const env = await apiClient.startOa({
        idempotency_key: crypto.randomUUID(),
        oa_notice_file_id: noticeId,
        application_file_id: appId,
        prior_art_file_ids: priorIds,
        original_claims: {},
        prior_arts_paths: [],
        metadata: { llm: metadata() },
      }, headers());
      applyEnvelope(env);
      cacheSession(env.session_id, "oa", 0);
      openSse(env.session_id, 0);
      if (env.status === "running") startPolling(env.session_id);
    });
  };

  const startCompareWithFiles = async (appId: string, priorIds: string[]) => {
    await withApiGuard(async () => {
      clearEvents();
      reset();
      setMode("compare");
      setActiveTab("compare");
      setStreamOffset(0);
      const env = await apiClient.startCompare({
        idempotency_key: crypto.randomUUID(),
        comparison_goal: "patentability",
        application_file_id: appId,
        prior_art_file_ids: priorIds,
        original_claims: {},
        application_specification: {},
        prior_arts_paths: [],
        metadata: { llm: metadata() },
      }, headers());
      applyEnvelope(env);
      cacheSession(env.session_id, "compare", 0);
      openSse(env.session_id, 0);
      if (env.status === "running") startPolling(env.session_id);
    });
  };

  const startPolishWithFile = async (appId: string) => {
    await withApiGuard(async () => {
      clearEvents();
      reset();
      setMode("polish");
      setActiveTab("polish");
      setStreamOffset(0);
      const env = await apiClient.startPolish({
        idempotency_key: crypto.randomUUID(),
        application_file_id: appId,
        original_claims: {},
        application_specification: {},
        metadata: { llm: metadata() },
      }, headers());
      applyEnvelope(env);
      cacheSession(env.session_id, "polish", 0);
      openSse(env.session_id, 0);
      if (env.status === "running") startPolling(env.session_id);
    });
  };


  const parseApprovedClaims = (text: string): Record<string, unknown> => {
    const fallback = claims;
    try {
      const payload = text.trim() ? (JSON.parse(text) as Record<string, unknown>) : (fallback as Record<string, unknown>);
      if (!payload || typeof payload !== "object") throw new Error("empty");
      return payload;
    } catch {
      throw new ApiClientError({ code: "E400_INVALID_INPUT", message: "approved_claims must be valid JSON ClaimsSet.", httpStatus: 400, retryable: false, sessionId: sessionId ?? "unknown" });
    }
  };

  const submitHitlClaims = async (text: string) => {
    if (!sessionId) return;
    await withApiGuard(async () => {
      const env = await apiClient.continueDraft({ session_id: sessionId, approved_claims: parseApprovedClaims(text) }, headers());
      applyEnvelope(env);
      openSse(sessionId, streamOffset);
      if (env.status === "running") startPolling(sessionId);
    });
  };
  const submitAutoReviseClaims = async () => {
    if (!sessionId) return;
    await withApiGuard(async () => {
      const env = await apiClient.continueDraft({ session_id: sessionId, apply_auto_claim_revision: true }, headers());
      applyEnvelope(env);
      openSse(sessionId, streamOffset);
      if (env.status === "running") startPolling(sessionId);
    });
  };
  const submitHitlSpecReview = async (revisionInstruction: string) => {
    if (!sessionId) return;
    await withApiGuard(async () => {
      const env = await apiClient.continueDraft({ session_id: sessionId, apply_targeted_revision: true, revision_instruction: revisionInstruction || undefined }, headers());
      applyEnvelope(env);
      openSse(sessionId, streamOffset);
      if (env.status === "running") startPolling(sessionId);
    });
  };

  const confirmDraftUpload = async () => {
    if (draftPreview) {
      setShowStartHint(true);
      await startDraftWithDisclosure(draftPreview.fileId, draftPreview.filename);
      return;
    }
    if (!draftDisclosureFile) return setMessage("请先上传技术交底书。");
    if (!isSupportedUpload(draftDisclosureFile)) return setMessage("技术交底书仅支持：pdf/doc/docx/txt。");
    await withApiGuard(async () => {
      const uploaded = await apiClient.uploadFile(draftDisclosureFile, "draft_disclosure");
      const preview = await apiClient.previewFiles(
        { workflow: "draft", disclosure_file_id: uploaded.file_id },
        headers(),
      );
      setDraftPreview({
        fileId: uploaded.file_id,
        filename: uploaded.filename,
        disclosureText: String(preview.disclosure_preview_text ?? ""),
      });
      setMessage("");
    });
  };

  const confirmOaUpload = async () => {
    if (oaPreview) {
      setShowStartHint(true);
      await startOaWithFiles(oaPreview.noticeFileId, oaPreview.applicationFileId, oaPreview.priorArtFileIds);
      return;
    }
    const files = comparisonFiles.filter((f): f is File => f !== null);
    if (!oaNoticeFile || !applicationFile || files.length < 1) return setMessage("请上传审查意见通知书、申请文件，以及至少 1 份对比文件。");
    const invalid = [oaNoticeFile, applicationFile, ...files].find((f) => !isSupportedUpload(f));
    if (invalid) return setMessage(`不支持的文件类型：${invalid.name}。仅支持：pdf/doc/docx/txt。`);
    await withApiGuard(async () => {
      const notice = await apiClient.uploadFile(oaNoticeFile, "oa_notice");
      const app = await apiClient.uploadFile(applicationFile, "application");
      const prior = await Promise.all(files.map((f) => apiClient.uploadFile(f, "prior_art")));
      const priorIds = prior.map((p) => p.file_id);
      const preview = await apiClient.previewFiles(
        {
          workflow: "oa",
          oa_notice_file_id: notice.file_id,
          application_file_id: app.file_id,
          prior_art_file_ids: priorIds,
        },
        headers(),
      );
      setOaPreview({
        noticeFileId: notice.file_id,
        applicationFileId: app.file_id,
        priorArtFileIds: priorIds,
        oaNoticeFocusText: String(preview.oa_notice_focus_text ?? ""),
        originalClaimsText: String(preview.original_claims_text ?? ""),
        applicationSpecificationText: String(preview.application_specification_text ?? ""),
        priorArtPreviews: Array.isArray(preview.prior_art_previews)
          ? (preview.prior_art_previews as PriorPreview[])
          : [],
      });
      setMessage("");
    });
  };

  const confirmCompareUpload = async () => {
    if (comparePreview) {
      setShowStartHint(true);
      await startCompareWithFiles(comparePreview.applicationFileId, comparePreview.priorArtFileIds);
      return;
    }
    const files = comparePriorFiles.filter((f): f is File => f !== null);
    if (!compareApplicationFile || files.length < 1) return setMessage("请上传申请文件，并至少上传 1 份对比文件。");
    const invalid = [compareApplicationFile, ...files].find((f) => !isSupportedUpload(f));
    if (invalid) return setMessage(`不支持的文件类型：${invalid.name}。仅支持：pdf/doc/docx/txt。`);
    await withApiGuard(async () => {
      const app = await apiClient.uploadFile(compareApplicationFile, "application");
      const prior = await Promise.all(files.map((f) => apiClient.uploadFile(f, "prior_art")));
      const priorIds = prior.map((p) => p.file_id);
      const preview = await apiClient.previewFiles(
        {
          workflow: "compare",
          application_file_id: app.file_id,
          prior_art_file_ids: priorIds,
        },
        headers(),
      );
      setComparePreview({
        applicationFileId: app.file_id,
        priorArtFileIds: priorIds,
        originalClaimsText: String(preview.original_claims_text ?? ""),
        applicationSpecificationText: String(preview.application_specification_text ?? ""),
        priorArtPreviews: Array.isArray(preview.prior_art_previews)
          ? (preview.prior_art_previews as PriorPreview[])
          : [],
      });
      setMessage("");
    });
  };

  const confirmPolishUpload = async () => {
    if (polishPreview) {
      setShowStartHint(true);
      await startPolishWithFile(polishPreview.applicationFileId);
      return;
    }
    if (!polishApplicationFile) return setMessage("请上传待润色的申请文件。");
    if (!isSupportedUpload(polishApplicationFile)) return setMessage("申请文件仅支持：pdf/doc/docx/txt。");
    await withApiGuard(async () => {
      const app = await apiClient.uploadFile(polishApplicationFile, "application");
      const preview = await apiClient.previewFiles(
        { workflow: "polish", application_file_id: app.file_id },
        headers(),
      );
      setPolishPreview({
        applicationFileId: app.file_id,
        originalClaimsText: String(preview.original_claims_text ?? ""),
        applicationSpecificationText: String(preview.application_specification_text ?? ""),
      });
      setMessage("");
    });
  };

  const setComparisonFile = (index: number, file: File | null) => {
    setOaPreview(null);
    setComparisonFiles((prev) => prev.map((item, idx) => (idx === index ? file : item)));
  };
  const addComparisonFileRow = () => {
    setOaPreview(null);
    setComparisonFiles((prev) => (prev.length >= 4 ? prev : [...prev, null]));
  };
  const removeComparisonFileRow = (index: number) => {
    setOaPreview(null);
    setComparisonFiles((prev) => (prev.length <= 1 ? prev : prev.filter((_, idx) => idx !== index)));
  };
  const setComparePriorFile = (index: number, file: File | null) => {
    setComparePreview(null);
    setComparePriorFiles((prev) => prev.map((item, idx) => (idx === index ? file : item)));
  };
  const addComparePriorFileRow = () => {
    setComparePreview(null);
    setComparePriorFiles((prev) => (prev.length >= 4 ? prev : [...prev, null]));
  };
  const removeComparePriorFileRow = (index: number) => {
    setComparePreview(null);
    setComparePriorFiles((prev) => (prev.length <= 1 ? prev : prev.filter((_, idx) => idx !== index)));
  };

  useEffect(() => {
    const cachedMode = localStorage.getItem(MODE_KEY);
    if (isWorkflowMode(cachedMode)) {
      setMode(cachedMode);
      setActiveTab(cachedMode);
    }
    return () => {
      stopPolling();
      closeSse();
    };
  }, []);

  useEffect(() => {
    if (sessionId) cacheSession(sessionId, mode, streamOffset);
  }, [mode, sessionId, streamOffset]);

  useEffect(() => {
    if (!showStartHint) return;
    if (status === "waiting_human" || status === "completed" || status === "failed" || status === "cancelled") {
      setShowStartHint(false);
    }
  }, [showStartHint, status]);

  useEffect(() => {
    if (!isWorkflowMode(activeTab)) return;
    const cached = getCachedSessionForMode(activeTab);
    if (!cached?.sessionId) return;

    let cancelled = false;
    void (async () => {
      try {
        stopPolling();
        closeSse();
        setMode(activeTab);
        setStreamOffset(cached.streamOffset);
        const env = await refreshSessionStatus(cached.sessionId);
        if (cancelled) return;
        const safeOffset = Number.isFinite(cached.streamOffset) ? cached.streamOffset : 0;
        openSse(cached.sessionId, safeOffset);
        if (env.status === "running") startPolling(cached.sessionId);
      } catch (error) {
        if (error instanceof ApiClientError && error.code === "E404_SESSION_NOT_FOUND") {
          clearCachedSessionForMode(activeTab);
          if (!cancelled) setMessage("");
          return;
        }
        if (error instanceof ApiClientError) {
          if (!cancelled) {
            setMessage(`${error.code}: ${error.message}`);
            appendEvent({
              timestamp: new Date().toISOString(),
              type: "api_error",
              payload: { code: error.code, message: error.message, status: error.httpStatus },
            });
          }
          return;
        }
        if (!cancelled) setMessage(error instanceof Error ? error.message : "未知错误");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeTab]);

  useEffect(() => {
    if (activeTab === "home" || activeTab === "settings") return;
    if (!hasActiveWorkflowSession) {
      setSelectedNodeId("upload");
      return;
    }
    if (status === "running" || status === "waiting_human" || status === "queued" || status === "failed" || status === "cancelled") {
      setSelectedNodeId(
        getAutoSelectedNodeId({
          mode: activeTab,
          status,
          currentStep,
          hasSession: hasActiveWorkflowSession,
        }),
      );
    }
  }, [activeTab, status, currentStep, hasActiveWorkflowSession]);

  const withCurrentWindow = useCallback(async () => {
    if (!isDesktopRuntime) return null;
    if (tauriWindowRef.current) {
      return tauriWindowRef.current as Awaited<ReturnType<typeof import("@tauri-apps/api/window").getCurrentWindow>>;
    }
    const mod = await import("@tauri-apps/api/window");
    const current = mod.getCurrentWindow();
    tauriWindowRef.current = current;
    return current;
  }, [isDesktopRuntime]);

  const windowActionInFlightRef = useRef(false);
  const runWindowAction = useCallback(
    async (action: (current: NonNullable<Awaited<ReturnType<typeof withCurrentWindow>>>) => Promise<void>) => {
      if (windowActionInFlightRef.current) return;
      windowActionInFlightRef.current = true;
      try {
        const current = await withCurrentWindow();
        if (!current) return;
        await action(current as NonNullable<Awaited<ReturnType<typeof withCurrentWindow>>>);
      } catch (error) {
        if (error === false || error == null) return;
        /* swallow window action errors */
      } finally {
        windowActionInFlightRef.current = false;
      }
    },
    [withCurrentWindow],
  );

  const minimizeWindow = useCallback(async () => {
    await runWindowAction(async (current) => {
      await current.minimize();
    });
  }, [runWindowAction]);

  const toggleMaximizeWindow = useCallback(async () => {
    await runWindowAction(async (current) => {
      if (isMacRuntime) {
        // macOS: avoid native fullscreen (which overlays a translucent titlebar
        // on top of our custom one). Manually zoom to the monitor work area.
        try {
          const winMod = await import("@tauri-apps/api/window");
          const { LogicalPosition, LogicalSize } = winMod;
          type MonitorLike = { size: { width: number; height: number }; position: { x: number; y: number }; scaleFactor: number };
          const currentMonitor = (winMod as unknown as { currentMonitor?: () => Promise<MonitorLike | null> }).currentMonitor;
          let monitor: MonitorLike | null = null;
          if (typeof currentMonitor === "function") {
            monitor = await currentMonitor();
          }
          const isMax = await current.isMaximized();
          if (isMax) {
            await current.unmaximize();
          } else if (monitor) {
            const scale = monitor.scaleFactor || 1;
            const w = monitor.size.width / scale;
            const h = monitor.size.height / scale;
            const x = monitor.position.x / scale;
            const y = monitor.position.y / scale;
            await current.setPosition(new LogicalPosition(x, y));
            await current.setSize(new LogicalSize(w, h));
            await current.maximize();
          } else {
            await current.maximize();
          }
        } catch {
          /* ignore */
        }
      } else {
        await current.toggleMaximize();
      }
      try {
        setIsMaximized(await current.isMaximized());
      } catch {
        /* ignore */
      }
    });
  }, [isMacRuntime, runWindowAction]);

  const closeWindow = useCallback(async () => {
    await runWindowAction(async (current) => {
      await current.close();
    });
  }, [runWindowAction]);

  const startWindowDrag = useCallback(
    async (event: MouseEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      await runWindowAction(async (current) => {
        await current.startDragging();
      });
    },
    [runWindowAction],
  );

  const showWorkflowMain = activeTab !== "home" && activeTab !== "settings";

  const renderUploadStage = () => {
    const showFlowHint = showStartHint && (busy || status === "queued" || status === "running");
    const flowHint = showFlowHint ? (
      <div className="mt-3 flex items-center gap-2 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-amber-300 border-t-amber-700" />
        <span>流程已启动，请稍等 5-10 分钟…</span>
      </div>
    ) : null;

    if (activeTab === "draft") {
      return (
        <section className="mx-auto w-full max-w-4xl space-y-10 py-6 text-center">
          <div className="border-b border-gray-100 pb-4">
            <h2 className="text-base font-semibold text-gray-800">上传技术交底书</h2>
            <p className="mt-1 text-sm text-gray-500">将技术交底书（Word/PDF/TXT）拖拽至此或点击上传。</p>
          </div>
          <input
            accept=".txt,.doc,.docx,.pdf"
            className="mx-auto block w-full max-w-xl border-b border-gray-300 px-1 py-3 text-sm outline-none file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-xs file:text-gray-700"
            type="file"
            onChange={(e) => {
              resetForNewUpload();
              setDraftPreview(null);
              setDraftDisclosureFile(e.target.files?.[0] ?? null);
            }}
          />
          <p className="text-xs text-gray-500">已选择：{draftDisclosureFile?.name ?? "未选择文件"}</p>
          {draftPreview ? (
            <section className="space-y-3 border-t border-gray-100 pt-6 text-left">
              <h3 className="text-sm font-semibold text-gray-800">技术交底书预览</h3>
              <textarea className="h-56 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={draftPreview.disclosureText || "未提取到可预览文本"} />
            </section>
          ) : null}
          <div className="pt-2">
            <Button disabled={busy || !draftDisclosureFile} onClick={() => void confirmDraftUpload()}>
              {draftPreview ? "确认预览并开始智能撰写" : "确认文件并生成预览"}
            </Button>
            {flowHint}
          </div>
          {message ? <p className="mt-3 text-xs text-red-600">{message}</p> : null}
        </section>
      );
    }

    if (activeTab === "oa") {
      return (
        <section className="mx-auto w-full max-w-4xl space-y-10 py-4">
          <div className="border-b border-gray-100 pb-4">
            <h2 className="text-base font-semibold text-gray-800">上传 OA 答复文件</h2>
            <p className="mt-1 text-sm text-gray-500">上传审查意见通知书、申请文件与对比文件。</p>
          </div>
          <div className="mt-4 grid gap-8 border-b border-gray-100 pb-6 md:grid-cols-2">
            <div>
              <label className="text-sm text-gray-700">审查意见通知书</label>
              <input
                accept=".txt,.doc,.docx,.pdf"
                className="mt-2 block w-full border-b border-gray-300 px-1 py-3 text-sm outline-none file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-xs file:text-gray-700"
                type="file"
                onChange={(e) => {
                  resetForNewUpload();
                  setOaPreview(null);
                  setOaNoticeFile(e.target.files?.[0] ?? null);
                }}
              />
            </div>
            <div>
              <label className="text-sm text-gray-700">申请文件</label>
              <input
                accept=".txt,.doc,.docx,.pdf"
                className="mt-2 block w-full border-b border-gray-300 px-1 py-3 text-sm outline-none file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-xs file:text-gray-700"
                type="file"
                onChange={(e) => {
                  resetForNewUpload();
                  setOaPreview(null);
                  setApplicationFile(e.target.files?.[0] ?? null);
                }}
              />
            </div>
          </div>
          <div className="mt-1 space-y-3 border-b border-gray-100 pb-6">
            <div className="flex justify-between">
              <span className="text-sm text-gray-700">对比文件</span>
              <Button disabled={busy || comparisonFiles.length >= 4} onClick={addComparisonFileRow} size="sm" variant="outline">新增一行</Button>
            </div>
            {comparisonFiles.map((file, idx) => (
              <div className="border-b border-gray-100 pb-4" key={`oa-pa-${idx}`}>
                <div className="flex justify-between">
                  <span className="text-xs text-gray-600">D{idx + 1}</span>
                  <Button disabled={busy || comparisonFiles.length <= 1} onClick={() => removeComparisonFileRow(idx)} size="sm" variant="outline">删除</Button>
                </div>
                <input accept=".txt,.doc,.docx,.pdf" className="mt-2 block w-full border-b border-gray-300 px-1 py-3 text-sm outline-none file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-xs file:text-gray-700" type="file" onChange={(e) => { resetForNewUpload(); setComparisonFile(idx, e.target.files?.[0] ?? null); }} />
                <p className="mt-1 text-xs text-gray-500">{file?.name ?? "未选择文件"}</p>
              </div>
            ))}
          </div>
          {oaPreview ? (
            <section className="mt-4 space-y-6 border-t border-gray-100 pt-6">
              <h3 className="text-sm font-semibold text-gray-800">预览：审查员具体意见</h3>
              <textarea className="h-36 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={oaPreview.oaNoticeFocusText || "未定位到审查员具体意见"} />
              <h3 className="text-sm font-semibold text-gray-800">预览：申请文件权利要求</h3>
              <textarea className="h-36 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={oaPreview.originalClaimsText || "未提取到权利要求"} />
              <h3 className="text-sm font-semibold text-gray-800">预览：申请文件说明书</h3>
              <textarea className="h-36 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={oaPreview.applicationSpecificationText || "未提取到说明书"} />
              <h3 className="text-sm font-semibold text-gray-800">预览：对比文件</h3>
              <div className="space-y-4">
                {oaPreview.priorArtPreviews.map((item) => (
                  <div className="border-b border-gray-100 pb-4" key={item.file_id}>
                    <p className="text-xs font-medium text-gray-700">{item.filename}</p>
                    <textarea className="mt-2 h-28 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={item.text} />
                  </div>
                ))}
              </div>
            </section>
          ) : null}
          <div className="mt-4 flex justify-end">
            <Button disabled={busy || !oaNoticeFile || !applicationFile || comparisonFiles.every((f) => f === null)} onClick={() => void confirmOaUpload()}>
              {oaPreview ? "确认预览并开始智能答复" : "确认文件并生成预览"}
            </Button>
          </div>
          {flowHint}
          {message ? <p className="mt-3 text-right text-xs text-red-600">{message}</p> : null}
        </section>
      );
    }

    if (activeTab === "compare") {
      return (
        <section className="mx-auto w-full max-w-4xl space-y-10 py-4">
          <div className="border-b border-gray-100 pb-4">
            <h2 className="text-base font-semibold text-gray-800">上传专利对比文件</h2>
            <p className="mt-1 text-sm text-gray-500">上传本案申请文件与现有专利（D1/D2...）。</p>
          </div>
          <div className="mt-4">
            <label className="text-sm text-gray-700">本案申请文件</label>
            <input
              accept=".txt,.doc,.docx,.pdf"
              className="mt-2 block w-full border-b border-gray-300 px-1 py-3 text-sm outline-none file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-xs file:text-gray-700"
              type="file"
              onChange={(e) => {
                resetForNewUpload();
                setComparePreview(null);
                setCompareApplicationFile(e.target.files?.[0] ?? null);
              }}
            />
          </div>
          <div className="mt-1 space-y-3 border-b border-gray-100 pb-6">
            <div className="flex justify-between">
              <span className="text-sm text-gray-700">对比文件</span>
              <Button disabled={busy || comparePriorFiles.length >= 4} onClick={addComparePriorFileRow} size="sm" variant="outline">新增一行</Button>
            </div>
            {comparePriorFiles.map((file, idx) => (
              <div className="border-b border-gray-100 pb-4" key={`cmp-pa-${idx}`}>
                <div className="flex justify-between">
                  <span className="text-xs text-gray-600">D{idx + 1}</span>
                  <Button disabled={busy || comparePriorFiles.length <= 1} onClick={() => removeComparePriorFileRow(idx)} size="sm" variant="outline">删除</Button>
                </div>
                <input accept=".txt,.doc,.docx,.pdf" className="mt-2 block w-full border-b border-gray-300 px-1 py-3 text-sm outline-none file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-xs file:text-gray-700" type="file" onChange={(e) => { resetForNewUpload(); setComparePriorFile(idx, e.target.files?.[0] ?? null); }} />
                <p className="mt-1 text-xs text-gray-500">{file?.name ?? "未选择文件"}</p>
              </div>
            ))}
          </div>
          {comparePreview ? (
            <section className="mt-4 space-y-6 border-t border-gray-100 pt-6">
              <h3 className="text-sm font-semibold text-gray-800">预览：本案权利要求</h3>
              <textarea className="h-36 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={comparePreview.originalClaimsText || "未提取到权利要求"} />
              <h3 className="text-sm font-semibold text-gray-800">预览：本案说明书</h3>
              <textarea className="h-36 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={comparePreview.applicationSpecificationText || "未提取到说明书"} />
              <h3 className="text-sm font-semibold text-gray-800">预览：对比文件</h3>
              <div className="space-y-4">
                {comparePreview.priorArtPreviews.map((item) => (
                  <div className="border-b border-gray-100 pb-4" key={item.file_id}>
                    <p className="text-xs font-medium text-gray-700">{item.filename}</p>
                    <textarea className="mt-2 h-28 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={item.text} />
                  </div>
                ))}
              </div>
            </section>
          ) : null}
          <div className="mt-4 flex justify-end">
            <Button disabled={busy || !compareApplicationFile || comparePriorFiles.every((f) => f === null)} onClick={() => void confirmCompareUpload()}>
              {comparePreview ? "确认预览并开始专利对比" : "确认文件并生成预览"}
            </Button>
          </div>
          {flowHint}
          {message ? <p className="mt-3 text-right text-xs text-red-600">{message}</p> : null}
        </section>
      );
    }

    return (
      <section className="mx-auto w-full max-w-4xl space-y-10 py-6 text-center">
        <div className="border-b border-gray-100 pb-4">
          <h2 className="text-base font-semibold text-gray-800">上传待润色申请文件</h2>
          <p className="mt-1 text-sm text-gray-500">上传现有申请文件（权利要求+说明书）。</p>
        </div>
        <input
          accept=".txt,.doc,.docx,.pdf"
          className="mx-auto block w-full max-w-xl border-b border-gray-300 px-1 py-3 text-sm outline-none file:mr-3 file:rounded file:border-0 file:bg-gray-100 file:px-3 file:py-1.5 file:text-xs file:text-gray-700"
          type="file"
          onChange={(e) => {
            resetForNewUpload();
            setPolishPreview(null);
            setPolishApplicationFile(e.target.files?.[0] ?? null);
          }}
        />
        <p className="mt-2 text-xs text-gray-500">已选择：{polishApplicationFile?.name ?? "未选择文件"}</p>
        {polishPreview ? (
          <section className="space-y-6 border-t border-gray-100 pt-6 text-left">
            <h3 className="text-sm font-semibold text-gray-800">预览：本案权利要求</h3>
            <textarea className="h-36 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={polishPreview.originalClaimsText || "未提取到权利要求"} />
            <h3 className="text-sm font-semibold text-gray-800">预览：本案说明书</h3>
            <textarea className="h-36 w-full border-y border-gray-200 bg-transparent p-3 text-xs leading-relaxed text-gray-700" readOnly value={polishPreview.applicationSpecificationText || "未提取到说明书"} />
          </section>
        ) : null}
        <div className="mt-5">
          <Button disabled={busy || !polishApplicationFile} onClick={() => void confirmPolishUpload()}>
            {polishPreview ? "确认预览并开始专利润色" : "确认文件并生成预览"}
          </Button>
          {flowHint}
        </div>
        {message ? <p className="mt-3 text-xs text-red-600">{message}</p> : null}
      </section>
    );
  };

  const renderWorkflowStage = () => {
    const node = selectedTimelineNode;
    if (!node) return null;

    if (node.id === "upload") return renderUploadStage();

    if (node.state === "running") {
      return (
        <section className="p-2">
          <h2 className="text-base font-semibold text-gray-800">{currentStepLabel || node.label}</h2>
          <p className="mt-1 text-sm text-gray-500">AI 正在处理当前节点，请稍候…</p>
          <div className="mt-4 space-y-2">
            <div className="h-4 w-11/12 animate-pulse rounded bg-gray-200" />
            <div className="h-4 w-10/12 animate-pulse rounded bg-gray-200" />
            <div className="h-4 w-8/12 animate-pulse rounded bg-gray-200" />
            <div className="h-36 w-full animate-pulse rounded bg-gray-100" />
          </div>
        </section>
      );
    }

    if (node.state === "waiting_human" && activeTab === "draft" && mode === "draft") {
      if (hitlStage === "spec_review") {
        return (
          <div className="space-y-3">
            <NodeStageRenderer
              busy={busy}
              mode="draft"
              nodeId="logic_review_node"
              nodeTitle="逻辑审校"
              onExecuteRevision={(instruction) => {
                void submitHitlSpecReview(instruction);
              }}
              onFinish={() => {
                void submitHitlSpecReview("");
              }}
              sessionData={sessionData}
              showSpecReviewActions={true}
              value={sessionData?.review_issues ?? []}
            />
            {message ? <p className="mt-3 text-xs text-red-600">{message}</p> : null}
          </div>
        );
      }
      return (
        <div>
          <HitlActionPanel
            busy={busy}
            initialClaimsText={claimsJsonForHitl}
            stage={hitlStage}
            visible={true}
            onSubmitAutoReviseClaims={submitAutoReviseClaims}
            onSubmitClaims={submitHitlClaims}
            onSubmitSpecReview={submitHitlSpecReview}
          />
          {message ? <p className="mt-3 text-xs text-red-600">{message}</p> : null}
        </div>
      );
    }

    return (
      <div className="space-y-3">
        <NodeStageRenderer
          mode={activeTab}
          nodeId={node.id}
          nodeTitle={selectedOutputConfig?.title ?? node.label}
          sessionData={sessionData}
          value={selectedNodeValue}
        />
        {activeTab === "polish" && node.id === "multimodal_draft_parser_node" ? (
          <section className="p-2">
            <h3 className="text-sm font-semibold text-gray-800">输入识别预览</h3>
            <p className="mt-1 text-xs text-gray-500">申请文件图片：{applicationImagesMeta.length} | 对比文件图片：{priorArtImagesMeta.length}</p>
            <div className="mt-3 grid gap-3 lg:grid-cols-2">
              <textarea className="h-44 w-full rounded border border-gray-200 bg-gray-50 p-2 text-xs text-gray-700" readOnly value={compareOriginalClaimsText || "暂无识别到的权利要求文本"} />
              <textarea className="h-44 w-full rounded border border-gray-200 bg-gray-50 p-2 text-xs text-gray-700" readOnly value={compareSpecificationText || "暂无识别到的说明书文本"} />
            </div>
          </section>
        ) : null}
      </div>
    );
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white text-gray-800">
      {isDesktopRuntime ? (
        <header className="flex h-11 flex-shrink-0 items-center border-b border-gray-200 bg-white">
          {isMacRuntime ? (
            // macOS: native traffic-light buttons are overlaid by the OS in the
            // top-left thanks to titleBarStyle: "Overlay". Reserve space for them.
            <div className="w-[78px] shrink-0" />
          ) : null}
          <div
            className="flex min-w-0 flex-1 select-none items-center gap-2 px-3 text-gray-800"
            data-tauri-drag-region
            onDoubleClick={() => void toggleMaximizeWindow()}
            onMouseDown={(event) => void startWindowDrag(event)}
          >
            <img alt="M-Cube" className="h-5 w-5 shrink-0" src="/icons/icon.svg" />
            <span className="truncate text-sm font-semibold">M-Cube</span>
          </div>
          {!isMacRuntime ? (
            <div className="flex items-center pr-1">
              <button
                aria-label="Minimize"
                className="inline-flex h-8 w-10 items-center justify-center rounded text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                onClick={() => void minimizeWindow()}
                type="button"
              >
                <span className="h-px w-3 bg-current" />
              </button>
              <button
                aria-label={isMaximized ? "Restore" : "Maximize"}
                className="inline-flex h-8 w-10 items-center justify-center rounded text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                onClick={() => void toggleMaximizeWindow()}
                type="button"
              >
                {isMaximized ? (
                  <span className="relative block h-3.5 w-3.5">
                    <span className="absolute left-1 top-0.5 h-2.5 w-2.5 border border-current bg-white" />
                    <span className="absolute left-0 top-1.5 h-2.5 w-2.5 border border-current bg-white" />
                  </span>
                ) : (
                  <span className="block h-3 w-3 border border-current" />
                )}
              </button>
              <button
                aria-label="Close"
                className="inline-flex h-8 w-10 items-center justify-center rounded text-gray-600 hover:bg-red-600 hover:text-white"
                onClick={() => void closeWindow()}
                type="button"
              >
                <span className="relative block h-3.5 w-3.5">
                  <span className="absolute left-0 top-1.5 h-px w-3.5 rotate-45 bg-current" />
                  <span className="absolute left-0 top-1.5 h-px w-3.5 -rotate-45 bg-current" />
                </span>
              </button>
            </div>
          ) : null}
        </header>
      ) : null}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <AppSidebar activeTab={activeTab} onSelect={setActiveTab} />
        <main className="h-full flex-1 overflow-hidden bg-white">
          <div className="mx-auto flex h-full max-w-5xl min-h-0 flex-col">
          {activeTab === "home" ? (
            <div className="flex-1 min-h-0 overflow-y-auto">
              <HomeView />
            </div>
          ) : null}

          {activeTab === "settings" ? (
            <div className="flex-1 min-h-0 overflow-y-auto">
              <SettingsView
                settings={llm}
                onSave={(next) => llm.setSettings(next)}
                onReset={llm.resetSettings}
              />
            </div>
          ) : null}

          {showWorkflowMain ? (
            <section className="flex h-full min-h-0 flex-col bg-white">
              <div className="flex-1 min-h-0 overflow-y-auto p-8 flex justify-center">
                <div className="w-full max-w-4xl space-y-12 pb-16">
                  {renderWorkflowStage()}
                </div>
              </div>
              <div className="flex-shrink-0 border-t border-gray-100 bg-white/90 py-6 backdrop-blur-md">
                <div className="mx-auto w-full max-w-5xl">
                  <BottomHorizontalTimeline
                    nodes={timelineNodes}
                    onSelectDoneNode={(id) => setSelectedNodeId(id)}
                    selectedNodeId={selectedNodeId}
                  />
                </div>
              </div>
            </section>
          ) : null}
        </div>
      </main>
    </div>
    </div>
  );
}

export default App;

