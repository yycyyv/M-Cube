import type {
  ApiEnvelope,
  CompareStartRequest,
  DraftContinueRequest,
  DraftStartRequest,
  ExportWordMode,
  FilePreviewData,
  FilePreviewRequest,
  OAStartRequest,
  PolishStartRequest,
  UploadFileData,
} from "@/types/api";

export class ApiClientError extends Error {
  readonly code: string;
  readonly httpStatus: number;
  readonly retryable: boolean;
  readonly sessionId: string;
  readonly details: Record<string, unknown>;

  constructor(params: {
    code: string;
    message: string;
    httpStatus: number;
    retryable: boolean;
    sessionId: string;
    details?: Record<string, unknown>;
  }) {
    super(params.message);
    this.code = params.code;
    this.httpStatus = params.httpStatus;
    this.retryable = params.retryable;
    this.sessionId = params.sessionId;
    this.details = params.details ?? {};
  }
}

interface ApiClientOptions {
  baseUrl: string;
  apiKey?: string;
}

type RequestHeaders = Record<string, string>;

const NETWORK_RETRY_ATTEMPTS = 18;
const NETWORK_RETRY_BASE_DELAY_MS = 300;
const NETWORK_RETRY_MAX_DELAY_MS = 3000;

export class ApiClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;

  constructor(options: ApiClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
  }

  async startDraft(payload: DraftStartRequest, headers?: RequestHeaders): Promise<ApiEnvelope> {
    return this.request<ApiEnvelope>("/api/v1/draft/start", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  }

  async continueDraft(payload: DraftContinueRequest, headers?: RequestHeaders): Promise<ApiEnvelope> {
    return this.request<ApiEnvelope>("/api/v1/draft/continue", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  }

  async startOa(payload: OAStartRequest, headers?: RequestHeaders): Promise<ApiEnvelope> {
    return this.request<ApiEnvelope>("/api/v1/oa/start", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  }

  async startCompare(payload: CompareStartRequest, headers?: RequestHeaders): Promise<ApiEnvelope> {
    return this.request<ApiEnvelope>("/api/v1/compare/start", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  }

  async startPolish(payload: PolishStartRequest, headers?: RequestHeaders): Promise<ApiEnvelope> {
    return this.request<ApiEnvelope>("/api/v1/polish/start", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  }

  async previewFiles(payload: FilePreviewRequest, headers?: RequestHeaders): Promise<FilePreviewData> {
    const envelope = await this.request<ApiEnvelope<FilePreviewData>>("/api/v1/files/preview", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!envelope.data) {
      throw new ApiClientError({
        code: "E500_INTERNAL_ERROR",
        message: "Preview response missing data.",
        httpStatus: 500,
        retryable: false,
        sessionId: envelope.session_id,
      });
    }
    return envelope.data;
  }

  async exportSessionWord(sessionId: string, mode: ExportWordMode, headers?: RequestHeaders): Promise<Blob> {
    return this.requestBlob(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/export-word?mode=${encodeURIComponent(mode)}`,
      {
        method: "GET",
        headers,
      },
    );
  }

  async uploadFile(file: File, purpose?: string): Promise<UploadFileData> {
    const formData = new FormData();
    formData.append("file", file);
    if (purpose) {
      formData.append("purpose", purpose);
    }
    const envelope = await this.request<ApiEnvelope<UploadFileData>>("/api/v1/files/upload", {
      method: "POST",
      body: formData,
    });
    if (!envelope.data) {
      throw new ApiClientError({
        code: "E500_INTERNAL_ERROR",
        message: "Upload response missing data.",
        httpStatus: 500,
        retryable: false,
        sessionId: envelope.session_id,
      });
    }
    return envelope.data;
  }

  async getSession(sessionId: string): Promise<ApiEnvelope> {
    return this.request<ApiEnvelope>(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, {
      method: "GET",
    });
  }

  async cancelSession(sessionId: string): Promise<ApiEnvelope> {
    return this.request<ApiEnvelope>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/cancel`, {
      method: "POST",
    });
  }

  private async fetchWithNetworkRetry(url: string, init: RequestInit): Promise<Response> {
    let lastError: unknown;
    for (let attempt = 1; attempt <= NETWORK_RETRY_ATTEMPTS; attempt += 1) {
      try {
        return await fetch(url, init);
      } catch (error) {
        lastError = error;
        if (attempt >= NETWORK_RETRY_ATTEMPTS) {
          break;
        }
        const delay = Math.min(NETWORK_RETRY_BASE_DELAY_MS * 2 ** (attempt - 1), NETWORK_RETRY_MAX_DELAY_MS);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
    throw lastError instanceof Error ? lastError : new Error("Network request failed.");
  }

  private async requestBlob(path: string, init: RequestInit): Promise<Blob> {
    const headers = new Headers(init.headers);
    if (this.apiKey) {
      headers.set("X-API-Key", this.apiKey);
    }

    let response: Response;
    try {
      response = await this.fetchWithNetworkRetry(`${this.baseUrl}${path}`, { ...init, headers });
    } catch (error) {
      throw new ApiClientError({
        code: "E_NETWORK",
        message: error instanceof Error ? error.message : "Network request failed.",
        httpStatus: 0,
        retryable: true,
        sessionId: "unknown",
      });
    }

    if (!response.ok) {
      let errorBody: ApiEnvelope | null = null;
      try {
        errorBody = (await response.json()) as ApiEnvelope;
      } catch {
        errorBody = null;
      }
      throw new ApiClientError({
        code: errorBody?.error?.code ?? "E_UNKNOWN",
        message: errorBody?.error?.message ?? `HTTP ${response.status}`,
        httpStatus: response.status,
        retryable: errorBody?.error?.retryable ?? false,
        sessionId: errorBody?.session_id ?? "unknown",
        details: errorBody?.error?.details ?? {},
      });
    }

    return response.blob();
  }

  private async request<TEnvelope extends ApiEnvelope<any>>(path: string, init: RequestInit): Promise<TEnvelope> {
    const headers = new Headers(init.headers);
    const isFormDataBody = typeof FormData !== "undefined" && init.body instanceof FormData;
    if (!isFormDataBody) {
      headers.set("Content-Type", "application/json");
    }
    if (this.apiKey) {
      headers.set("X-API-Key", this.apiKey);
    }

    let response: Response;
    try {
      response = await this.fetchWithNetworkRetry(`${this.baseUrl}${path}`, { ...init, headers });
    } catch (error) {
      throw new ApiClientError({
        code: "E_NETWORK",
        message: error instanceof Error ? error.message : "Network request failed.",
        httpStatus: 0,
        retryable: true,
        sessionId: "unknown",
      });
    }

    const body = (await response.json()) as TEnvelope;
    if (!response.ok || body.error) {
      throw new ApiClientError({
        code: body.error?.code ?? "E_UNKNOWN",
        message: body.error?.message ?? `HTTP ${response.status}`,
        httpStatus: response.status,
        retryable: body.error?.retryable ?? false,
        sessionId: body.session_id ?? "unknown",
        details: body.error?.details ?? {},
      });
    }
    return body;
  }
}

const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
// API key is read from process environment only; it is never written to local/session storage.
const defaultApiKey = import.meta.env.VITE_API_KEY;

export const apiClient = new ApiClient({
  baseUrl: defaultBaseUrl,
  apiKey: defaultApiKey,
});

