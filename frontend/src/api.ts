import type {
  ChatResponse,
  ClusterBuildResponse,
  CompareJudgesResponse,
  DocumentPageRead,
  DocumentUploadResponse,
  GeneratorTestResponse,
  HealthResponse,
  JudgeId,
  JudgeTestResponse,
  LocalClientSettings,
  MCMode,
  NamespaceDetailRead,
  NamespaceRead,
  PassageRead,
  PipelineStepId,
  ProfileName,
  QueryResponse,
  RetrieveResponse,
  RuntimeCredentials,
} from "./types";

const STORAGE_KEY = "dec_rag_client_settings_v1";

export const DEFAULT_LOCAL_SETTINGS: LocalClientSettings = {
  apiKey: "",
  clefApiUrl: "",
  clefApiKey: "",
  openrouterApiKey: "",
  openaiApiKey: "",
  generatorModel: "",
  generatorApiKey: "",
  generatorApiBase: "",
  generatorApiVersion: "",
};

export function loadLocalSettings(): LocalClientSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_LOCAL_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<LocalClientSettings>;
    return { ...DEFAULT_LOCAL_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_LOCAL_SETTINGS };
  }
}

export function saveLocalSettings(settings: LocalClientSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function toRuntimeCredentials(
  local: LocalClientSettings
): RuntimeCredentials | undefined {
  const creds: RuntimeCredentials = {};
  if (local.clefApiUrl.trim()) creds.clef_api_url = local.clefApiUrl.trim();
  if (local.clefApiKey.trim()) creds.clef_api_key = local.clefApiKey.trim();
  if (local.openrouterApiKey.trim()) creds.openrouter_api_key = local.openrouterApiKey.trim();
  if (local.openaiApiKey.trim()) creds.openai_api_key = local.openaiApiKey.trim();
  if (local.generatorModel.trim()) creds.generator_model = local.generatorModel.trim();
  if (local.generatorApiKey.trim()) creds.generator_api_key = local.generatorApiKey.trim();
  if (local.generatorApiBase.trim()) creds.generator_api_base = local.generatorApiBase.trim();
  if (local.generatorApiVersion.trim()) creds.generator_api_version = local.generatorApiVersion.trim();
  return Object.keys(creds).length > 0 ? creds : undefined;
}

export function isJudgeReady(
  judgeId: JudgeId,
  serverConfigured: boolean,
  local: LocalClientSettings
): boolean {
  if (serverConfigured) return true;
  if (judgeId === "clef") {
    return Boolean(local.clefApiUrl.trim() && local.clefApiKey.trim());
  }
  if (judgeId === "openrouter_jev" || judgeId === "openrouter_luna") {
    return Boolean(local.openrouterApiKey.trim());
  }
  if (judgeId === "openai_luna") {
    return Boolean(local.openaiApiKey.trim());
  }
  return false;
}

async function apiFetch<T>(
  path: string,
  local: LocalClientSettings,
  init?: RequestInit
): Promise<T> {
  const headers = new Headers(init?.headers || {});
  if (local.apiKey.trim()) {
    headers.set("X-API-Key", local.apiKey.trim());
  }
  if (init?.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const resp = await fetch(`/api/v1${path}`, {
    ...init,
    headers,
  });

  if (resp.status === 204) {
    return undefined as T;
  }

  const contentType = resp.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await resp.json()
    : await resp.text();

  if (!resp.ok) {
    const detail =
      typeof data === "object" && data !== null && "detail" in data
        ? typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail)
        : String(data || resp.statusText);
    throw new Error(`HTTP ${resp.status}: ${detail}`);
  }

  return data as T;
}

export function subscribeToRunSteps(
  runId: string,
  onStep: (step: PipelineStepId) => void
): () => void {
  const es = new EventSource(`/api/v1/runs/${encodeURIComponent(runId)}/events`);
  es.onmessage = (ev) => {
    try {
      const parsed = JSON.parse(ev.data) as { step?: string; done?: boolean };
      if (parsed.done || parsed.step === "done") {
        es.close();
        return;
      }
      if (parsed.step) {
        onStep(parsed.step as PipelineStepId);
      }
    } catch {
      // ignore malformed keepalive
    }
  };
  es.onerror = () => {
    es.close();
  };
  return () => es.close();
}

export interface NavParams {
  judge: JudgeId;
  profile: ProfileName;
  mc_mode: MCMode;
  k_pages?: number;
  r_passages?: number;
  m_shortlist?: number;
}

export const api = {
  getHealth: (local: LocalClientSettings) =>
    apiFetch<HealthResponse>("/health", local),

  testJudge: (judge: JudgeId, local: LocalClientSettings) =>
    apiFetch<JudgeTestResponse>("/judges/test", local, {
      method: "POST",
      body: JSON.stringify({
        judge,
        runtime_credentials: toRuntimeCredentials(local),
      }),
    }),

  testGenerator: (local: LocalClientSettings) =>
    apiFetch<GeneratorTestResponse>("/judges/test-generator", local, {
      method: "POST",
      body: JSON.stringify({
        runtime_credentials: toRuntimeCredentials(local),
      }),
    }),

  listNamespaces: (local: LocalClientSettings) =>
    apiFetch<NamespaceRead[]>("/namespaces", local),

  getNamespace: (nsId: string, local: LocalClientSettings) =>
    apiFetch<NamespaceDetailRead>(`/namespaces/${nsId}`, local),

  createNamespace: (
    payload: {
      name: string;
      description?: string;
      system_prompt?: string;
      default_profile: ProfileName;
    },
    local: LocalClientSettings
  ) =>
    apiFetch<NamespaceRead>("/namespaces", local, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  deleteNamespace: (nsId: string, local: LocalClientSettings) =>
    apiFetch<void>(`/namespaces/${nsId}`, local, { method: "DELETE" }),

  buildClusters: (
    nsId: string,
    payload: { num_clusters?: number; force: boolean },
    local: LocalClientSettings
  ) =>
    apiFetch<ClusterBuildResponse>(`/namespaces/${nsId}/clusters/build`, local, {
      method: "POST",
      body: JSON.stringify({
        ...payload,
        runtime_credentials: toRuntimeCredentials(local),
      }),
    }),

  listDocuments: (nsId: string, local: LocalClientSettings) =>
    apiFetch<DocumentPageRead[]>(`/namespaces/${nsId}/documents`, local),

  uploadDocument: (
    nsId: string,
    file: File,
    options: { category?: string; mergePartSuffix?: boolean },
    local: LocalClientSettings
  ) => {
    const form = new FormData();
    form.append("file", file);
    if (options.category?.trim()) {
      form.append("category", options.category.trim());
    }
    if (options.mergePartSuffix) {
      form.append("merge_part_suffix", "true");
    }
    return apiFetch<DocumentUploadResponse>(
      `/namespaces/${nsId}/documents`,
      local,
      {
        method: "POST",
        body: form,
      }
    );
  },

  getPassages: (nsId: string, pageId: string, local: LocalClientSettings) =>
    apiFetch<PassageRead[]>(
      `/namespaces/${nsId}/documents/${pageId}/passages`,
      local
    ),

  deleteDocument: (nsId: string, pageId: string, local: LocalClientSettings) =>
    apiFetch<void>(`/namespaces/${nsId}/documents/${pageId}`, local, {
      method: "DELETE",
    }),

  retrieve: (
    nsId: string,
    query: string,
    scenario: string,
    nav: NavParams,
    local: LocalClientSettings,
    runId?: string
  ) =>
    apiFetch<RetrieveResponse>(`/namespaces/${nsId}/retrieve`, local, {
      method: "POST",
      headers: runId ? { "X-Run-Id": runId } : undefined,
      body: JSON.stringify({
        query,
        scenario: scenario.trim() || null,
        ...nav,
        runtime_credentials: toRuntimeCredentials(local),
      }),
    }),

  query: (
    nsId: string,
    query: string,
    scenario: string,
    nav: NavParams,
    local: LocalClientSettings,
    runId?: string
  ) =>
    apiFetch<QueryResponse>(`/namespaces/${nsId}/query`, local, {
      method: "POST",
      headers: runId ? { "X-Run-Id": runId } : undefined,
      body: JSON.stringify({
        query,
        scenario: scenario.trim() || null,
        ...nav,
        runtime_credentials: toRuntimeCredentials(local),
      }),
    }),

  chat: (
    nsId: string,
    message: string,
    sessionId: string | null,
    nav: NavParams,
    local: LocalClientSettings,
    runId?: string
  ) =>
    apiFetch<ChatResponse>(`/namespaces/${nsId}/chat`, local, {
      method: "POST",
      headers: runId ? { "X-Run-Id": runId } : undefined,
      body: JSON.stringify({
        message,
        session_id: sessionId,
        ...nav,
        runtime_credentials: toRuntimeCredentials(local),
      }),
    }),

  compareJudges: (
    nsId: string,
    query: string,
    scenario: string,
    judges: JudgeId[],
    nav: Omit<NavParams, "judge">,
    local: LocalClientSettings,
    runId?: string
  ) =>
    apiFetch<CompareJudgesResponse>(
      `/namespaces/${nsId}/compare-judges`,
      local,
      {
        method: "POST",
        headers: runId ? { "X-Run-Id": runId } : undefined,
        body: JSON.stringify({
          query,
          scenario: scenario.trim() || null,
          judges,
          ...nav,
          runtime_credentials: toRuntimeCredentials(local),
        }),
      }
    ),
};
