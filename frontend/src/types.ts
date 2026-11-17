export type JudgeId = "clef" | "openrouter_jev" | "openrouter_luna" | "openai_luna";
export type ProfileName = "auto" | "cqa" | "md2d" | "mtrag" | "nq";
export type MCMode = "order" | "top1" | "off";
export type WorkbenchMode = "query" | "retrieve" | "chat" | "compare";
export type PipelineStepId =
  | "query"
  | "categories"
  | "clusters"
  | "pages"
  | "rescore"
  | "sections"
  | "passages"
  | "ranking"
  | "ranked"
  | "answer";

export interface RuntimeCredentials {
  clef_api_url?: string | null;
  clef_api_key?: string | null;
  openrouter_api_key?: string | null;
  openai_api_key?: string | null;
  generator_model?: string | null;
  generator_api_key?: string | null;
  generator_api_base?: string | null;
  generator_api_version?: string | null;
}

export interface LocalClientSettings {
  apiKey: string;
  clefApiUrl: string;
  clefApiKey: string;
  openrouterApiKey: string;
  openaiApiKey: string;
  generatorModel: string;
  generatorApiKey: string;
  generatorApiBase: string;
  generatorApiVersion: string;
}

export interface JudgeStatusInfo {
  judge_id: JudgeId;
  model: string;
  endpoint: string;
  configured: boolean;
}

export interface HealthResponse {
  status: string;
  default_judge: JudgeId;
  judges: JudgeStatusInfo[];
  generator_model?: string;
  generator_configured?: boolean;
}

export interface CategoryRead {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  is_manual_description: boolean;
  order_index: number;
}

export interface ClusterRead {
  id: string;
  cluster_index: number;
  summary: string;
  page_count: number;
  created_at: string;
}

export interface NamespaceRead {
  id: string;
  name: string;
  slug: string;
  description: string | null;
  system_prompt: string | null;
  default_profile: ProfileName;
  created_at: string;
  page_count: number;
  category_count: number;
  cluster_count: number;
}

export interface NamespaceDetailRead extends NamespaceRead {
  categories: CategoryRead[];
  clusters: ClusterRead[];
}

export interface DocumentPageRead {
  id: string;
  namespace_id: string;
  category_id: string | null;
  cluster_id: string | null;
  cluster_rank: number | null;
  doc_order: number;
  filename: string;
  title: string;
  first_sentence: string;
  headings_summary: string[];
  content_type: string;
  file_size: number;
  passage_count: number;
  status: "QUEUED" | "PROCESSING" | "READY" | "FAILED";
  error_message: string | null;
  created_at: string;
}

export interface PassageRead {
  id: string;
  page_id: string;
  heading_id: string | null;
  passage_index: number;
  heading_path: string;
  content: string;
  token_estimate: number;
}

export interface DocumentUploadResponse {
  page: DocumentPageRead;
  merged_into_existing: boolean;
}

export interface StageMetrics {
  stage: string;
  candidates_in: number;
  candidates_out: number;
  requests_made: number;
  input_tokens: number;
  cost_usd: number;
  latency_ms: number;
}

export interface NavigationTelemetry {
  judge: JudgeId;
  profile_used: ProfileName;
  mc_mode: MCMode;
  total_requests: number;
  total_input_tokens: number;
  total_cost_usd: number;
  total_latency_ms: number;
  notepad_page_ids: string[];
  stages: StageMetrics[];
}

export interface RankedPassageItem {
  rank: number;
  passage_id: string;
  page_id: string;
  page_title: string;
  heading_path: string;
  passage_index: number;
  content: string;
  p_page_bin: number | null;
  p_para_bin: number;
  stage2_score: number;
  p_mul: number | null;
}

export interface RetrieveResponse {
  namespace_id: string;
  query: string;
  passages: RankedPassageItem[];
  telemetry: NavigationTelemetry;
}

export interface QueryResponse {
  namespace_id: string;
  query: string;
  answer: string;
  passages: RankedPassageItem[];
  telemetry: NavigationTelemetry;
  generator_latency_ms: number;
}

export interface ChatResponse {
  namespace_id: string;
  session_id: string;
  answer: string;
  notepad_page_ids: string[];
  passages: RankedPassageItem[];
  telemetry: NavigationTelemetry;
  generator_latency_ms: number;
}

export interface JudgeComparisonEntry {
  judge: JudgeId;
  passages: RankedPassageItem[];
  telemetry: NavigationTelemetry;
}

export interface CompareJudgesResponse {
  namespace_id: string;
  query: string;
  results: JudgeComparisonEntry[];
}

export interface JudgeTestResponse {
  judge: JudgeId;
  model: string;
  endpoint: string;
  ok: boolean;
  latency_ms: number;
  p_relevant: number;
  p_irrelevant: number;
  choice_winner: string;
}

export interface GeneratorTestResponse {
  model: string;
  endpoint: string;
  ok: boolean;
  latency_ms: number;
  reply: string;
}

export interface ClusterBuildResponse {
  namespace_id: string;
  clusters_built: number;
  pages_clustered: number;
  clusters: ClusterRead[];
}

export interface ChatMessageEntry {
  role: "user" | "assistant";
  content: string;
  passages?: RankedPassageItem[];
  telemetry?: NavigationTelemetry;
  generator_latency_ms?: number;
}
