from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from dec_rag.config import JudgeId, Settings

ProfileName = Literal["auto", "cqa", "md2d", "mtrag", "nq"]
MCMode = Literal["order", "top1", "off"]
L0Mode = Literal["none", "categories", "clusters"]
PageStatus = Literal["QUEUED", "PROCESSING", "READY", "FAILED"]


@dataclass(frozen=True)
class ResolvedProfileConfig:
    profile: ProfileName
    l0_mode: L0Mode
    cluster_keep_ratio: float
    page_batch_max_tokens: int | None
    min_titles_per_batch: int
    group_pages_by_category: bool
    two_pass_top_pages: int | None
    k_pages: int
    enable_heading_stage: bool
    heading_passage_threshold: int
    top_headings: int
    max_heading_passages: int
    tiered_passage_pool: bool
    multiply_page_prob: bool
    r_passages: int
    m_shortlist: int
    mc_mode: MCMode
    deduplicate: bool


class RuntimeCredentials(BaseModel):
    clef_api_url: str | None = None
    clef_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    generator_model: str | None = None
    generator_api_key: str | None = None
    generator_api_base: str | None = None
    generator_api_version: str | None = None


def apply_runtime_credentials(
    settings: Settings,
    creds: RuntimeCredentials | None,
) -> Settings:
    """Return an ephemeral copy of Settings with non-empty runtime overrides applied."""
    if creds is None:
        return settings
    updates: dict[str, str] = {}
    if creds.clef_api_url and creds.clef_api_url.strip():
        updates["CLEF_API_URL"] = creds.clef_api_url.strip()
    if creds.clef_api_key and creds.clef_api_key.strip():
        updates["CLEF_API_KEY"] = creds.clef_api_key.strip()
    if creds.openrouter_api_key and creds.openrouter_api_key.strip():
        updates["OPENROUTER_API_KEY"] = creds.openrouter_api_key.strip()
    if creds.openai_api_key and creds.openai_api_key.strip():
        updates["OPENAI_API_KEY"] = creds.openai_api_key.strip()
    if creds.generator_model and creds.generator_model.strip():
        updates["GENERATOR_MODEL"] = creds.generator_model.strip()
    if creds.generator_api_base and creds.generator_api_base.strip():
        updates["GENERATOR_API_BASE"] = creds.generator_api_base.strip()
    if creds.generator_api_version and creds.generator_api_version.strip():
        updates["GENERATOR_API_VERSION"] = creds.generator_api_version.strip()
    if not updates:
        return settings
    return settings.model_copy(update=updates)


class NavigationOverrides(BaseModel):
    profile: ProfileName | None = None
    judge: JudgeId | None = None
    k_pages: int | None = Field(default=None, ge=1, le=200)
    r_passages: int | None = Field(default=None, ge=1, le=255)
    m_shortlist: int | None = Field(default=None, ge=1, le=255)
    mc_mode: MCMode | None = None
    enable_heading_stage: bool | None = None
    deduplicate: bool = True
    runtime_credentials: RuntimeCredentials | None = None


def resolve_profile_config(
    profile: ProfileName,
    *,
    has_categories: bool,
    has_clusters: bool,
    total_ready_pages: int,
    est_page_tokens: int = 0,
    conversational: bool = False,
    overrides: NavigationOverrides | None = None,
) -> ResolvedProfileConfig:
    """Resolve exact paper navigation budgets per Spec §7.1 & Appendix B."""
    if profile == "cqa":
        l0_mode: L0Mode = "none"
        page_batch_max_tokens: int | None = None
        min_titles_per_batch = 1
        group_pages_by_category = False
        two_pass_top_pages: int | None = None
        k_pages = 3
        enable_heading_stage = False
        tiered_passage_pool = False
        multiply_page_prob = False
        r_passages = 20
        m_shortlist = 20
        mc_mode: MCMode = "order"
    elif profile == "md2d":
        l0_mode = "categories" if has_categories else "none"
        page_batch_max_tokens = None
        min_titles_per_batch = 1
        group_pages_by_category = False
        two_pass_top_pages = None
        k_pages = 10
        enable_heading_stage = False
        tiered_passage_pool = False
        multiply_page_prob = False
        r_passages = 20
        m_shortlist = 20
        mc_mode = "order"
    elif profile == "mtrag":
        l0_mode = "none"
        page_batch_max_tokens = 10_000
        min_titles_per_batch = 1
        group_pages_by_category = True
        two_pass_top_pages = 50
        k_pages = 10
        enable_heading_stage = True
        tiered_passage_pool = False
        multiply_page_prob = False
        r_passages = 20
        m_shortlist = 20
        mc_mode = "top1"
    elif profile == "nq":
        l0_mode = "clusters" if has_clusters else "none"
        page_batch_max_tokens = 20_000
        min_titles_per_batch = 50
        group_pages_by_category = False
        two_pass_top_pages = None
        k_pages = 30
        enable_heading_stage = False
        tiered_passage_pool = True
        multiply_page_prob = True
        r_passages = 20
        m_shortlist = 5
        mc_mode = "order"
    else:  # "auto"
        if has_categories:
            l0_mode = "categories"
        elif has_clusters:
            l0_mode = "clusters"
        else:
            l0_mode = "none"

        if total_ready_pages <= 652 and est_page_tokens <= 20_000:
            page_batch_max_tokens = None
        else:
            page_batch_max_tokens = 15_000

        min_titles_per_batch = 1
        group_pages_by_category = has_categories
        two_pass_top_pages = 50
        k_pages = 10 if (conversational or total_ready_pages > 652) else 3
        enable_heading_stage = True
        tiered_passage_pool = k_pages > 10
        multiply_page_prob = k_pages > 10
        r_passages = 20
        m_shortlist = 20
        mc_mode = "order"

    deduplicate = True
    if overrides is not None:
        if overrides.k_pages is not None:
            k_pages = overrides.k_pages
            if profile == "auto":
                tiered_passage_pool = k_pages > 10
                multiply_page_prob = k_pages > 10
        if overrides.r_passages is not None:
            r_passages = overrides.r_passages
        if overrides.m_shortlist is not None:
            m_shortlist = overrides.m_shortlist
        if overrides.mc_mode is not None:
            mc_mode = overrides.mc_mode
        if overrides.enable_heading_stage is not None:
            enable_heading_stage = overrides.enable_heading_stage
        deduplicate = overrides.deduplicate

    m_shortlist = min(m_shortlist, r_passages, 255)

    return ResolvedProfileConfig(
        profile=profile,
        l0_mode=l0_mode,
        cluster_keep_ratio=0.15,
        page_batch_max_tokens=page_batch_max_tokens,
        min_titles_per_batch=min_titles_per_batch,
        group_pages_by_category=group_pages_by_category,
        two_pass_top_pages=two_pass_top_pages,
        k_pages=k_pages,
        enable_heading_stage=enable_heading_stage,
        heading_passage_threshold=60,
        top_headings=8,
        max_heading_passages=60,
        tiered_passage_pool=tiered_passage_pool,
        multiply_page_prob=multiply_page_prob,
        r_passages=r_passages,
        m_shortlist=m_shortlist,
        mc_mode=mc_mode,
        deduplicate=deduplicate,
    )


# --- Health ---


class JudgeStatusInfo(BaseModel):
    judge_id: JudgeId
    model: str
    endpoint: str
    configured: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    default_judge: JudgeId
    judges: list[JudgeStatusInfo]
    generator_model: str = ""
    generator_configured: bool = False


# --- Namespaces, Categories & Clusters ---


class NamespaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str | None = Field(default=None, max_length=255)
    description: str | None = None
    system_prompt: str | None = None
    default_profile: ProfileName = "auto"


class CategoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    description: str | None
    is_manual_description: bool
    order_index: int


class ClusterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    cluster_index: int
    summary: str
    page_count: int
    created_at: datetime


class NamespaceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    description: str | None
    system_prompt: str | None
    default_profile: ProfileName
    created_at: datetime
    page_count: int = 0
    category_count: int = 0
    cluster_count: int = 0


class NamespaceDetailRead(NamespaceRead):
    categories: list[CategoryRead] = Field(default_factory=list)
    clusters: list[ClusterRead] = Field(default_factory=list)


class ClusterBuildRequest(BaseModel):
    num_clusters: int | None = Field(default=None, ge=1, le=5000)
    force: bool = False
    runtime_credentials: RuntimeCredentials | None = None


class ClusterBuildResponse(BaseModel):
    namespace_id: UUID
    clusters_built: int
    pages_clustered: int
    clusters: list[ClusterRead]


# --- Documents & Passages ---


class DocumentPageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    namespace_id: UUID
    category_id: UUID | None
    cluster_id: UUID | None
    cluster_rank: int | None
    doc_order: int
    filename: str
    title: str
    first_sentence: str
    headings_summary: list[str]
    content_type: str
    file_size: int
    passage_count: int
    status: PageStatus
    error_message: str | None
    created_at: datetime


class PassageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    page_id: UUID
    heading_id: UUID | None
    passage_index: int
    heading_path: str
    content: str
    token_estimate: int


class DocumentUploadResponse(BaseModel):
    page: DocumentPageRead
    merged_into_existing: bool = False


# --- Retrieval, Query, Chat & Compare-Judges ---


class DialogueTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class StageMetrics(BaseModel):
    stage: str
    candidates_in: int
    candidates_out: int
    requests_made: int
    input_tokens: int
    cost_usd: float
    latency_ms: float


class NavigationTelemetry(BaseModel):
    judge: JudgeId
    profile_used: ProfileName
    mc_mode: MCMode
    total_requests: int
    total_input_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    notepad_page_ids: list[str] = Field(default_factory=list)
    stages: list[StageMetrics] = Field(default_factory=list)


class RankedPassageItem(BaseModel):
    rank: int
    passage_id: UUID
    page_id: UUID
    page_title: str
    heading_path: str
    passage_index: int
    content: str
    p_page_bin: float | None = None
    p_para_bin: float
    stage2_score: float
    p_mul: float | None = None


class RetrieveRequest(NavigationOverrides):
    query: str = Field(..., min_length=1)
    scenario: str | None = None


class RetrieveResponse(BaseModel):
    namespace_id: UUID
    query: str
    passages: list[RankedPassageItem]
    telemetry: NavigationTelemetry


class QueryRequest(RetrieveRequest):
    top_k_context: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    namespace_id: UUID
    query: str
    answer: str
    passages: list[RankedPassageItem]
    telemetry: NavigationTelemetry
    generator_latency_ms: float


class ChatRequest(NavigationOverrides):
    message: str = Field(..., min_length=1)
    session_id: UUID | None = None
    turns: list[DialogueTurn] | None = None
    top_k_context: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    namespace_id: UUID
    session_id: UUID
    answer: str
    notepad_page_ids: list[str]
    passages: list[RankedPassageItem]
    telemetry: NavigationTelemetry
    generator_latency_ms: float


class CompareJudgesRequest(NavigationOverrides):
    query: str = Field(..., min_length=1)
    scenario: str | None = None
    judges: list[JudgeId] = Field(..., min_length=1)


class JudgeComparisonEntry(BaseModel):
    judge: JudgeId
    passages: list[RankedPassageItem]
    telemetry: NavigationTelemetry


class CompareJudgesResponse(BaseModel):
    namespace_id: UUID
    query: str
    results: list[JudgeComparisonEntry]


# --- API Keys ---


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    namespace_id: UUID | None = None


class ApiKeyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    key_prefix: str
    namespace_id: UUID | None
    created_at: datetime
    last_used_at: datetime | None


class ApiKeyCreatedResponse(ApiKeyRead):
    api_key: str


    # --- Live Judge Diagnostic ---


class JudgeTestRequest(BaseModel):
    judge: JudgeId
    runtime_credentials: RuntimeCredentials | None = None


class JudgeTestResponse(BaseModel):
    judge: JudgeId
    model: str
    endpoint: str
    ok: bool = True
    latency_ms: float
    p_relevant: float
    p_irrelevant: float
    choice_winner: str


class GeneratorTestRequest(BaseModel):
    runtime_credentials: RuntimeCredentials | None = None


class GeneratorTestResponse(BaseModel):
    model: str
    endpoint: str
    ok: bool = True
    latency_ms: float
    reply: str