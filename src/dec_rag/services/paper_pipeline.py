import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import time
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.models import (
    CategoryNode,
    DocumentPage,
    Namespace,
    Passage,
    SectionHeading,
    TopicCluster,
)
from dec_rag.schemas import (
    DialogueTurn,
    NavigationOverrides,
    NavigationTelemetry,
    ProfileName,
    RankedPassageItem,
    ResolvedProfileConfig,
    StageMetrics,
    resolve_profile_config,
)
from dec_rag.services.judges.base import BinaryBatchResult, DecisionJudge
from dec_rag.services.parser import estimate_tokens

STAGE2_MAX_BATCH_TOKENS = 25_000
CLUSTER_MAX_BATCH_TOKENS = 20_000


@dataclass
class NavigationOutput:
    passages: list[RankedPassageItem]
    telemetry: NavigationTelemetry
    updated_notepad_page_ids: list[str]


@dataclass
class _CandidatePassage:
    passage: Passage
    page: DocumentPage
    p_page_bin: float | None
    p_para_bin: float = 0.0
    stage2_score: float = 0.0
    p_mul: float | None = None


def _format_query_header(
    *,
    query: str | None,
    scenario: str | None,
    dialogue_turns: Sequence[DialogueTurn] | None,
    conversational: bool,
) -> str:
    if conversational and dialogue_turns:
        recent = list(dialogue_turns[-6:])
        turns_text = "\n".join(f"{t.role.upper()}: {t.content}" for t in recent)
        return f"Conversation (most recent {len(recent)} turns):\n{turns_text}"

    q = (query or "").strip()
    if scenario and scenario.strip():
        return f"Scenario: {scenario.strip()}\nUser Question: {q}"
    return f"User Question: {q}"


def _format_page_entry(index_1based: int, page: DocumentPage) -> str:
    parts = [f"Page {index_1based}: {page.title}"]
    if page.first_sentence:
        parts.append(f"  Opening: {page.first_sentence}")
    if page.headings_summary:
        sections = "; ".join(page.headings_summary[:12])
        parts.append(f"  Sections: {sections}")
    return "\n".join(parts)


def _split_page_batches(
    pages: Sequence[DocumentPage],
    cfg: ResolvedProfileConfig,
    base_header_tokens: int,
) -> list[list[DocumentPage]]:
    if not pages:
        return []
    if cfg.page_batch_max_tokens is None:
        return [list(pages)]

    max_tokens = cfg.page_batch_max_tokens
    batches: list[list[DocumentPage]] = []
    current_batch: list[DocumentPage] = []
    current_tokens = base_header_tokens
    last_cat_id: UUID | None = pages[0].category_id

    for p in pages:
        entry_tokens = estimate_tokens(_format_page_entry(len(current_batch) + 1, p)) + 16
        category_changed = cfg.group_pages_by_category and (p.category_id != last_cat_id)

        would_exceed = (current_tokens + entry_tokens > max_tokens) and (
            len(current_batch) >= cfg.min_titles_per_batch
        )
        # Keep categories cohesive when approaching token limit in grouped mode
        cat_boundary_flush = (
            category_changed
            and current_batch
            and (current_tokens + entry_tokens > int(max_tokens * 0.85))
            and len(current_batch) >= cfg.min_titles_per_batch
        )

        if current_batch and (would_exceed or cat_boundary_flush):
            batches.append(current_batch)
            current_batch = []
            current_tokens = base_header_tokens

        current_batch.append(p)
        current_tokens += entry_tokens
        last_cat_id = p.category_id

    if current_batch:
        batches.append(current_batch)
    return batches


def _render_grouped_passages(
    items: Sequence[ tuple[DocumentPage, Passage] ],
    *,
    unit_label: str,
) -> str:
    """Render passages numbered 1..N grouped under parent Page Title & Section in reading order."""
    lines: list[str] = []
    current_page_id: UUID | None = None
    current_heading_path: str | None = None

    for idx, (page, passage) in enumerate(items, start=1):
        if page.id != current_page_id:
            if lines:
                lines.append("")
            lines.append(f"=== Page: {page.title} ===")
            current_page_id = page.id
            current_heading_path = None

        h_path = passage.heading_path or ""
        if h_path and h_path != current_heading_path:
            lines.append(f"--- Section: {h_path} ---")
            current_heading_path = h_path
        elif not h_path:
            current_heading_path = ""

        lines.append(f"[{unit_label} {idx}] {passage.content}")

    return "\n".join(lines)


async def _score_page_batches(
    judge: DecisionJudge,
    batches: Sequence[Sequence[DocumentPage]],
    *,
    query_header: str,
    notepad_titles: Sequence[str],
    conversational: bool,
) -> tuple[dict[UUID, float], int, int, float]:
    notepad_block = ""
    if notepad_titles:
        listed = "\n".join(f"- {t}" for t in notepad_titles)
        notepad_block = f"\n\nRecent Notepad Pages (most recent first):\n{listed}"

    async def run_one(batch: Sequence[DocumentPage]) -> tuple[dict[UUID, float], BinaryBatchResult]:
        entries = "\n".join(_format_page_entry(i, p) for i, p in enumerate(batch, start=1))
        state = f"{query_header}{notepad_block}\n\nCandidate Pages:\n{entries}"
        questions: dict[str, str] = {}
        id_map: dict[str, UUID] = {}
        for i, p in enumerate(batch, start=1):
            qid = f"page_{i}"
            id_map[qid] = p.id
            if conversational:
                questions[qid] = f"Is the information the agent needs for its next reply on Page {i}?"
            else:
                questions[qid] = f"Is the information needed to answer the user's question on Page {i}?"

        res = await judge.abinary(state, questions)
        mapped = {id_map[qid]: prob for qid, prob in res.probabilities.items()}
        return mapped, res

    batch_outputs = await asyncio.gather(*(run_one(b) for b in batches))
    merged_probs: dict[UUID, float] = {}
    reqs = 0
    toks = 0
    cost = 0.0
    for mapped, res in batch_outputs:
        merged_probs.update(mapped)
        reqs += res.requests_made
        toks += res.input_tokens
        cost += res.cost_usd
    return merged_probs, reqs, toks, cost


async def _screen_headings_for_page(
    judge: DecisionJudge,
    *,
    query_header: str,
    page: DocumentPage,
    headings: Sequence[SectionHeading],
    passages: Sequence[Passage],
    top_headings: int,
    max_passages: int,
) -> tuple[list[Passage], BinaryBatchResult]:
    section_lines = [
        f"Section {i}: {h.heading_path}"
        for i, h in enumerate(headings, start=1)
    ]
    state = (
        f"{query_header}\n\n"
        f"Page: {page.title}\n"
        f"Sections on this page:\n" + "\n".join(section_lines)
    )
    questions = {
        f"sec_{i}": f"Is the information needed on this page under Section {i}?"
        for i in range(1, len(headings) + 1)
    }
    res = await judge.abinary(state, questions)
    heading_scores: dict[UUID, float] = {
        headings[i - 1].id: res.probabilities[f"sec_{i}"]
        for i in range(1, len(headings) + 1)
    }

    ranked_headings = sorted(
        headings,
        key=lambda h: (-heading_scores.get(h.id, 0.0), h.heading_index),
    )
    kept_heading_ids = {h.id for h in ranked_headings[:top_headings]}

    selected = [p for p in passages if p.heading_id in kept_heading_ids]
    if not selected:
        selected = list(passages[:max_passages])
    elif len(selected) > max_passages:
        # Trim lowest-scoring heading passages first, then restore strict document reading order
        selected.sort(
            key=lambda p: (
                -heading_scores.get(p.heading_id, 0.0) if p.heading_id else 0.0,
                p.passage_index,
            )
        )
        selected = selected[:max_passages]

    selected.sort(key=lambda p: p.passage_index)
    return selected, res


async def navigate_corpus(
    db: AsyncSession,
    *,
    namespace: Namespace,
    judge: DecisionJudge,
    query: str | None = None,
    scenario: str | None = None,
    dialogue_turns: Sequence[DialogueTurn] | None = None,
    notepad_page_ids: Sequence[str] | None = None,
    overrides: NavigationOverrides | None = None,
    on_stage: Callable[[str], Awaitable[None]] | None = None,
) -> NavigationOutput:
    """Execute exact Retriever-Free DecisionRAG (Algorithm 1 & Appendix B)."""
    t_start = time.perf_counter()
    conversational = bool(dialogue_turns)
    active_notepad = [str(pid) for pid in (notepad_page_ids or []) if str(pid).strip()]

    # Load categories, clusters, and READY pages in deterministic corpus order
    categories = list(
        (
            await db.execute(
                select(CategoryNode)
                .where(CategoryNode.namespace_id == namespace.id)
                .order_by(CategoryNode.order_index.asc())
            )
        ).scalars().all()
    )
    clusters = list(
        (
            await db.execute(
                select(TopicCluster)
                .where(TopicCluster.namespace_id == namespace.id)
                .order_by(TopicCluster.cluster_index.asc())
            )
        ).scalars().all()
    )
    all_ready_pages = list(
        (
            await db.execute(
                select(DocumentPage)
                .where(
                    DocumentPage.namespace_id == namespace.id,
                    DocumentPage.status == "READY",
                )
                .order_by(DocumentPage.doc_order.asc())
            )
        ).scalars().all()
    )

    requested_profile: ProfileName = (
        overrides.profile
        if (overrides and overrides.profile is not None)
        else namespace.default_profile  # type: ignore[assignment]
    )

    if not all_ready_pages:
        empty_cfg = resolve_profile_config(
            requested_profile,
            has_categories=bool(categories),
            has_clusters=bool(clusters),
            total_ready_pages=0,
            conversational=conversational,
            overrides=overrides,
        )
        return NavigationOutput(
            passages=[],
            telemetry=NavigationTelemetry(
                judge=judge.judge_id,
                profile_used=empty_cfg.profile,
                mc_mode=empty_cfg.mc_mode,
                total_requests=0,
                total_input_tokens=0,
                total_cost_usd=0.0,
                total_latency_ms=(time.perf_counter() - t_start) * 1000.0,
                notepad_page_ids=active_notepad,
                stages=[],
            ),
            updated_notepad_page_ids=active_notepad,
        )

    est_all_page_tokens = sum(
        estimate_tokens(_format_page_entry(i + 1, p)) for i, p in enumerate(all_ready_pages)
    )
    cfg = resolve_profile_config(
        requested_profile,
        has_categories=bool(categories),
        has_clusters=bool(clusters),
        total_ready_pages=len(all_ready_pages),
        est_page_tokens=est_all_page_tokens,
        conversational=conversational,
        overrides=overrides,
    )

    pages_by_id: dict[UUID, DocumentPage] = {p.id: p for p in all_ready_pages}
    pages_by_str_id: dict[str, DocumentPage] = {str(p.id): p for p in all_ready_pages}
    category_order_map: dict[UUID | None, int] = {c.id: c.order_index for c in categories}
    category_order_map[None] = len(categories) + 1

    query_header = _format_query_header(
        query=query,
        scenario=scenario,
        dialogue_turns=dialogue_turns,
        conversational=conversational,
    )
    stages_telemetry: list[StageMetrics] = []
    candidate_pages = list(all_ready_pages)

    # =========================================================================
    # LEVEL 0: Category Routing (App. B.2) OR Topic Cluster Routing (App. B.5)
    # =========================================================================
    if cfg.l0_mode == "categories" and categories:
        if on_stage is not None:
            await on_stage("categories")
        t0 = time.perf_counter()
        unit_name = "Help Center" if conversational else "Category"
        cat_lines = [
            f"{unit_name} {i}: {c.name}" + (f" — {c.description}" if c.description else "")
            for i, c in enumerate(categories, start=1)
        ]
        l0_state = f"{query_header}\n\nAvailable {unit_name}s:\n" + "\n".join(cat_lines)
        l0_questions: dict[str, str] = {}
        for i in range(1, len(categories) + 1):
            if conversational:
                l0_questions[f"cat_{i}"] = (
                    f"Is the information the agent needs for its next reply in Help Center {i}?"
                )
            else:
                l0_questions[f"cat_{i}"] = (
                    f"Is the information needed to answer the query in Category {i}?"
                )

        cat_res = await judge.abinary(l0_state, l0_questions)
        scored_cats = sorted(
            enumerate(categories, start=1),
            key=lambda pair: (-cat_res.probabilities[f"cat_{pair[0]}"], pair[1].order_index),
        )
        kept_cat_ids: set[UUID] = {scored_cats[0][1].id}
        top_cat_prob = cat_res.probabilities[f"cat_{scored_cats[0][0]}"]
        if top_cat_prob < 0.8 and len(scored_cats) >= 2:
            kept_cat_ids.add(scored_cats[1][1].id)

        # App. B.2 Rule 1: retain category of most recent Notepad page
        if active_notepad:
            np_first_page = pages_by_str_id.get(active_notepad[0])
            if np_first_page is not None and np_first_page.category_id is not None:
                kept_cat_ids.add(np_first_page.category_id)

        pruned_pages = [p for p in candidate_pages if p.category_id in kept_cat_ids]
        if pruned_pages:
            candidate_pages = pruned_pages

        stages_telemetry.append(
            StageMetrics(
                stage="L0_categories",
                candidates_in=len(categories),
                candidates_out=len(kept_cat_ids),
                requests_made=cat_res.requests_made,
                input_tokens=cat_res.input_tokens,
                cost_usd=cat_res.cost_usd,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    elif cfg.l0_mode == "clusters" and clusters:
        if on_stage is not None:
            await on_stage("clusters")
        t0 = time.perf_counter()
        # Split clusters only if summaries exceed CLUSTER_MAX_BATCH_TOKENS
        cluster_batches: list[list[TopicCluster]] = []
        cur_c_batch: list[TopicCluster] = []
        cur_c_toks = estimate_tokens(query_header)
        for tc in clusters:
            t_est = estimate_tokens(f"Topic 1: {tc.summary}") + 16
            if cur_c_batch and (cur_c_toks + t_est > CLUSTER_MAX_BATCH_TOKENS):
                cluster_batches.append(cur_c_batch)
                cur_c_batch = []
                cur_c_toks = estimate_tokens(query_header)
            cur_c_batch.append(tc)
            cur_c_toks += t_est
        if cur_c_batch:
            cluster_batches.append(cur_c_batch)

        async def score_cluster_batch(
            batch: Sequence[TopicCluster],
        ) -> tuple[dict[UUID, float], BinaryBatchResult]:
            lines = [f"Topic {i}: {tc.summary}" for i, tc in enumerate(batch, start=1)]
            c_state = f"{query_header}\n\nTopics:\n" + "\n".join(lines)
            c_qs = {
                f"top_{i}": (
                    f"Is the page with the information needed to answer the user's question likely in Topic {i}?"
                )
                for i in range(1, len(batch) + 1)
            }
            res = await judge.abinary(c_state, c_qs)
            mapped = {batch[i - 1].id: res.probabilities[f"top_{i}"] for i in range(1, len(batch) + 1)}
            return mapped, res

        c_outputs = await asyncio.gather(*(score_cluster_batch(b) for b in cluster_batches))
        cluster_probs: dict[UUID, float] = {}
        c_reqs = 0
        c_toks = 0
        c_cost = 0.0
        for mapped, res in c_outputs:
            cluster_probs.update(mapped)
            c_reqs += res.requests_made
            c_toks += res.input_tokens
            c_cost += res.cost_usd

        keep_k_clusters = max(1, round(len(clusters) * cfg.cluster_keep_ratio))
        ranked_clusters = sorted(
            clusters,
            key=lambda tc: (-cluster_probs.get(tc.id, 0.0), tc.cluster_index),
        )
        kept_cluster_ids = {tc.id for tc in ranked_clusters[:keep_k_clusters]}
        pruned_pages = [p for p in candidate_pages if p.cluster_id in kept_cluster_ids]
        if pruned_pages:
            candidate_pages = pruned_pages

        stages_telemetry.append(
            StageMetrics(
                stage="L0_clusters",
                candidates_in=len(clusters),
                candidates_out=len(kept_cluster_ids),
                requests_made=c_reqs,
                input_tokens=c_toks,
                cost_usd=c_cost,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )

    # =========================================================================
    # LEVEL 1: Page Screening & Optional Pooled Top-50 Joint Rescore (Eq. 2)
    # =========================================================================
    if on_stage is not None:
        await on_stage("pages")
    t_l1 = time.perf_counter()
    if cfg.group_pages_by_category:
        ordered_l1_pages = sorted(
            candidate_pages,
            key=lambda p: (category_order_map.get(p.category_id, 999999), p.doc_order),
        )
    else:
        ordered_l1_pages = sorted(candidate_pages, key=lambda p: p.doc_order)

    notepad_titles = [
        pages_by_str_id[pid].title
        for pid in active_notepad
        if pid in pages_by_str_id
    ]
    base_l1_header_tokens = estimate_tokens(query_header) + estimate_tokens(" ".join(notepad_titles))
    l1_batches = _split_page_batches(ordered_l1_pages, cfg, base_l1_header_tokens)

    page_probs, l1_reqs, l1_toks, l1_cost = await _score_page_batches(
        judge,
        l1_batches,
        query_header=query_header,
        notepad_titles=notepad_titles,
        conversational=conversational,
    )

    ranked_l1 = sorted(
        candidate_pages,
        key=lambda p: (-page_probs.get(p.id, 0.0), p.doc_order),
    )

    stages_telemetry.append(
        StageMetrics(
            stage="L1_pages",
            candidates_in=len(candidate_pages),
            candidates_out=min(len(ranked_l1), cfg.two_pass_top_pages or cfg.k_pages),
            requests_made=l1_reqs,
            input_tokens=l1_toks,
            cost_usd=l1_cost,
            latency_ms=(time.perf_counter() - t_l1) * 1000.0,
        )
    )

    # Optional Pass 2: Jointly rescore pooled Top-50 pages in 1 request (App. B.4)
    should_rescore_top50 = (
        cfg.two_pass_top_pages is not None
        and len(l1_batches) > 1
        and (
            cfg.profile != "auto"
            or any(bool(p.headings_summary) for p in candidate_pages)
        )
    )
    if should_rescore_top50 and cfg.two_pass_top_pages is not None:
        if on_stage is not None:
            await on_stage("rescore")
        t_rescore = time.perf_counter()
        pooled_top = ranked_l1[: cfg.two_pass_top_pages]
        # Re-sort pooled top pages into strict corpus doc_order with zero Pass 1 scores
        pooled_in_doc_order = sorted(pooled_top, key=lambda p: p.doc_order)
        rescore_probs, r_reqs, r_toks, r_cost = await _score_page_batches(
            judge,
            [pooled_in_doc_order],
            query_header=query_header,
            notepad_titles=notepad_titles,
            conversational=conversational,
        )
        page_probs.update(rescore_probs)
        ranked_l1 = sorted(
            pooled_top,
            key=lambda p: (-rescore_probs.get(p.id, 0.0), p.doc_order),
        )
        stages_telemetry.append(
            StageMetrics(
                stage="L1_pages_rescore",
                candidates_in=len(pooled_top),
                candidates_out=min(len(ranked_l1), cfg.k_pages),
                requests_made=r_reqs,
                input_tokens=r_toks,
                cost_usd=r_cost,
                latency_ms=(time.perf_counter() - t_rescore) * 1000.0,
            )
        )

    kept_l1_pages = ranked_l1[: cfg.k_pages]
    l1_rank_by_page_id: dict[UUID, int] = {p.id: idx for idx, p in enumerate(kept_l1_pages)}

    # App. B.2 Rule 3: Union retained k_pages with top 2 Notepad pages for Stage 2
    stage2_page_ids: set[UUID] = {p.id for p in kept_l1_pages}
    if active_notepad:
        for pid_str in active_notepad[:2]:
            np_page = pages_by_str_id.get(pid_str)
            if np_page is not None:
                stage2_page_ids.add(np_page.id)

    stage2_pages_in_doc_order = sorted(
        (pages_by_id[pid] for pid in stage2_page_ids),
        key=lambda p: p.doc_order,
    )

    # Load all passages (and headings if needed) for ONLY the Stage 2 pages
    all_stage2_passages = list(
        (
            await db.execute(
                select(Passage)
                .where(Passage.page_id.in_(stage2_page_ids))
                .order_by(Passage.page_id.asc(), Passage.passage_index.asc())
            )
        ).scalars().all()
    )
    passages_by_page: dict[UUID, list[Passage]] = {pid: [] for pid in stage2_page_ids}
    for pas in all_stage2_passages:
        passages_by_page[pas.page_id].append(pas)

    # =========================================================================
    # LEVEL 1.5: Section Heading Screening (>60 Passages -> Top 8 Headings)
    # =========================================================================
    heading_filtered_passages: dict[UUID, list[Passage]] = {}
    if cfg.enable_heading_stage:
        long_pages = [
            p
            for p in stage2_pages_in_doc_order
            if len(passages_by_page.get(p.id, [])) > cfg.heading_passage_threshold
        ]
        if long_pages:
            t_h = time.perf_counter()
            long_page_ids = {p.id for p in long_pages}
            all_headings = list(
                (
                    await db.execute(
                        select(SectionHeading)
                        .where(SectionHeading.page_id.in_(long_page_ids))
                        .order_by(SectionHeading.page_id.asc(), SectionHeading.heading_index.asc())
                    )
                ).scalars().all()
            )
            headings_by_page: dict[UUID, list[SectionHeading]] = {pid: [] for pid in long_page_ids}
            for sh in all_headings:
                headings_by_page[sh.page_id].append(sh)

            screenable_pages = [p for p in long_pages if headings_by_page.get(p.id)]
            if screenable_pages:
                if on_stage is not None:
                    await on_stage("sections")
                h_tasks = [
                    _screen_headings_for_page(
                        judge,
                        query_header=query_header,
                        page=p,
                        headings=headings_by_page[p.id],
                        passages=passages_by_page[p.id],
                        top_headings=cfg.top_headings,
                        max_passages=cfg.max_heading_passages,
                    )
                    for p in screenable_pages
                ]
                h_results = await asyncio.gather(*h_tasks)
                h_in = sum(len(headings_by_page[p.id]) for p in screenable_pages)
                h_out = 0
                h_reqs = 0
                h_toks = 0
                h_cost = 0.0
                for p, (sel_passages, h_res) in zip(screenable_pages, h_results, strict=True):
                    heading_filtered_passages[p.id] = sel_passages
                    h_out += min(len(headings_by_page[p.id]), cfg.top_headings)
                    h_reqs += h_res.requests_made
                    h_toks += h_res.input_tokens
                    h_cost += h_res.cost_usd

                stages_telemetry.append(
                    StageMetrics(
                        stage="L1_5_headings",
                        candidates_in=h_in,
                        candidates_out=h_out,
                        requests_made=h_reqs,
                        input_tokens=h_toks,
                        cost_usd=h_cost,
                        latency_ms=(time.perf_counter() - t_h) * 1000.0,
                    )
                )

    # =========================================================================
    # STAGE 2: Shared-Context Passage Binary Scoring (Eq. 2, App. B.1 & B.5)
    # =========================================================================
    if on_stage is not None:
        await on_stage("passages")
    t_s2 = time.perf_counter()
    candidate_pool: list[_CandidatePassage] = []

    for page in stage2_pages_in_doc_order:
        page_passages = passages_by_page.get(page.id, [])
        if page.id in heading_filtered_passages:
            chosen_passages = heading_filtered_passages[page.id]
        elif cfg.tiered_passage_pool:
            # App. B.5 NQ Tiered Pool: all of ranks 1-3, first 10 of 4-10, first 3 of 11-30
            l1_rank = l1_rank_by_page_id.get(page.id, 0)
            if l1_rank < 3:
                chosen_passages = page_passages
            elif l1_rank < 10:
                chosen_passages = page_passages[:10]
            else:
                chosen_passages = page_passages[:3]
        else:
            chosen_passages = page_passages

        p_page_val = page_probs.get(page.id)
        for pas in chosen_passages:
            candidate_pool.append(
                _CandidatePassage(
                    passage=pas,
                    page=page,
                    p_page_bin=p_page_val,
                )
            )

    # Guarantee strict (page.doc_order, passage.passage_index) reading order
    candidate_pool.sort(key=lambda c: (c.page.doc_order, c.passage.passage_index))

    # Split Stage 2 along whole-page boundaries ONLY if pool exceeds STAGE2_MAX_BATCH_TOKENS
    s2_batches: list[list[_CandidatePassage]] = []
    cur_s2: list[_CandidatePassage] = []
    cur_s2_tokens = estimate_tokens(query_header)
    for cand in candidate_pool:
        cand_toks = cand.passage.token_estimate + 12
        same_page_as_prev = bool(cur_s2 and cur_s2[-1].page.id == cand.page.id)
        if cur_s2 and not same_page_as_prev and (cur_s2_tokens + cand_toks > STAGE2_MAX_BATCH_TOKENS):
            s2_batches.append(cur_s2)
            cur_s2 = []
            cur_s2_tokens = estimate_tokens(query_header)
        cur_s2.append(cand)
        cur_s2_tokens += cand_toks
    if cur_s2:
        s2_batches.append(cur_s2)

    s2_unit = "Passage" if conversational else "Paragraph"

    async def score_s2_batch(batch: Sequence[_CandidatePassage]) -> BinaryBatchResult:
        grouped_text = _render_grouped_passages(
            [(c.page, c.passage) for c in batch],
            unit_label=s2_unit,
        )
        s2_state = f"{query_header}\n\nCandidate {s2_unit}s:\n{grouped_text}"
        s2_questions: dict[str, str] = {}
        for i in range(1, len(batch) + 1):
            if conversational:
                s2_questions[f"p_{i}"] = f"Should the agent's next reply be based on Passage {i}?"
            else:
                s2_questions[f"p_{i}"] = (
                    f"Does Paragraph {i} contain information needed to answer the user's question?"
                )
        return await judge.abinary(s2_state, s2_questions)

    s2_results = await asyncio.gather(*(score_s2_batch(b) for b in s2_batches))
    s2_reqs = 0
    s2_toks = 0
    s2_cost = 0.0
    for batch, res in zip(s2_batches, s2_results, strict=True):
        s2_reqs += res.requests_made
        s2_toks += res.input_tokens
        s2_cost += res.cost_usd
        for i, cand in enumerate(batch, start=1):
            p_para = res.probabilities[f"p_{i}"]
            cand.p_para_bin = p_para
            if cfg.multiply_page_prob:
                p_pg = cand.p_page_bin if cand.p_page_bin is not None else 1.0
                cand.stage2_score = p_para * p_pg
            else:
                cand.stage2_score = p_para

    ranked_s2 = sorted(
        candidate_pool,
        key=lambda c: (
            -c.stage2_score,
            -c.p_para_bin,
            c.page.doc_order,
            c.passage.passage_index,
        ),
    )

    if cfg.deduplicate:
        deduped: list[_CandidatePassage] = []
        seen_hashes: set[str] = set()
        for cand in ranked_s2:
            if cand.passage.norm_hash in seen_hashes:
                continue
            seen_hashes.add(cand.passage.norm_hash)
            deduped.append(cand)
        ranked_s2 = deduped

    retained_r = ranked_s2[: cfg.r_passages]
    stages_telemetry.append(
        StageMetrics(
            stage="S2_passages",
            candidates_in=len(candidate_pool),
            candidates_out=len(retained_r),
            requests_made=s2_reqs,
            input_tokens=s2_toks,
            cost_usd=s2_cost,
            latency_ms=(time.perf_counter() - t_s2) * 1000.0,
        )
    )

    # =========================================================================
    # STAGE 3: Isolated Reading-Order Multiple-Choice Calibration (Eq. 3, App. B.6)
    # =========================================================================
    m_actual = min(cfg.m_shortlist, len(retained_r), 255)
    if cfg.mc_mode != "off" and m_actual >= 2:
        if on_stage is not None:
            await on_stage("ranking")
        t_s3 = time.perf_counter()
        shortlist = list(retained_r[:m_actual])
        tail = list(retained_r[m_actual:])

        # Re-sort shortlist strictly back into (page.doc_order, passage.passage_index) reading order
        shortlist_reading_order = sorted(
            shortlist,
            key=lambda c: (c.page.doc_order, c.passage.passage_index),
        )
        mc_grouped_text = _render_grouped_passages(
            [(c.page, c.passage) for c in shortlist_reading_order],
            unit_label="Passage",
        )
        mc_state = f"{query_header}\n\nCandidate Passages:\n{mc_grouped_text}"
        if conversational:
            mc_instruction = "Which passage should the agent's next reply be based on?"
        else:
            mc_instruction = (
                "Which passage most directly contains the information needed to answer the user's question?"
            )

        mc_options = {
            f"opt_{i}": f"Passage {i} (from {c.page.title})"
            for i, c in enumerate(shortlist_reading_order, start=1)
        }
        mc_res = await judge.achoice(mc_state, mc_instruction, mc_options)

        for i, cand in enumerate(shortlist_reading_order, start=1):
            cand.p_mul = mc_res.probabilities[f"opt_{i}"]

        if cfg.mc_mode == "order":
            reordered_head = sorted(
                shortlist,
                key=lambda c: (
                    -(c.p_mul if c.p_mul is not None else 0.0),
                    -c.stage2_score,
                    c.page.doc_order,
                    c.passage.passage_index,
                ),
            )
            final_candidates = reordered_head + tail
        else:  # mc_mode == "top1"
            winner_cand = max(
                shortlist,
                key=lambda c: (
                    c.p_mul if c.p_mul is not None else -1.0,
                    c.stage2_score,
                    -c.page.doc_order,
                    -c.passage.passage_index,
                ),
            )
            # Move winner to Rank #1; preserve exact Stage 2 relative order for all others
            final_candidates = [winner_cand] + [
                c for c in retained_r if c.passage.id != winner_cand.passage.id
            ]

        stages_telemetry.append(
            StageMetrics(
                stage="S3_mc",
                candidates_in=m_actual,
                candidates_out=m_actual,
                requests_made=mc_res.requests_made,
                input_tokens=mc_res.input_tokens,
                cost_usd=mc_res.cost_usd,
                latency_ms=(time.perf_counter() - t_s3) * 1000.0,
            )
        )
    else:
        final_candidates = retained_r

    # App. B.2 Rule 4: Move Rank #1's parent page_id to front of Notepad
    if final_candidates:
        top1_page_id_str = str(final_candidates[0].page.id)
        updated_notepad = [top1_page_id_str] + [
            pid for pid in active_notepad if pid != top1_page_id_str
        ]
    else:
        updated_notepad = list(active_notepad)

    ranked_items = [
        RankedPassageItem(
            rank=idx,
            passage_id=c.passage.id,
            page_id=c.page.id,
            page_title=c.page.title,
            heading_path=c.passage.heading_path,
            passage_index=c.passage.passage_index,
            content=c.passage.content,
            p_page_bin=c.p_page_bin,
            p_para_bin=c.p_para_bin,
            stage2_score=c.stage2_score,
            p_mul=c.p_mul,
        )
        for idx, c in enumerate(final_candidates, start=1)
    ]

    total_reqs = sum(s.requests_made for s in stages_telemetry)
    total_toks = sum(s.input_tokens for s in stages_telemetry)
    total_cost = sum(s.cost_usd for s in stages_telemetry)
    total_ms = (time.perf_counter() - t_start) * 1000.0

    telemetry = NavigationTelemetry(
        judge=judge.judge_id,
        profile_used=cfg.profile,
        mc_mode=cfg.mc_mode,
        total_requests=total_reqs,
        total_input_tokens=total_toks,
        total_cost_usd=total_cost,
        total_latency_ms=total_ms,
        notepad_page_ids=updated_notepad,
        stages=stages_telemetry,
    )

    if on_stage is not None:
        await on_stage("ranked")

    return NavigationOutput(
        passages=ranked_items,
        telemetry=telemetry,
        updated_notepad_page_ids=updated_notepad,
    )