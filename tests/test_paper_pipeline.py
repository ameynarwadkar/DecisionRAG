from collections.abc import Callable, Mapping
import hashlib
import re
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.models import (
    CategoryNode,
    DocumentPage,
    Namespace,
    Passage,
    SectionHeading,
    TopicCluster,
)
from dec_rag.schemas import DialogueTurn, NavigationOverrides
from dec_rag.services.ingestion import ingest_document
from dec_rag.services.judges.base import BinaryBatchResult, ChoiceResult, DecisionJudge
from dec_rag.services.paper_pipeline import navigate_corpus


class ScriptedDecisionJudge(DecisionJudge):
    """Deterministic DecisionJudge recorder for testing exact Algorithm 1 invariants."""

    def __init__(
        self,
        *,
        binary_fn: Callable[[str, Mapping[str, str]], dict[str, float]],
        choice_fn: Callable[[str, str, Mapping[str, str]], dict[str, float]] | None = None,
    ) -> None:
        super().__init__(
            judge_id="openrouter_jev",
            model="typesafe/jev-1.13",
            endpoint="https://mock.local/decisions",
            api_key="mock-key",
        )
        self._binary_fn = binary_fn
        self._choice_fn = choice_fn
        self.binary_calls: list[tuple[str, dict[str, str]]] = []
        self.choice_calls: list[tuple[str, str, dict[str, str]]] = []

    async def abinary(
        self,
        state: str | dict[str, Any],
        questions: Mapping[str, str],
        *,
        yes_desc: str = "Yes, relevant.",
        no_desc: str = "No, not relevant.",
    ) -> BinaryBatchResult:
        del yes_desc, no_desc
        s_str = str(state)
        q_dict = dict(questions)
        self.binary_calls.append((s_str, q_dict))
        probs = self._binary_fn(s_str, q_dict)
        return BinaryBatchResult(probabilities=probs, requests_made=1, input_tokens=50, cost_usd=0.001)

    async def achoice(
        self,
        state: str | dict[str, Any],
        instruction: str,
        options: Mapping[str, str],
    ) -> ChoiceResult:
        s_str = str(state)
        opt_dict = dict(options)
        self.choice_calls.append((s_str, instruction, opt_dict))
        if self._choice_fn is not None:
            raw = self._choice_fn(s_str, instruction, opt_dict)
        else:
            raw = {k: 1.0 / len(opt_dict) for k in opt_dict}
        total = sum(raw.values())
        norm = {k: v / total for k, v in raw.items()}
        winner = max(norm, key=lambda k: norm[k])
        return ChoiceResult(
            probabilities=norm,
            winner=winner,
            requests_made=1,
            input_tokens=30,
            cost_usd=0.001,
        )


async def test_irreversible_pruning_and_no_leaked_scores(db_session: AsyncSession) -> None:
    ns = Namespace(name="CQA Bench", slug="cqa-bench", default_profile="cqa")
    db_session.add(ns)
    await db_session.commit()

    # 4 pages; cqa keeps k_pages=3, so Page 4 must be irreversibly pruned
    for i in range(4):
        await ingest_document(
            db_session,
            namespace_id=ns.id,
            filename=f"doc_{i}.md",
            raw_bytes=f"# Doc {i}\n\nUniqueBodyToken_{i} explains policy {i}.".encode("utf-8"),
        )

    page_scores = [0.9123, 0.7654, 0.5432, 0.0111]

    def binary_fn(state: str, questions: Mapping[str, str]) -> dict[str, float]:
        first_q = next(iter(questions.values()))
        if "on Page" in first_q:
            return {f"page_{i + 1}": page_scores[i] for i in range(len(questions))}
        return {qid: 0.6 for qid in questions}

    judge = ScriptedDecisionJudge(binary_fn=binary_fn)
    out = await navigate_corpus(db_session, namespace=ns, judge=judge, query="Policy question?")

    assert len(out.passages) == 3
    s2_state, _ = judge.binary_calls[1]
    assert "UniqueBodyToken_3" not in s2_state  # Pruned page's passage never queried
    assert "0.9123" not in s2_state  # No leaked L1 probabilities
    mc_state, _, _ = judge.choice_calls[0]
    assert "0.6" not in mc_state  # No leaked Stage 2 probabilities


async def test_category_runner_up_below_08_and_notepad_center_retention(
    db_session: AsyncSession,
) -> None:
    ns = Namespace(name="MD2D Help", slug="md2d-help", default_profile="md2d")
    db_session.add(ns)
    await db_session.flush()

    cats = [
        CategoryNode(namespace_id=ns.id, name=f"Center {i}", slug=f"center-{i}", order_index=i)
        for i in range(4)
    ]
    db_session.add_all(cats)
    await db_session.flush()

    pages: list[DocumentPage] = []
    for i, cat in enumerate(cats):
        p = DocumentPage(
            namespace_id=ns.id,
            category_id=cat.id,
            doc_order=i,
            filename=f"c{i}.md",
            title=f"Guide_{i}",
            first_sentence=f"Opening {i}.",
            headings_summary=[],
            passage_count=1,
            status="READY",
        )
        db_session.add(p)
        await db_session.flush()
        db_session.add(
            Passage(
                page_id=p.id,
                namespace_id=ns.id,
                passage_index=0,
                heading_path="",
                content=f"Content from Guide_{i}.",
                norm_hash=hashlib.sha256(f"c_{i}".encode()).hexdigest(),
                token_estimate=6,
            )
        )
        pages.append(p)
    await db_session.commit()

    # Center 0 gets 0.72 (< 0.8 -> runner-up Center 1 @ 0.55 ALSO kept),
    # Center 2 gets 0.10 (pruned!), Center 3 gets 0.05 (kept ONLY because it's Notepad[0]'s category!)
    cat_probs = {"cat_1": 0.72, "cat_2": 0.55, "cat_3": 0.10, "cat_4": 0.05}

    def binary_fn(state: str, questions: Mapping[str, str]) -> dict[str, float]:
        del state
        if "cat_1" in questions:
            return cat_probs
        return {qid: 0.5 for qid in questions}

    judge = ScriptedDecisionJudge(binary_fn=binary_fn)
    out = await navigate_corpus(
        db_session,
        namespace=ns,
        judge=judge,
        dialogue_turns=[DialogueTurn(role="user", content="Where is my package?")],
        notepad_page_ids=[str(pages[3].id)],
    )

    # Verify verbatim conversational Help Center question
    _, l0_qs = judge.binary_calls[0]
    assert l0_qs["cat_1"] == "Is the information the agent needs for its next reply in Help Center 1?"

    # L1 must receive Guide_0, Guide_1 (runner-up), and Guide_3 (Notepad center), but NOT Guide_2
    l1_state, _ = judge.binary_calls[1]
    assert "Guide_0" in l1_state
    assert "Guide_1" in l1_state
    assert "Guide_3" in l1_state
    assert "Guide_2" not in l1_state
    assert len(out.passages) == 3


async def test_mtrag_multibatch_top50_rescore_and_heading_screening(
    db_session: AsyncSession,
) -> None:
    ns = Namespace(name="MTRAG Corpus", slug="mtrag-corpus", default_profile="mtrag")
    db_session.add(ns)
    await db_session.flush()

    # Create 55 pages with substantial openings/headings so 10k-token budget splits L1 into >=2 batches,
    # and give Page 0 65 passages across 10 headings (>60 threshold) to trigger L1.5!
    filler = "Detailed technical documentation section heading description " * 25
    for i in range(55):
        n_passages = 65 if i == 0 else 1
        p = DocumentPage(
            namespace_id=ns.id,
            doc_order=i,
            filename=f"m_{i}.md",
            title=f"Manual_{i}",
            first_sentence=f"Manual {i} {filler}",
            headings_summary=[f"Sec {s} {filler[:80]}" for s in range(6)],
            passage_count=n_passages,
            status="READY",
        )
        db_session.add(p)
        await db_session.flush()

        if i == 0:
            headings: list[SectionHeading] = []
            for h_idx in range(10):
                sh = SectionHeading(
                    page_id=p.id,
                    namespace_id=ns.id,
                    heading_index=h_idx,
                    level=1,
                    title=f"Chapter {h_idx}",
                    heading_path=f"Chapter {h_idx}",
                )
                db_session.add(sh)
                headings.append(sh)
            await db_session.flush()

            for p_idx in range(65):
                h_obj = headings[p_idx % 10]
                db_session.add(
                    Passage(
                        page_id=p.id,
                        namespace_id=ns.id,
                        heading_id=h_obj.id,
                        passage_index=p_idx,
                        heading_path=h_obj.heading_path,
                        content=f"Manual_0 paragraph {p_idx} under {h_obj.title}.",
                        norm_hash=hashlib.sha256(f"m0_{p_idx}".encode()).hexdigest(),
                        token_estimate=12,
                    )
                )
        else:
            db_session.add(
                Passage(
                    page_id=p.id,
                    namespace_id=ns.id,
                    passage_index=0,
                    heading_path="Intro",
                    content=f"Manual_{i} single paragraph.",
                    norm_hash=hashlib.sha256(f"m_{i}".encode()).hexdigest(),
                    token_estimate=8,
                )
            )
    await db_session.commit()

    def binary_fn(state: str, questions: Mapping[str, str]) -> dict[str, float]:
        first_q = next(iter(questions.values()))
        if "on Page" in first_q:
            # Give Manual_0 highest score wherever it appears
            probs: dict[str, float] = {}
            for i in range(1, len(questions) + 1):
                probs[f"page_{i}"] = 0.99 if f"Page {i}: Manual_0\n" in state else 0.50 - (i * 0.001)
            return probs
        if "under Section" in first_q:
            # Score Chapters 0..7 high (0.9) and Chapters 8..9 low (0.01)
            return {f"sec_{i}": (0.9 if i <= 8 else 0.01) for i in range(1, len(questions) + 1)}
        return {qid: 0.7 for qid in questions}

    judge = ScriptedDecisionJudge(binary_fn=binary_fn)
    out = await navigate_corpus(db_session, namespace=ns, judge=judge, query="Find Manual_0 detail")

    stage_names = [s.stage for s in out.telemetry.stages]
    assert "L1_pages" in stage_names
    assert "L1_pages_rescore" in stage_names
    assert "L1_5_headings" in stage_names

    rescore_stage = next(s for s in out.telemetry.stages if s.stage == "L1_pages_rescore")
    assert rescore_stage.candidates_in == 50
    assert rescore_stage.requests_made == 1

    heading_stage = next(s for s in out.telemetry.stages if s.stage == "L1_5_headings")
    assert heading_stage.candidates_in == 10
    assert heading_stage.candidates_out == 8

    # Chapters 8 and 9 must have been pruned from Stage 2!
    s2_state = next(st for st, qs in judge.binary_calls if "p_1" in qs)
    assert "Chapter 8" not in s2_state
    assert "Chapter 9" not in s2_state


async def test_nq_cluster_15pct_30page_tiered_selection_and_prob_product(
    db_session: AsyncSession,
) -> None:
    ns = Namespace(name="NQ Corpus", slug="nq-corpus", default_profile="nq")
    db_session.add(ns)
    await db_session.flush()

    # 20 clusters -> 15% keeps exact 3 clusters; put 12 pages in each of the 3 kept clusters (=36 pages)
    clusters = [
        TopicCluster(namespace_id=ns.id, cluster_index=i, summary=f"Topic group {i}.", page_count=12 if i < 3 else 1)
        for i in range(20)
    ]
    db_session.add_all(clusters)
    await db_session.flush()

    doc_order = 0
    for c_idx, tc in enumerate(clusters):
        n_pages = 12 if c_idx < 3 else 1
        for p_local in range(n_pages):
            p = DocumentPage(
                namespace_id=ns.id,
                cluster_id=tc.id,
                cluster_rank=p_local,
                doc_order=doc_order,
                filename=f"w_{doc_order}.md",
                title=f"WikiPage_{doc_order}",
                first_sentence=f"Wiki opening {doc_order}.",
                headings_summary=[],
                passage_count=12,
                status="READY",
            )
            db_session.add(p)
            await db_session.flush()
            for pas_idx in range(12):
                db_session.add(
                    Passage(
                        page_id=p.id,
                        namespace_id=ns.id,
                        passage_index=pas_idx,
                        heading_path="",
                        content=f"WikiPage_{doc_order} para {pas_idx}.",
                        norm_hash=hashlib.sha256(f"w_{doc_order}_{pas_idx}".encode()).hexdigest(),
                        token_estimate=8,
                    )
                )
            doc_order += 1
    await db_session.commit()

    def binary_fn(state: str, questions: Mapping[str, str]) -> dict[str, float]:
        del state
        first_key = next(iter(questions.keys()))
        if first_key.startswith("top_"):
            return {f"top_{i}": (0.95 if i <= 3 else 0.05) for i in range(1, len(questions) + 1)}
        if first_key.startswith("page_"):
            # Decreasing page probabilities from 0.90 downward
            return {
                f"page_{i}": round(0.90 - (i - 1) * 0.01, 4)
                for i in range(1, len(questions) + 1)
            }
        # Stage 2 paragraphs: uniform 0.5 so stage2_score = 0.5 * p_page
        return {qid: 0.5 for qid in questions}

    judge = ScriptedDecisionJudge(binary_fn=binary_fn)
    out = await navigate_corpus(db_session, namespace=ns, judge=judge, query="Who invented it?")

    l0_stage = next(s for s in out.telemetry.stages if s.stage == "L0_clusters")
    assert l0_stage.candidates_in == 20
    assert l0_stage.candidates_out == 3  # 15% of 20

    s2_stage = next(s for s in out.telemetry.stages if s.stage == "S2_passages")
    # Tiered selection across 30 kept pages (each has 12 passages):
    # ranks 1..3 -> 3*12 = 36; ranks 4..10 -> 7*10 = 70; ranks 11..30 -> 20*3 = 60 => 166 total!
    assert s2_stage.candidates_in == 166

    # Verify p_para * p_page multiplication (0.5 * 0.90 = 0.45 on top page)
    assert out.passages[0].p_page_bin == pytest.approx(0.90)
    assert out.passages[0].p_para_bin == pytest.approx(0.50)
    assert out.passages[0].stage2_score == pytest.approx(0.45)

    # NQ shortlist m=5
    s3_stage = next(s for s in out.telemetry.stages if s.stage == "S3_mc")
    assert s3_stage.candidates_in == 5


async def test_stage3_isolated_reading_order_context_and_mc_modes(
    db_session: AsyncSession,
) -> None:
    ns = Namespace(name="MC Modes", slug="mc-modes", default_profile="cqa")
    db_session.add(ns)
    await db_session.commit()

    # 3 pages (k_pages=3), 1 passage each; plus a 2nd passage on Page 2 so r=4, m=3 leaves 1 tail passage
    await ingest_document(db_session, namespace_id=ns.id, filename="p0.md", raw_bytes=b"# DocZero\n\nBody_Zero.")
    await ingest_document(db_session, namespace_id=ns.id, filename="p1.md", raw_bytes=b"# DocOne\n\nBody_One.")
    await ingest_document(
        db_session,
        namespace_id=ns.id,
        filename="p2.md",
        raw_bytes=b"# DocTwo\n\nBody_TwoA.\n\nBody_TwoB_Tail.",
    )

    # Stage 2 ranks: Body_TwoA (0.95) > Body_Zero (0.85) > Body_One (0.75) > Body_TwoB_Tail (0.20)
    def binary_fn(state: str, questions: Mapping[str, str]) -> dict[str, float]:
        del state
        if "page_1" in questions:
            return {qid: 0.9 for qid in questions}
        return {"p_1": 0.85, "p_2": 0.75, "p_3": 0.95, "p_4": 0.20}

    # In Stage 3 reading order of top m=3 is: opt_1=Body_Zero, opt_2=Body_One, opt_3=Body_TwoA.
    # Make opt_2 (Body_One) the MC winner, opt_1 (Body_Zero) #2, opt_3 (Body_TwoA) #3!
    def choice_fn(state: str, instruction: str, options: Mapping[str, str]) -> dict[str, float]:
        del instruction, options
        assert "Body_TwoB_Tail" not in state  # Outside shortlist m=3 -> MUST NOT appear in Stage 3!
        positions = [
            state.index("Body_Zero"),
            state.index("Body_One"),
            state.index("Body_TwoA"),
        ]
        assert positions == sorted(positions)  # Strictly re-sorted into reading order!
        return {"opt_1": 0.30, "opt_2": 0.60, "opt_3": 0.10}

    # 1. mc_mode == "order": top 3 sorted by p_mul -> [Body_One, Body_Zero, Body_TwoA], tail stays #4
    judge_order = ScriptedDecisionJudge(binary_fn=binary_fn, choice_fn=choice_fn)
    out_order = await navigate_corpus(
        db_session,
        namespace=ns,
        judge=judge_order,
        query="Test MC",
        overrides=NavigationOverrides(r_passages=4, m_shortlist=3, mc_mode="order"),
    )
    assert [p.content for p in out_order.passages] == [
        "Body_One.",
        "Body_Zero.",
        "Body_TwoA.",
        "Body_TwoB_Tail.",
    ]

    # 2. mc_mode == "top1": winner Body_One moves to #1, others keep exact Stage 2 order [Body_TwoA, Body_Zero, Body_TwoB_Tail]
    judge_top1 = ScriptedDecisionJudge(binary_fn=binary_fn, choice_fn=choice_fn)
    out_top1 = await navigate_corpus(
        db_session,
        namespace=ns,
        judge=judge_top1,
        query="Test MC",
        overrides=NavigationOverrides(r_passages=4, m_shortlist=3, mc_mode="top1"),
    )
    assert [p.content for p in out_top1.passages] == [
        "Body_One.",
        "Body_TwoA.",
        "Body_Zero.",
        "Body_TwoB_Tail.",
    ]

    # 3. mc_mode == "off": zero choice calls, exact Stage 2 order preserved
    judge_off = ScriptedDecisionJudge(binary_fn=binary_fn, choice_fn=choice_fn)
    out_off = await navigate_corpus(
        db_session,
        namespace=ns,
        judge=judge_off,
        query="Test MC",
        overrides=NavigationOverrides(r_passages=4, m_shortlist=3, mc_mode="off"),
    )
    assert len(judge_off.choice_calls) == 0
    assert all(p.p_mul is None for p in out_off.passages)
    assert [p.content for p in out_off.passages] == [
        "Body_TwoA.",
        "Body_Zero.",
        "Body_One.",
        "Body_TwoB_Tail.",
    ]


async def test_conversational_notepad_s1_list_s2_top2_union_and_front_update(
    db_session: AsyncSession,
) -> None:
    ns = Namespace(name="Conv Notepad", slug="conv-notepad", default_profile="cqa")
    db_session.add(ns)
    await db_session.commit()

    created_pages: list[DocumentPage] = []
    for i in range(5):
        p, _ = await ingest_document(
            db_session,
            namespace_id=ns.id,
            filename=f"np_{i}.md",
            raw_bytes=f"# PageTitle_{i}\n\nPassageFromPage_{i}.".encode("utf-8"),
        )
        created_pages.append(p)

    # Initial Notepad has [Page 3, Page 4, Page 2].
    # Keep k_pages=1 (Page 0 wins L1). Stage 2 must union Page 0 with Notepad[:2] = [Page 3, Page 4] (NOT Page 2)!
    initial_notepad = [
        str(created_pages[3].id),
        str(created_pages[4].id),
        str(created_pages[2].id),
    ]

    def binary_fn(state: str, questions: Mapping[str, str]) -> dict[str, float]:
        if "page_1" in questions:
            return {"page_1": 0.99, "page_2": 0.10, "page_3": 0.05, "page_4": 0.04, "page_5": 0.03}
        # In Stage 2, candidate pages in doc_order are [Page 0, Page 3, Page 4] -> p_1, p_2, p_3.
        # Make p_3 (Page 4) win Rank #1 so Page 4 moves to front of Notepad!
        assert re.search(r"PassageFromPage_0", state)
        assert re.search(r"PassageFromPage_3", state)
        assert re.search(r"PassageFromPage_4", state)
        assert not re.search(r"PassageFromPage_2", state)
        return {"p_1": 0.30, "p_2": 0.40, "p_3": 0.95}

    judge = ScriptedDecisionJudge(binary_fn=binary_fn)
    turns = [
        DialogueTurn(role="user" if i % 2 == 0 else "assistant", content=f"TurnMsg_{i}")
        for i in range(8)
    ]

    out = await navigate_corpus(
        db_session,
        namespace=ns,
        judge=judge,
        dialogue_turns=turns,
        notepad_page_ids=initial_notepad,
        overrides=NavigationOverrides(k_pages=1, mc_mode="off"),
    )

    l1_state, _ = judge.binary_calls[0]
    assert "TurnMsg_0" not in l1_state  # Capped to 6 most recent turns
    assert "TurnMsg_7" in l1_state
    assert "Recent Notepad Pages" in l1_state
    assert "PageTitle_3" in l1_state

    # Page 4 won Rank #1 -> moved to index 0 of Notepad without duplicates!
    assert out.updated_notepad_page_ids == [
        str(created_pages[4].id),
        str(created_pages[3].id),
        str(created_pages[2].id),
    ]