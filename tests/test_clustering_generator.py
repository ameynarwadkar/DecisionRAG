from typing import Any
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.config import Settings
from dec_rag.models import DocumentPage, Namespace
from dec_rag.schemas import DialogueTurn, RankedPassageItem
from dec_rag.services.clustering import (
    VERBATIM_CLUSTER_SUMMARY_PROMPT,
    build_page_clusters,
)
from dec_rag.services.generator import generate_grounded_answer
from dec_rag.services.ingestion import ingest_document


async def test_offline_clustering_centroid_order_and_verbatim_prompt(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns = Namespace(name="Wiki Flat", slug="wiki-flat")
    db_session.add(ns)
    await db_session.commit()

    docs = [
        ("quantum_mechanics.md", "# Quantum Mechanics\n\nWavefunctions govern subatomic states.\n\nSecretBodyAlpha."),
        ("quantum_entanglement.md", "# Quantum Entanglement\n\nCorrelated particles share quantum states.\n\nSecretBodyBeta."),
        ("italian_pasta.md", "# Italian Pasta\n\nDurum wheat semolina shapes regional pasta.\n\nSecretBodyGamma."),
        ("neapolitan_pizza.md", "# Neapolitan Pizza\n\nWood-fired dough topped with San Marzano tomatoes.\n\nSecretBodyDelta."),
    ]
    for fname, text in docs:
        await ingest_document(
            db_session,
            namespace_id=ns.id,
            filename=fname,
            raw_bytes=text.encode("utf-8"),
        )

    cfg = Settings(_env_file=None)

    # Guardrail: <=650 pages without force or num_clusters raises ValueError
    with pytest.raises(ValueError, match="<= 650"):
        await build_page_clusters(db_session, namespace_id=ns.id, settings=cfg)

    embedded_inputs: list[str] = []
    summary_prompts: list[str] = []

    async def fake_aembedding(*, model: str, input: list[str], **kwargs: Any) -> dict[str, Any]:
        del model, kwargs
        embedded_inputs.extend(input)
        vecs = []
        for text in input:
            if "Quantum" in text:
                # Quantum Mechanics closer to [1, 0] than Quantum Entanglement
                v = [1.0, 0.05] if "Mechanics" in text else [0.92, 0.39]
            else:
                v = [0.05, 1.0] if "Pasta" in text else [0.35, 0.93]
            vecs.append({"embedding": v})
        return {"data": vecs}

    async def fake_acompletion(*, model: str, messages: list[dict[str, str]], temperature: float, **kwargs: Any) -> dict[str, Any]:
        del model, kwargs
        assert temperature == 0.0
        prompt = messages[0]["content"]
        summary_prompts.append(prompt)
        # Return >30 words to verify the 30-word cap
        overlong = " ".join([f"topicword{i}" for i in range(36)])
        return {"choices": [{"message": {"content": overlong}}]}

    monkeypatch.setattr("dec_rag.services.clustering.litellm.aembedding", fake_aembedding)
    monkeypatch.setattr("dec_rag.services.clustering.litellm.acompletion", fake_acompletion)

    clusters = await build_page_clusters(
        db_session,
        namespace_id=ns.id,
        settings=cfg,
        num_clusters=2,
    )
    assert len(clusters) == 2

    # Strictly embeds ONLY title + first_sentence, never passage bodies
    for emb_text in embedded_inputs:
        assert "SecretBody" not in emb_text

    # Verbatim paper prompt used & summary capped to <= 30 words
    for sp in summary_prompts:
        assert VERBATIM_CLUSTER_SUMMARY_PROMPT in sp
    for tc in clusters:
        assert tc.page_count == 2
        assert len(tc.summary.split()) <= 30

    pages = list(
        (
            await db_session.execute(
                select(DocumentPage).where(DocumentPage.namespace_id == ns.id)
            )
        ).scalars().all()
    )
    assert all(p.cluster_id is not None and p.cluster_rank in {0, 1} for p in pages)


async def test_stage4_generator_top5_numbering_6turn_window_and_greedy_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(*, model: str, messages: list[dict[str, str]], temperature: float, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        captured["model"] = model
        captured["messages"] = messages
        captured["temperature"] = temperature
        return {"choices": [{"message": {"content": "Refunds take 5 business days [1]."}}]}

    monkeypatch.setattr("dec_rag.services.generator.litellm.acompletion", fake_acompletion)

    passages = [
        RankedPassageItem(
            rank=i + 1,
            passage_id=uuid.uuid4(),
            page_id=uuid.uuid4(),
            page_title=f"Guide {i + 1}",
            heading_path="Section A",
            passage_index=i,
            content=f"Passage body {i + 1}.",
            p_page_bin=0.9,
            p_para_bin=0.8,
            stage2_score=0.8,
            p_mul=0.5,
        )
        for i in range(7)
    ]
    turns = [
        DialogueTurn(role="user" if i % 2 == 0 else "assistant", content=f"Turn_{i}")
        for i in range(8)
    ]

    cfg = Settings(_env_file=None, GENERATOR_MODEL="openai/gpt-4o-mini")
    res = await generate_grounded_answer(
        passages=passages,
        settings=cfg,
        dialogue_turns=turns,
        top_k_context=5,
    )

    assert res.answer == "Refunds take 5 business days [1]."
    assert captured["temperature"] == 0.0
    user_msg = captured["messages"][1]["content"]

    # Top 5 numbered [1]..[5], while [6] and [7] are excluded
    assert "[1] Page: Guide 1 | Section: Section A" in user_msg
    assert "[5] Page: Guide 5 | Section: Section A" in user_msg
    assert "Guide 6" not in user_msg

    # Only the 6 most recent turns (Turn_2..Turn_7) included
    assert "Turn_0" not in user_msg
    assert "Turn_1" not in user_msg
    assert "Turn_2" in user_msg
    assert "Turn_7" in user_msg


async def test_generator_graceful_fallback_when_llm_credentials_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_creds_acompletion(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(
            "OpenAIException - Missing credentials. Please pass an `api_key` or set `OPENAI_API_KEY`."
        )

    monkeypatch.setattr("dec_rag.services.generator.litellm.acompletion", missing_creds_acompletion)

    top_passage = RankedPassageItem(
        rank=1,
        passage_id=uuid.uuid4(),
        page_id=uuid.uuid4(),
        page_title="Refund Guide",
        heading_path="Eligibility",
        passage_index=0,
        content="Full refunds within 30 calendar days.",
        p_page_bin=0.95,
        p_para_bin=0.98,
        stage2_score=0.98,
        p_mul=0.97,
    )
    cfg = Settings(_env_file=None, OPENAI_API_KEY="", GENERATOR_API_KEY="")
    res = await generate_grounded_answer(
        passages=[top_passage],
        settings=cfg,
        query="What is the refund window?",
    )
    assert "Full refunds within 30 calendar days. [1]" in res.answer
    assert "Generator LLM key not configured" in res.answer
