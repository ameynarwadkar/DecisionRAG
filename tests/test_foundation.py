import hashlib
from pathlib import Path
import tomllib
import uuid

from fastapi import HTTPException
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.auth import generate_api_key, verify_api_key_token
from dec_rag.config import Settings
from dec_rag.database import Base
from dec_rag.models import (
    ApiKey,
    CategoryNode,
    ConversationSession,
    DocumentPage,
    Namespace,
    Passage,
    SectionHeading,
    TopicCluster,
)
from dec_rag.schemas import NavigationOverrides, resolve_profile_config


def test_pyproject_strictly_excludes_pgvector_and_includes_numpy() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"]).lower()
    assert "pgvector" not in deps
    assert "numpy" in deps


def test_settings_defaults_and_concurrency_cap() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.DEFAULT_JUDGE == "openrouter_jev"
    assert cfg.DECISION_CONCURRENCY == 8
    assert cfg.MAX_PASSAGE_CHARS == 1200

    with pytest.raises(ValidationError):
        Settings(_env_file=None, DECISION_CONCURRENCY=9)


async def test_sqlite_foreign_keys_enabled(db_session: AsyncSession) -> None:
    result = await db_session.execute(text("PRAGMA foreign_keys"))
    assert result.scalar_one() == 1


def test_orm_models_have_zero_embedding_or_vector_columns() -> None:
    expected_tables = {
        "namespaces",
        "category_nodes",
        "topic_clusters",
        "document_pages",
        "section_headings",
        "passages",
        "conversation_sessions",
        "api_keys",
    }
    assert set(Base.metadata.tables.keys()) == expected_tables

    for table_name, table in Base.metadata.tables.items():
        for col in table.columns:
            col_lower = col.name.lower()
            assert "embedding" not in col_lower, f"Unexpected embedding column on {table_name}.{col.name}"
            assert "vector" not in col_lower, f"Unexpected vector column on {table_name}.{col.name}"


async def test_hierarchy_persistence_and_cascade_delete(db_session: AsyncSession) -> None:
    ns = Namespace(name="Paper Benchmark", slug="paper-bench")
    db_session.add(ns)
    await db_session.flush()

    cat = CategoryNode(namespace_id=ns.id, name="Billing", slug="billing")
    cluster = TopicCluster(namespace_id=ns.id, cluster_index=0, summary="Billing and invoices.", page_count=1)
    db_session.add_all([cat, cluster])
    await db_session.flush()

    page = DocumentPage(
        namespace_id=ns.id,
        category_id=cat.id,
        cluster_id=cluster.id,
        cluster_rank=0,
        doc_order=0,
        filename="refunds.md",
        title="Refund Policy",
        first_sentence="Customers may request a refund within 30 days.",
        headings_summary=["Eligibility"],
        passage_count=1,
        status="READY",
    )
    db_session.add(page)
    await db_session.flush()

    heading = SectionHeading(
        page_id=page.id,
        namespace_id=ns.id,
        heading_index=0,
        level=1,
        title="Eligibility",
        heading_path="Eligibility",
    )
    db_session.add(heading)
    await db_session.flush()

    passage = Passage(
        page_id=page.id,
        namespace_id=ns.id,
        heading_id=heading.id,
        passage_index=0,
        heading_path="Eligibility",
        content="Customers may request a refund within 30 days of purchase.",
        norm_hash=hashlib.sha256(b"customers may request a refund within 30 days of purchase.").hexdigest(),
        token_estimate=12,
    )
    conv = ConversationSession(
        namespace_id=ns.id,
        turns_json=[{"role": "user", "content": "How do refunds work?"}],
        notepad_page_ids=[str(page.id)],
    )
    db_session.add_all([passage, conv])
    await db_session.commit()

    # Delete namespace and verify full cascade across hierarchy
    await db_session.delete(ns)
    await db_session.commit()

    for model in (CategoryNode, TopicCluster, DocumentPage, SectionHeading, Passage, ConversationSession):
        remaining = (await db_session.execute(select(func.count()).select_from(model))).scalar_one()
        assert remaining == 0


def test_paper_profile_presets_and_overrides() -> None:
    cqa = resolve_profile_config("cqa", has_categories=False, has_clusters=False, total_ready_pages=200)
    assert (cqa.l0_mode, cqa.k_pages, cqa.mc_mode, cqa.multiply_page_prob) == ("none", 3, "order", False)

    mtrag = resolve_profile_config("mtrag", has_categories=True, has_clusters=False, total_ready_pages=800)
    assert (mtrag.two_pass_top_pages, mtrag.enable_heading_stage, mtrag.mc_mode) == (50, True, "top1")

    nq = resolve_profile_config("nq", has_categories=False, has_clusters=True, total_ready_pages=10_000)
    assert (nq.l0_mode, nq.k_pages, nq.tiered_passage_pool, nq.multiply_page_prob, nq.m_shortlist) == (
        "clusters",
        30,
        True,
        True,
        5,
    )

    auto_over = resolve_profile_config(
        "auto",
        has_categories=True,
        has_clusters=False,
        total_ready_pages=100,
        overrides=NavigationOverrides(k_pages=15, mc_mode="off"),
    )
    assert auto_over.l0_mode == "categories"
    assert auto_over.k_pages == 15
    assert auto_over.tiered_passage_pool is True
    assert auto_over.multiply_page_prob is True
    assert auto_over.mc_mode == "off"


async def test_api_key_auth_and_namespace_scoping(db_session: AsyncSession) -> None:
    ns1 = Namespace(name="NS1", slug="ns-1")
    ns2 = Namespace(name="NS2", slug="ns-2")
    db_session.add_all([ns1, ns2])
    await db_session.flush()

    raw_key, prefix, key_hash = generate_api_key()
    api_key = ApiKey(name="NS1 Scoped Key", key_prefix=prefix, key_hash=key_hash, namespace_id=ns1.id)
    db_session.add(api_key)
    await db_session.commit()

    strict_cfg = Settings(_env_file=None, DEV_BYPASS_AUTH=False)
    bypass_cfg = Settings(_env_file=None, DEV_BYPASS_AUTH=True)

    with pytest.raises(HTTPException) as exc_missing:
        await verify_api_key_token(None, db_session, strict_cfg)
    assert exc_missing.value.status_code == 401

    bypass_ctx = await verify_api_key_token(None, db_session, bypass_cfg)
    assert bypass_ctx.bypassed is True

    valid_ctx = await verify_api_key_token(raw_key, db_session, strict_cfg)
    assert valid_ctx.api_key_id == api_key.id
    valid_ctx.ensure_namespace_access(ns1.id)

    with pytest.raises(HTTPException) as exc_forbidden:
        valid_ctx.ensure_namespace_access(ns2.id)
    assert exc_forbidden.value.status_code == 403
    assert exc_forbidden.value.detail == "API key is not authorized for this namespace."
    assert uuid.UUID(str(valid_ctx.api_key_id)) == api_key.id


def test_get_settings_hot_reloads_modified_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    from dec_rag import config as config_mod

    fake_env = tmp_path / ".env"
    fake_env.write_text("CLEF_API_URL=https://first-tunnel.example.com\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "resolve_env_file", lambda: fake_env)

    s1 = config_mod.get_settings()
    assert s1.CLEF_API_URL == "https://first-tunnel.example.com"

    fake_env.write_text("CLEF_API_URL=https://second-tunnel.example.com\n", encoding="utf-8")
    stat = fake_env.stat()
    os.utime(fake_env, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000))

    s2 = config_mod.get_settings()
    assert s2.CLEF_API_URL == "https://second-tunnel.example.com"
