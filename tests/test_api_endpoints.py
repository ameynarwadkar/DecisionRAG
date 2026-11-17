from collections.abc import AsyncGenerator, Mapping
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from dec_rag.config import JudgeId, Settings, get_settings
from dec_rag.database import create_db_engine, get_db, init_db
from dec_rag.main import create_app
from dec_rag.routers.retrieval import get_judge_factory
from dec_rag.services.judges.base import BinaryBatchResult, ChoiceResult, DecisionJudge


class StubApiJudge(DecisionJudge):
    def __init__(self, judge_id: JudgeId, prefer_last: bool = False) -> None:
        super().__init__(
            judge_id=judge_id,
            model=f"stub-{judge_id}",
            endpoint="https://stub.local/decisions",
            api_key="stub-key",
        )
        self.prefer_last = prefer_last

    async def abinary(
        self,
        state: str | dict[str, Any],
        questions: Mapping[str, str],
        *,
        yes_desc: str = "Yes, relevant.",
        no_desc: str = "No, not relevant.",
    ) -> BinaryBatchResult:
        del state, yes_desc, no_desc
        keys = list(questions.keys())
        probs: dict[str, float] = {}
        for idx, k in enumerate(keys):
            score = (0.2 + 0.7 * (idx / max(1, len(keys) - 1))) if self.prefer_last else (0.9 - 0.1 * idx)
            probs[k] = max(0.05, min(0.99, score))
        return BinaryBatchResult(probabilities=probs, requests_made=1, input_tokens=40, cost_usd=0.001)

    async def achoice(
        self,
        state: str | dict[str, Any],
        instruction: str,
        options: Mapping[str, str],
    ) -> ChoiceResult:
        del state, instruction
        keys = list(options.keys())
        winner = keys[-1] if self.prefer_last else keys[0]
        probs = {k: (0.8 if k == winner else 0.2 / max(1, len(keys) - 1)) for k in keys}
        return ChoiceResult(
            probabilities=probs,
            winner=winner,
            requests_made=1,
            input_tokens=25,
            cost_usd=0.001,
        )


@pytest_asyncio.fixture
async def api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[tuple[httpx.AsyncClient, Any], None]:
    engine: AsyncEngine = create_db_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    test_settings = Settings(_env_file=None, DEV_BYPASS_AUTH=True)
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: test_settings

    async def fake_acompletion(*, model: str, messages: list[dict[str, str]], temperature: float, **kwargs: Any) -> dict[str, Any]:
        del model, messages, kwargs
        assert temperature == 0.0
        return {"choices": [{"message": {"content": "Grounded answer [1]."}}]}

    monkeypatch.setattr("dec_rag.services.generator.litellm.acompletion", fake_acompletion)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, app

    await engine.dispose()


async def test_health_and_api_key_lifecycle(
    api_client: tuple[httpx.AsyncClient, Any],
) -> None:
    client, app = api_client

    # 1. Health check
    h_resp = await client.get("/api/v1/health")
    assert h_resp.status_code == 200
    h_json = h_resp.json()
    assert h_json["status"] == "ok"
    assert {j["judge_id"] for j in h_json["judges"]} == {
        "clef",
        "openrouter_jev",
        "openrouter_luna",
        "openai_luna",
    }

    # 2. Create API key while DEV_BYPASS_AUTH=True
    k_resp = await client.post("/api/v1/api-keys", json={"name": "CI Runner"})
    assert k_resp.status_code == 201
    created_key = k_resp.json()
    raw_key = created_key["api_key"]
    key_id = created_key["id"]

    # 3. Enforce strict auth (DEV_BYPASS_AUTH=False)
    strict_settings = Settings(_env_file=None, DEV_BYPASS_AUTH=False)
    app.dependency_overrides[get_settings] = lambda: strict_settings

    unauth = await client.get("/api/v1/namespaces")
    assert unauth.status_code == 401

    authed = await client.get("/api/v1/api-keys", headers={"X-API-Key": raw_key})
    assert authed.status_code == 200
    assert len(authed.json()) == 1

    del_resp = await client.delete(f"/api/v1/api-keys/{key_id}", headers={"Authorization": f"Bearer {raw_key}"})
    assert del_resp.status_code == 204


async def test_end_to_end_ingest_retrieve_query_chat_and_compare_judges(
    api_client: tuple[httpx.AsyncClient, Any],
) -> None:
    client, app = api_client

    # 1. Create namespace
    ns_resp = await client.post(
        "/api/v1/namespaces",
        json={"name": "Support KB", "default_profile": "auto"},
    )
    assert ns_resp.status_code == 201
    ns_id = ns_resp.json()["id"]

    # 2. Upload 2 structured Markdown guides under a category
    up1 = await client.post(
        f"/api/v1/namespaces/{ns_id}/documents",
        data={"category": "Billing"},
        files={"file": ("refunds.md", b"# Refund Policy\n\n## Window\n\nFull refunds within 30 days.", "text/markdown")},
    )
    assert up1.status_code == 201
    page1_id = up1.json()["page"]["id"]

    up2 = await client.post(
        f"/api/v1/namespaces/{ns_id}/documents",
        data={"category": "Billing"},
        files={"file": ("invoices.md", b"# Invoice Exports\n\n## PDF\n\nDownload invoices from Settings.", "text/markdown")},
    )
    assert up2.status_code == 201
    page2_id = up2.json()["page"]["id"]

    # Inspect namespace detail & reading-order passages
    ns_detail = await client.get(f"/api/v1/namespaces/{ns_id}")
    assert ns_detail.status_code == 200
    assert ns_detail.json()["page_count"] == 2
    assert len(ns_detail.json()["categories"]) == 1

    pas_resp = await client.get(f"/api/v1/namespaces/{ns_id}/documents/{page1_id}/passages")
    assert pas_resp.status_code == 200
    assert pas_resp.json()[0]["heading_path"] == "Window"

    # 3. Verify fail-fast 503 when calling /retrieve with unconfigured real judge (zero BM25 fallback!)
    fail_resp = await client.post(
        f"/api/v1/namespaces/{ns_id}/retrieve",
        json={"query": "How do refunds work?", "judge": "openrouter_jev"},
    )
    assert fail_resp.status_code == 503
    assert fail_resp.json()["judge_id"] == "openrouter_jev"

    # 4. Inject deterministic StubApiJudge factory for /retrieve, /query, /chat, and /compare-judges
    app.dependency_overrides[get_judge_factory] = lambda: (
        lambda j_id, _cfg: StubApiJudge(j_id, prefer_last=(j_id == "clef"))
    )

    ret_resp = await client.post(
        f"/api/v1/namespaces/{ns_id}/retrieve",
        json={"query": "How do refunds work?", "judge": "openrouter_jev"},
    )
    assert ret_resp.status_code == 200
    assert ret_resp.json()["passages"][0]["page_id"] == page1_id

    q_resp = await client.post(
        f"/api/v1/namespaces/{ns_id}/query",
        json={"query": "How do refunds work?", "judge": "openrouter_jev"},
    )
    assert q_resp.status_code == 200
    assert q_resp.json()["answer"] == "Grounded answer [1]."

    # 5. Multi-turn /chat: Turn 1 prefers page1 (openrouter_jev), Turn 2 prefers page2 (clef) -> Notepad updates!
    chat1 = await client.post(
        f"/api/v1/namespaces/{ns_id}/chat",
        json={"message": "Tell me about refunds.", "judge": "openrouter_jev"},
    )
    assert chat1.status_code == 200
    session_id = chat1.json()["session_id"]
    assert chat1.json()["notepad_page_ids"] == [page1_id]

    chat2 = await client.post(
        f"/api/v1/namespaces/{ns_id}/chat",
        json={
            "session_id": session_id,
            "message": "Now where do I download invoices?",
            "judge": "clef",
        },
    )
    assert chat2.status_code == 200
    assert chat2.json()["notepad_page_ids"] == [page2_id, page1_id]

    # 6. Side-by-side /compare-judges
    cmp_resp = await client.post(
        f"/api/v1/namespaces/{ns_id}/compare-judges",
        json={
            "query": "Compare billing policies",
            "judges": ["openrouter_jev", "clef"],
        },
    )
    assert cmp_resp.status_code == 200
    cmp_results = cmp_resp.json()["results"]
    assert [r["judge"] for r in cmp_results] == ["openrouter_jev", "clef"]
    assert cmp_results[0]["passages"][0]["page_id"] == page1_id
    assert cmp_results[1]["passages"][0]["page_id"] == page2_id

    # 7. Delete page & namespace cleanup
    del_page = await client.delete(f"/api/v1/namespaces/{ns_id}/documents/{page2_id}")
    assert del_page.status_code == 204
    del_ns = await client.delete(f"/api/v1/namespaces/{ns_id}")
    assert del_ns.status_code == 204