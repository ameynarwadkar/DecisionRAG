from collections.abc import AsyncGenerator, Mapping
import re
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from dec_rag.config import JudgeId, Settings, get_settings
from dec_rag.database import create_db_engine, get_db, init_db
from dec_rag.main import create_app
from dec_rag.routers.retrieval import get_judge_factory
from dec_rag.services.judges import create_judge
from dec_rag.services.judges.base import BinaryBatchResult, ChoiceResult, DecisionJudge


class InspectingJudge(DecisionJudge):
    def __init__(self, judge_id: JudgeId, settings: Settings) -> None:
        # Delegate to real create_judge first so ensure_configured() validates credentials!
        real = create_judge(judge_id, settings)
        super().__init__(
            judge_id=judge_id,
            model=real.model,
            endpoint=real.endpoint,
            api_key=real.api_key,
        )

    async def abinary(
        self,
        state: str | dict[str, Any],
        questions: Mapping[str, str],
        *,
        yes_desc: str = "Yes, relevant.",
        no_desc: str = "No, not relevant.",
    ) -> BinaryBatchResult:
        del state, yes_desc, no_desc
        self.ensure_configured()
        keys = list(questions.keys())
        probs = {k: (0.95 if idx == 0 else 0.03) for idx, k in enumerate(keys)}
        return BinaryBatchResult(probabilities=probs, requests_made=1, input_tokens=20, cost_usd=0.0)

    async def achoice(
        self,
        state: str | dict[str, Any],
        instruction: str,
        options: Mapping[str, str],
    ) -> ChoiceResult:
        del state, instruction
        self.ensure_configured()
        keys = list(options.keys())
        probs = {k: (0.9 if idx == 0 else 0.1) for idx, k in enumerate(keys)}
        return ChoiceResult(probabilities=probs, winner=keys[0], requests_made=1, input_tokens=15, cost_usd=0.0)


@pytest_asyncio.fixture
async def runtime_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[tuple[httpx.AsyncClient, Settings, list[dict[str, Any]]], None]:
    engine: AsyncEngine = create_db_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    # Start with ZERO judge credentials in server Settings
    base_settings = Settings(
        _env_file=None,
        DEV_BYPASS_AUTH=True,
        CLEF_API_URL="",
        CLEF_API_KEY="",
        OPENROUTER_API_KEY="",
        OPENAI_API_KEY="",
        GENERATOR_API_KEY="",
        GENERATOR_API_BASE="",
        AZURE_API_BASE="",
        AZURE_API_KEY="",
        AZURE_OPENAI_ENDPOINT="",
        AZURE_OPENAI_API_KEY="",
    )
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: base_settings
    app.dependency_overrides[get_judge_factory] = lambda: (
        lambda j_id, cfg: InspectingJudge(j_id, cfg)
    )

    llm_calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        llm_calls.append(kwargs)
        return {"choices": [{"message": {"content": "Runtime answer [1]."}}]}

    monkeypatch.setattr("dec_rag.services.generator.litellm.acompletion", fake_acompletion)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, base_settings, llm_calls

    await engine.dispose()


async def test_judge_tester_and_ephemeral_runtime_credentials_isolation(
    runtime_client: tuple[httpx.AsyncClient, Settings, list[dict[str, Any]]],
) -> None:
    client, base_settings, llm_calls = runtime_client

    # 1. Without runtime_credentials, unconfigured 'clef' fails fast with 503
    fail_ping = await client.post("/api/v1/judges/test", json={"judge": "clef"})
    assert fail_ping.status_code == 503

    # 2. With browser runtime_credentials, /api/v1/judges/test succeeds!
    ok_ping = await client.post(
        "/api/v1/judges/test",
        json={
            "judge": "clef",
            "runtime_credentials": {
                "clef_api_url": "https://runtime-tunnel.example.com",
                "clef_api_key": "runtime-clef-secret",
            },
        },
    )
    assert ok_ping.status_code == 200
    body = ok_ping.json()
    assert body["ok"] is True
    assert body["endpoint"] == "https://runtime-tunnel.example.com/v1/systemone"
    assert body["p_relevant"] == pytest.approx(0.95)
    assert body["choice_winner"] == "opt_1"

    # Global base_settings MUST remain completely untouched
    assert base_settings.CLEF_API_URL == ""
    assert base_settings.CLEF_API_KEY == ""

    # 3. Verify /query forwards runtime generator_model & generator_api_key per-request
    ns_resp = await client.post("/api/v1/namespaces", json={"name": "Runtime NS"})
    ns_id = ns_resp.json()["id"]
    await client.post(
        f"/api/v1/namespaces/{ns_id}/documents",
        files={"file": ("doc.md", b"# Refund\n\n30 days refund.", "text/markdown")},
    )

    q_resp = await client.post(
        f"/api/v1/namespaces/{ns_id}/query",
        json={
            "query": "Refund window?",
            "judge": "openrouter_jev",
            "runtime_credentials": {
                "openrouter_api_key": "or-runtime-key",
                "generator_model": "openai/gpt-4o",
                "generator_api_key": "sk-runtime-llm",
            },
        },
    )
    assert q_resp.status_code == 200
    assert q_resp.json()["answer"] == "Runtime answer [1]."
    assert llm_calls[-1]["model"] == "openai/gpt-4o"
    assert llm_calls[-1]["api_key"] == "sk-runtime-llm"
    assert base_settings.OPENROUTER_API_KEY == ""

    # 4. Verify /api/v1/judges/test-generator works with Azure OpenAI runtime overrides
    gen_ping = await client.post(
        "/api/v1/judges/test-generator",
        json={
            "runtime_credentials": {
                "generator_model": "azure/gpt-5.4-nano",
                "generator_api_base": "https://test-res.cognitiveservices.azure.com",
                "generator_api_key": "az-runtime-key",
                "generator_api_version": "2025-04-01-preview",
            }
        },
    )
    assert gen_ping.status_code == 200
    assert gen_ping.json()["ok"] is True
    assert gen_ping.json()["model"] == "azure/gpt-5.4-nano"
    assert llm_calls[-1]["api_base"] == "https://test-res.cognitiveservices.azure.com"
    assert llm_calls[-1]["api_version"] == "2025-04-01-preview"
    assert llm_calls[-1]["api_key"] == "az-runtime-key"


async def test_playground_ui_and_compiled_typescript_assets_served(
    runtime_client: tuple[httpx.AsyncClient, Settings, list[dict[str, Any]]],
) -> None:
    client, _, _ = runtime_client

    root_resp = await client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers.get("content-type", "")
    html = root_resp.text
    assert "DecisionRAG" in html

    js_match = re.search(r'src="(/static/assets/[^"]+\.js)"', html)
    css_match = re.search(r'href="(/static/assets/[^"]+\.css)"', html)
    assert js_match is not None, "Compiled TypeScript bundle script tag not found in index.html"
    assert css_match is not None, "Compiled stylesheet link tag not found in index.html"

    js_resp = await client.get(js_match.group(1))
    assert js_resp.status_code == 200
    assert len(js_resp.text) > 1000

    css_resp = await client.get(css_match.group(1))
    assert css_resp.status_code == 200
    assert "--bg-base" in css_resp.text
