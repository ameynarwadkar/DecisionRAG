import asyncio
import json

import httpx
import pytest

from dec_rag.config import Settings
from dec_rag.services.judges import DecisionJudgeError, create_judge, list_judge_statuses


async def test_clef_wire_format_binary_and_choice() -> None:
    captured: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((str(request.url), request.headers.get("Authorization", ""), body))
        if "best" in body["questions"]:
            return httpx.Response(
                200,
                json={
                    "decisions": {"best": {"p1": 0.2, "p2": 0.8}},
                    "usage": {"input_tokens": 120, "cost_usd": 0.001},
                },
            )
        return httpx.Response(
            200,
            json={
                "decisions": {
                    "q1": {"yes": 0.91, "no": 0.09},
                    "q2": {"yes": 0.04, "no": 0.96},
                },
                "usage": {"input_tokens": 200, "cost_usd": 0.002},
            },
        )

    cfg = Settings(
        _env_file=None,
        CLEF_API_URL="https://colab-tunnel.example.com",
        CLEF_API_KEY="clef-secret",
    )
    judge = create_judge("clef", cfg, transport=httpx.MockTransport(handler), retry_base_delay=0.001)

    bin_res = await judge.abinary("Shared context", {"q1": "Page 1?", "q2": "Page 2?"})
    assert bin_res.probabilities == {"q1": pytest.approx(0.91), "q2": pytest.approx(0.04)}
    assert bin_res.requests_made == 1
    assert bin_res.input_tokens == 200

    url, auth, payload = captured[0]
    assert url == "https://colab-tunnel.example.com/v1/systemone"
    assert auth == "Bearer clef-secret"
    assert payload["questions"]["q1"]["type"] == "choice"
    assert set(payload["questions"]["q1"]["criteria"].keys()) == {"yes", "no"}

    mc_res = await judge.achoice("Shortlist context", "Which passage is best?", {"p1": "Passage 1", "p2": "Passage 2"})
    assert mc_res.winner == "p2"
    assert mc_res.probabilities["p2"] == pytest.approx(0.8)
    assert sum(mc_res.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("judge_id", "expected_model"),
    [
        ("openrouter_jev", "typesafe/jev-1.13"),
        ("openrouter_luna", "openai/gpt-6-luna-decisions"),
    ],
)
async def test_openrouter_jev_and_luna_wire_format(judge_id: str, expected_model: str) -> None:
    captured_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        assert request.headers["Authorization"] == "Bearer or-key"
        body = json.loads(request.content.decode("utf-8"))
        captured_models.append(body["model"])
        if "best" in body["questions"]:
            assert body["questions"]["best"]["type"] == "choice"
            return httpx.Response(
                200,
                json={"answers": {"best": {"probabilities": {"optA": 0.75, "optB": 0.25}}}},
            )
        assert body["questions"]["item_1"]["type"] == "noul"
        return httpx.Response(200, json={"answers": {"item_1": {"noul": 0.63}}})

    cfg = Settings(_env_file=None, OPENROUTER_API_KEY="or-key")
    judge = create_judge(judge_id, cfg, transport=httpx.MockTransport(handler), retry_base_delay=0.001)  # type: ignore[arg-type]

    b = await judge.abinary("Context", {"item_1": "Is item 1 relevant?"})
    assert b.probabilities["item_1"] == pytest.approx(0.63)

    c = await judge.achoice("Context", "Best option?", {"optA": "A", "optB": "B"})
    assert c.winner == "optA"
    assert captured_models == [expected_model, expected_model]


async def test_openai_luna_wire_format_and_400_predicate_chunking() -> None:
    chunk_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/decisions"
        assert request.headers["Authorization"] == "Bearer sk-openai"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gpt-6-luna"
        assert "input" in body

        q_list = body["questions"]
        if q_list[0]["type"] == "choice":
            assert q_list[0]["name"] == "best"
            assert q_list[0]["choices"] == [
                {"value": "c1", "description": "Passage 1"},
                {"value": "c2", "description": "Passage 2"},
            ]
            return httpx.Response(
                200,
                json={"answers": [{"name": "best", "probabilities": {"c1": 0.15, "c2": 0.85}}]},
            )

        chunk_sizes.append(len(q_list))
        return httpx.Response(
            200,
            json={
                "answers": [{"name": q["name"], "probability": 0.5} for q in q_list],
                "usage": {"input_tokens": 50, "cost_usd": 0.0005},
            },
        )

    cfg = Settings(_env_file=None, OPENAI_API_KEY="sk-openai")
    judge = create_judge("openai_luna", cfg, transport=httpx.MockTransport(handler), retry_base_delay=0.001)

    # 450 questions must split into 400 + 50 across 2 requests
    many_questions = {f"q_{i}": f"Question {i}?" for i in range(450)}
    bin_res = await judge.abinary("Large page title list", many_questions)
    assert sorted(chunk_sizes) == [50, 400]
    assert bin_res.requests_made == 2
    assert bin_res.input_tokens == 100
    assert len(bin_res.probabilities) == 450

    mc_res = await judge.achoice("Shortlist", "Pick best", {"c1": "Passage 1", "c2": "Passage 2"})
    assert mc_res.winner == "c2"


async def test_retry_on_503_and_semaphore_8_cap_and_fail_fast_missing_creds() -> None:
    # 1. Fail-fast 503 when credentials missing
    unconfigured_cfg = Settings(_env_file=None, OPENROUTER_API_KEY="")
    unconfigured_judge = create_judge("openrouter_jev", unconfigured_cfg)
    statuses = {s.judge_id: s.configured for s in list_judge_statuses(unconfigured_cfg)}
    assert statuses["openrouter_jev"] is False

    with pytest.raises(DecisionJudgeError) as exc_info:
        await unconfigured_judge.abinary("ctx", {"q1": "test?"})
    assert exc_info.value.status_code == 503

    # 2. Retry transient 503 + enforce <= 8 in-flight concurrent requests
    attempts = 0
    in_flight = 0
    peak_in_flight = 0

    async def async_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts, in_flight, peak_in_flight
        attempts += 1
        my_attempt = attempts
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await asyncio.sleep(0.01)
            body = json.loads(request.content.decode("utf-8"))
            if my_attempt == 1:
                return httpx.Response(503, text="Temporary overload")
            return httpx.Response(
                200,
                json={"answers": {qid: {"noul": 0.7} for qid in body["questions"]}},
            )
        finally:
            in_flight -= 1

    cfg = Settings(_env_file=None, OPENROUTER_API_KEY="or-key", DECISION_CONCURRENCY=8)
    judge = create_judge(
        "openrouter_jev",
        cfg,
        transport=httpx.MockTransport(async_handler),
        retry_base_delay=0.001,
    )

    results = await asyncio.gather(*(judge.abinary("ctx", {"q1": "ok?"}) for _ in range(16)))
    assert len(results) == 16
    assert peak_in_flight <= 8
    assert attempts == 17  # 16 calls + 1 transient 503 retry