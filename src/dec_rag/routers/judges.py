import time

from fastapi import APIRouter, Depends

from dec_rag.auth import AuthContext, require_api_key
from dec_rag.config import Settings, get_settings
from dec_rag.routers.retrieval import JudgeFactory, get_judge_factory
from dec_rag.schemas import (
    GeneratorTestRequest,
    GeneratorTestResponse,
    JudgeTestRequest,
    JudgeTestResponse,
    apply_runtime_credentials,
)
from dec_rag.services.generator import test_generator_connection

router = APIRouter(prefix="/judges", tags=["judges"])


@router.post("/test", response_model=JudgeTestResponse)
async def test_judge_connection(
    body: JudgeTestRequest,
    _auth: AuthContext = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
    judge_factory: JudgeFactory = Depends(get_judge_factory),
) -> JudgeTestResponse:
    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    judge = judge_factory(body.judge, effective_settings)

    sample_state = (
        "User Question: What is the refund window?\n\n"
        "Paragraph 1: Customers can request a full refund within 30 calendar days of purchase.\n"
        "Paragraph 2: Our headquarters is located in Austin, Texas."
    )
    t0 = time.perf_counter()
    bin_res = await judge.abinary(
        sample_state,
        {
            "p_1": "Does Paragraph 1 contain information needed to answer the user's question?",
            "p_2": "Does Paragraph 2 contain information needed to answer the user's question?",
        },
    )
    mc_res = await judge.achoice(
        sample_state,
        "Which passage most directly contains the information needed to answer the user's question?",
        {"opt_1": "Paragraph 1", "opt_2": "Paragraph 2"},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return JudgeTestResponse(
        judge=body.judge,
        model=judge.model,
        endpoint=judge.endpoint,
        ok=True,
        latency_ms=elapsed_ms,
        p_relevant=bin_res.probabilities["p_1"],
        p_irrelevant=bin_res.probabilities["p_2"],
        choice_winner=mc_res.winner,
    )


@router.post("/test-generator", response_model=GeneratorTestResponse)
async def test_generator_endpoint(
    body: GeneratorTestRequest,
    _auth: AuthContext = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> GeneratorTestResponse:
    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    gen_key = body.runtime_credentials.generator_api_key if body.runtime_credentials else None
    model, endpoint, latency_ms, reply = await test_generator_connection(
        settings=effective_settings,
        api_key=gen_key,
    )
    return GeneratorTestResponse(
        model=model,
        endpoint=endpoint,
        ok=True,
        latency_ms=latency_ms,
        reply=reply,
    )



@router.post("/test-generator", response_model=GeneratorTestResponse)
async def test_generator_endpoint(
    body: GeneratorTestRequest,
    _auth: AuthContext = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> GeneratorTestResponse:
    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    gen_key = body.runtime_credentials.generator_api_key if body.runtime_credentials else None
    model, endpoint, latency_ms, reply = await test_generator_connection(
        settings=effective_settings,
        api_key=gen_key,
    )
    return GeneratorTestResponse(
        model=model,
        endpoint=endpoint,
        ok=True,
        latency_ms=latency_ms,
        reply=reply,
    )