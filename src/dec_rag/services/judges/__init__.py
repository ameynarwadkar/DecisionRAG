import httpx

from dec_rag.config import JudgeId, Settings
from dec_rag.schemas import JudgeStatusInfo
from dec_rag.services.judges.base import (
    BinaryBatchResult,
    ChoiceResult,
    DecisionJudge,
    DecisionJudgeError,
)
from dec_rag.services.judges.clef import ClefDecisionJudge, resolve_clef_endpoint
from dec_rag.services.judges.openai_luna import OpenAILunaDecisionJudge
from dec_rag.services.judges.openrouter import OpenRouterDecisionJudge

ALL_JUDGE_IDS: tuple[JudgeId, ...] = (
    "clef",
    "openrouter_jev",
    "openrouter_luna",
    "openai_luna",
)


def create_judge(
    judge_id: JudgeId,
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    retry_base_delay: float = 0.25,
) -> DecisionJudge:
    common = {
        "max_concurrency": settings.DECISION_CONCURRENCY,
        "timeout_seconds": settings.DECISION_TIMEOUT_SECONDS,
        "max_retries": settings.DECISION_MAX_RETRIES,
        "retry_base_delay": retry_base_delay,
        "transport": transport,
    }
    if judge_id == "clef":
        return ClefDecisionJudge(
            base_url=settings.CLEF_API_URL,
            api_key=settings.CLEF_API_KEY,
            model=settings.CLEF_MODEL,
            **common,
        )
    if judge_id == "openrouter_jev":
        return OpenRouterDecisionJudge(
            judge_id="openrouter_jev",
            model=settings.OPENROUTER_JEV_MODEL,
            endpoint=settings.OPENROUTER_API_URL,
            api_key=settings.OPENROUTER_API_KEY,
            **common,
        )
    if judge_id == "openrouter_luna":
        return OpenRouterDecisionJudge(
            judge_id="openrouter_luna",
            model=settings.OPENROUTER_LUNA_MODEL,
            endpoint=settings.OPENROUTER_API_URL,
            api_key=settings.OPENROUTER_API_KEY,
            **common,
        )
    if judge_id == "openai_luna":
        return OpenAILunaDecisionJudge(
            endpoint=settings.OPENAI_DECISIONS_API_URL,
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_LUNA_MODEL,
            **common,
        )
    raise DecisionJudgeError(str(judge_id), 400, f"Unsupported decision judge '{judge_id}'.")


def list_judge_statuses(settings: Settings) -> list[JudgeStatusInfo]:
    clef_ep = resolve_clef_endpoint(settings.CLEF_API_URL)
    return [
        JudgeStatusInfo(
            judge_id="clef",
            model=settings.CLEF_MODEL,
            endpoint=clef_ep or "<unconfigured>/v1/systemone",
            configured=bool(clef_ep and settings.CLEF_API_KEY.strip()),
        ),
        JudgeStatusInfo(
            judge_id="openrouter_jev",
            model=settings.OPENROUTER_JEV_MODEL,
            endpoint=settings.OPENROUTER_API_URL,
            configured=bool(settings.OPENROUTER_API_URL.strip() and settings.OPENROUTER_API_KEY.strip()),
        ),
        JudgeStatusInfo(
            judge_id="openrouter_luna",
            model=settings.OPENROUTER_LUNA_MODEL,
            endpoint=settings.OPENROUTER_API_URL,
            configured=bool(settings.OPENROUTER_API_URL.strip() and settings.OPENROUTER_API_KEY.strip()),
        ),
        JudgeStatusInfo(
            judge_id="openai_luna",
            model=settings.OPENAI_LUNA_MODEL,
            endpoint=settings.OPENAI_DECISIONS_API_URL,
            configured=bool(settings.OPENAI_DECISIONS_API_URL.strip() and settings.OPENAI_API_KEY.strip()),
        ),
    ]


__all__ = [
    "ALL_JUDGE_IDS",
    "BinaryBatchResult",
    "ChoiceResult",
    "ClefDecisionJudge",
    "DecisionJudge",
    "DecisionJudgeError",
    "OpenAILunaDecisionJudge",
    "OpenRouterDecisionJudge",
    "create_judge",
    "list_judge_statuses",
]