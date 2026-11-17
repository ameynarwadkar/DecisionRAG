from collections.abc import Sequence
from dataclasses import dataclass
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException, status
import litellm

from dec_rag.config import Settings
from dec_rag.schemas import DialogueTurn, RankedPassageItem

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, grounded assistant. Answer the user's question using ONLY the "
    "retrieved passages below. Cite supporting passage numbers in square brackets like [1] "
    "when applicable. If the passages do not contain enough information to answer, state "
    "clearly what is missing."
)

AZURE_DEPLOYMENT_RE = re.compile(r"/openai/deployments/([^/]+)", re.IGNORECASE)


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    latency_ms: float


def format_top_passages_context(
    passages: Sequence[RankedPassageItem],
    *,
    top_k: int = 5,
) -> str:
    selected = list(passages[:top_k])
    if not selected:
        return "No relevant passages were retrieved."

    blocks: list[str] = []
    for idx, item in enumerate(selected, start=1):
        header = f"[{idx}] Page: {item.page_title}"
        if item.heading_path:
            header += f" | Section: {item.heading_path}"
        blocks.append(f"{header}\n{item.content}")
    return "\n\n".join(blocks)


def _extract_message_content(resp: Any) -> str:
    choices = resp["choices"] if isinstance(resp, dict) else resp.choices
    first = choices[0]
    msg = first["message"] if isinstance(first, dict) else first.message
    content = msg["content"] if isinstance(msg, dict) else msg.content
    return str(content or "").strip()


def resolve_litellm_params(
    model: str,
    settings: Settings,
    explicit_api_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve model name and LiteLLM connection kwargs (supporting OpenAI, Azure OpenAI, OpenRouter, etc.)."""
    clean_model = model.strip()
    raw_base = (
        settings.GENERATOR_API_BASE.strip()
        or settings.AZURE_API_BASE.strip()
        or settings.AZURE_OPENAI_ENDPOINT.strip()
    )
    raw_version = (
        settings.GENERATOR_API_VERSION.strip()
        or settings.AZURE_OPENAI_API_VERSION.strip()
        or settings.AZURE_API_VERSION.strip()
    )
    azure_key = (
        (explicit_api_key.strip() if explicit_api_key and explicit_api_key.strip() else "")
        or settings.GENERATOR_API_KEY.strip()
        or settings.AZURE_API_KEY.strip()
        or settings.AZURE_OPENAI_API_KEY.strip()
    )

    provider = clean_model.split("/", 1)[0].lower() if "/" in clean_model else ""
    direct_openai_requested = provider == "openai" and (
        bool(settings.OPENAI_API_KEY.strip())
        or (bool(explicit_api_key and explicit_api_key.strip()) and not settings.GENERATOR_API_BASE.strip())
    )
    other_non_azure_provider = bool(provider and provider not in {"azure", "openai"})

    is_azure = provider == "azure" or (
        not direct_openai_requested
        and not other_non_azure_provider
        and bool(raw_base and (".azure.com" in raw_base.lower() or azure_key))
    )

    if is_azure and (raw_base or clean_model.lower().startswith("azure/")):
        parsed_deployment: str | None = None
        api_base = raw_base
        api_version = raw_version or "2024-10-21"

        if raw_base:
            parsed = urlparse(raw_base)
            dep_match = AZURE_DEPLOYMENT_RE.search(parsed.path)
            if dep_match:
                parsed_deployment = dep_match.group(1)
            qs = parse_qs(parsed.query)
            if "api-version" in qs and qs["api-version"] and not settings.GENERATOR_API_VERSION.strip():
                api_version = qs["api-version"][0]
            if parsed.scheme and parsed.netloc:
                api_base = f"{parsed.scheme}://{parsed.netloc}"

        if clean_model.lower().startswith("azure/"):
            azure_model = clean_model
        elif parsed_deployment and clean_model == "openai/gpt-4o-mini":
            azure_model = f"azure/{parsed_deployment}"
        elif "/" not in clean_model:
            azure_model = f"azure/{clean_model}"
        elif parsed_deployment:
            azure_model = f"azure/{parsed_deployment}"
        else:
            # Strip openai/ prefix if user configured Azure base + key but left openai/<deployment>
            azure_model = f"azure/{clean_model.split('/', 1)[1]}"

        kwargs: dict[str, Any] = {"drop_params": True}
        if api_base:
            kwargs["api_base"] = api_base
        if api_version:
            kwargs["api_version"] = api_version
        if azure_key:
            kwargs["api_key"] = azure_key
        return azure_model, kwargs

    # Non-Azure providers
    key: str | None = None
    if explicit_api_key and explicit_api_key.strip():
        key = explicit_api_key.strip()
    elif settings.GENERATOR_API_KEY.strip():
        key = settings.GENERATOR_API_KEY.strip()
    elif clean_model.lower().startswith("openrouter/") and settings.OPENROUTER_API_KEY.strip():
        key = settings.OPENROUTER_API_KEY.strip()
    elif (clean_model.lower().startswith("openai/") or "/" not in clean_model) and settings.OPENAI_API_KEY.strip():
        key = settings.OPENAI_API_KEY.strip()

    extra: dict[str, Any] = {}
    if key:
        extra["api_key"] = key
    if raw_base:
        extra["api_base"] = raw_base
    return clean_model, extra


def _is_missing_credentials_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "missing credentials" in msg
        or "api_key client option must be set" in msg
        or "no api key provided" in msg
        or ("authenticationerror" in type(exc).__name__.lower() and "missing" in msg)
    )


def is_generator_configured(settings: Settings) -> bool:
    _, kwargs = resolve_litellm_params(settings.GENERATOR_MODEL, settings)
    return bool(kwargs.get("api_key"))


async def test_generator_connection(
    *,
    settings: Settings,
    api_key: str | None = None,
) -> tuple[str, str, float, str]:
    """Ping the configured Generator LLM (OpenAI / Azure OpenAI / etc.). Returns (model, endpoint, latency_ms, reply)."""
    effective_model, extra_kwargs = resolve_litellm_params(
        settings.GENERATOR_MODEL,
        settings,
        explicit_api_key=api_key,
    )
    endpoint = str(extra_kwargs.get("api_base") or "default")
    t0 = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=effective_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0.0,
            **extra_kwargs,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Generator test failed ({effective_model}): {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    reply = _extract_message_content(resp)
    return effective_model, endpoint, elapsed_ms, reply


async def generate_grounded_answer(
    *,
    passages: Sequence[RankedPassageItem],
    settings: Settings,
    query: str | None = None,
    scenario: str | None = None,
    dialogue_turns: Sequence[DialogueTurn] | None = None,
    system_prompt: str | None = None,
    top_k_context: int = 5,
    api_key: str | None = None,
) -> GeneratedAnswer:
    """Run Stage 4 greedy grounded answer synthesis over R[:5] (Spec §7.3, App. C.1)."""
    context_block = format_top_passages_context(passages, top_k=top_k_context)
    sys_text = (system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_SYSTEM_PROMPT)

    if dialogue_turns:
        recent_turns = list(dialogue_turns[-6:])
        convo_lines = "\n".join(f"{t.role.upper()}: {t.content}" for t in recent_turns)
        user_prompt = (
            f"Retrieved Passages:\n{context_block}\n\n"
            f"Recent Conversation (up to 6 turns):\n{convo_lines}\n\n"
            "Write the assistant's grounded next reply based on the retrieved passages."
        )
    else:
        q_text = (query or "").strip()
        if scenario and scenario.strip():
            question_block = f"Scenario: {scenario.strip()}\nQuestion: {q_text}"
        else:
            question_block = f"Question: {q_text}"
        user_prompt = f"Retrieved Passages:\n{context_block}\n\n{question_block}"

    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_prompt},
    ]

    effective_model, extra_kwargs = resolve_litellm_params(
        settings.GENERATOR_MODEL,
        settings,
        explicit_api_key=api_key,
    )

    t0 = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=effective_model,
            messages=messages,
            temperature=0.0,
            **extra_kwargs,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if _is_missing_credentials_error(exc):
            if passages:
                top = passages[0]
                fallback = (
                    f"{top.content} [1]\n\n"
                    "(Generator LLM key not configured — showing top passage [1] directly. "
                    "Set Generator / Azure Key in Settings or .env to synthesize full answers.)"
                )
            else:
                fallback = "No relevant passages were retrieved (and no Generator LLM key is configured)."
            return GeneratedAnswer(answer=fallback, latency_ms=elapsed_ms)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Generator LLM ({effective_model}) failed: {exc}",
        ) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return GeneratedAnswer(answer=_extract_message_content(resp), latency_ms=elapsed_ms)
