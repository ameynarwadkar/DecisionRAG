from abc import ABC, abstractmethod
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import httpx

from dec_rag.config import JudgeId

RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


@dataclass
class BinaryBatchResult:
    probabilities: dict[str, float]  # item_id -> P(Yes) in [0.0, 1.0] (not normalized across items)
    requests_made: int
    input_tokens: int
    cost_usd: float


@dataclass
class ChoiceResult:
    probabilities: dict[str, float]  # option_id -> P(option) summing to 1.0
    winner: str
    requests_made: int
    input_tokens: int
    cost_usd: float


class DecisionJudgeError(Exception):
    """Fail-fast error raised when a decision backend is unconfigured or fails."""

    def __init__(self, judge_id: str, status_code: int, detail: str) -> None:
        super().__init__(f"[{judge_id}] ({status_code}) {detail}")
        self.judge_id = judge_id
        self.status_code = status_code
        self.detail = detail


def clamp_probability(value: Any, *, judge_id: str, field_name: str) -> float:
    try:
        prob = float(value)
    except (TypeError, ValueError) as exc:
        raise DecisionJudgeError(
            judge_id,
            502,
            f"Malformed probability for '{field_name}': {value!r}",
        ) from exc
    if not math.isfinite(prob):
        raise DecisionJudgeError(
            judge_id,
            502,
            f"Non-finite probability for '{field_name}': {value!r}",
        )
    if prob < -1e-4 or prob > 1.0 + 1e-4:
        raise DecisionJudgeError(
            judge_id,
            502,
            f"Probability out of [0, 1] range for '{field_name}': {prob}",
        )
    return min(1.0, max(0.0, prob))


def normalize_choice_probabilities(
    raw_probs: Mapping[str, Any],
    option_ids: Sequence[str],
    *,
    judge_id: str,
) -> tuple[dict[str, float], str]:
    parsed: dict[str, float] = {}
    for oid in option_ids:
        if oid not in raw_probs:
            raise DecisionJudgeError(
                judge_id,
                502,
                f"Choice response missing probability for option '{oid}'.",
            )
        parsed[oid] = clamp_probability(raw_probs[oid], judge_id=judge_id, field_name=oid)

    total = sum(parsed.values())
    if total <= 0.0:
        uniform = 1.0 / len(option_ids)
        normalized = {oid: uniform for oid in option_ids}
    else:
        normalized = {oid: parsed[oid] / total for oid in option_ids}

    winner = max(option_ids, key=lambda oid: normalized[oid])
    return normalized, winner


def extract_usage(data: Mapping[str, Any]) -> tuple[int, float]:
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return 0, 0.0
    tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    cost = float(usage.get("cost_usd") or usage.get("cost") or 0.0)
    return tokens, cost


class DecisionJudge(ABC):
    """Unified non-generative Decision Model interface (Spec §6.1)."""

    def __init__(
        self,
        *,
        judge_id: JudgeId,
        model: str,
        endpoint: str,
        api_key: str,
        max_concurrency: int = 8,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.judge_id = judge_id
        self.model = model
        self.endpoint = endpoint.strip()
        self.api_key = api_key.strip()
        self.semaphore = asyncio.Semaphore(min(max(1, max_concurrency), 8))
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._transport = transport

    def ensure_configured(self) -> None:
        if not self.endpoint:
            raise DecisionJudgeError(
                self.judge_id,
                503,
                f"Decision backend '{self.judge_id}' endpoint URL is not configured.",
            )
        if not self.api_key:
            raise DecisionJudgeError(
                self.judge_id,
                503,
                f"Decision backend '{self.judge_id}' API key is not configured.",
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_configured()
        async with self.semaphore:
            last_error: DecisionJudgeError | None = None
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self._transport,
            ) as client:
                for attempt in range(self.max_retries + 1):
                    try:
                        resp = await client.post(
                            self.endpoint,
                            headers=self._headers(),
                            json=payload,
                        )
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        last_error = DecisionJudgeError(
                            self.judge_id,
                            503,
                            f"Network/timeout error contacting '{self.judge_id}': {exc}",
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_base_delay * (2**attempt))
                            continue
                        raise last_error from exc

                    if resp.status_code in RETRYABLE_STATUS_CODES:
                        last_error = DecisionJudgeError(
                            self.judge_id,
                            503 if resp.status_code in {429, 503, 504} else 502,
                            f"Upstream HTTP {resp.status_code} from '{self.judge_id}': {resp.text[:300]}",
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_base_delay * (2**attempt))
                            continue
                        raise last_error

                    if resp.status_code >= 400:
                        raise DecisionJudgeError(
                            self.judge_id,
                            502,
                            f"Upstream HTTP {resp.status_code} from '{self.judge_id}': {resp.text[:300]}",
                        )

                    try:
                        body = resp.json()
                    except ValueError as exc:
                        raise DecisionJudgeError(
                            self.judge_id,
                            502,
                            f"Non-JSON response from '{self.judge_id}'.",
                        ) from exc

                    if not isinstance(body, dict):
                        raise DecisionJudgeError(
                            self.judge_id,
                            502,
                            f"Expected JSON object from '{self.judge_id}', got {type(body).__name__}.",
                        )
                    return body

            assert last_error is not None
            raise last_error

    @abstractmethod
    async def abinary(
        self,
        state: str | dict[str, Any],
        questions: Mapping[str, str],
        *,
        yes_desc: str = "Yes, relevant.",
        no_desc: str = "No, not relevant.",
    ) -> BinaryBatchResult:
        """Evaluate Equation 2 parallel independent binary questions over shared state."""

    @abstractmethod
    async def achoice(
        self,
        state: str | dict[str, Any],
        instruction: str,
        options: Mapping[str, str],
    ) -> ChoiceResult:
        """Evaluate Equation 3 single multiple-choice question (m <= 255) over shared state."""