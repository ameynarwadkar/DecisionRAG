from collections.abc import Mapping
from typing import Any, Literal

import httpx

from dec_rag.services.judges.base import (
    BinaryBatchResult,
    ChoiceResult,
    DecisionJudge,
    DecisionJudgeError,
    clamp_probability,
    extract_usage,
    normalize_choice_probabilities,
)

OpenRouterJudgeId = Literal["openrouter_jev", "openrouter_luna"]


def _as_state_object(state: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(state, str):
        return {"context": state}
    return dict(state)


class OpenRouterDecisionJudge(DecisionJudge):
    """OpenRouter Decisions API (/api/alpha/decisions) adapter."""

    def __init__(
        self,
        *,
        judge_id: OpenRouterJudgeId,
        model: str,
        endpoint: str,
        api_key: str,
        max_concurrency: int = 8,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            judge_id=judge_id,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            transport=transport,
        )

    async def abinary(
        self,
        state: str | dict[str, Any],
        questions: Mapping[str, str],
        *,
        yes_desc: str = "Yes, relevant.",
        no_desc: str = "No, not relevant.",
    ) -> BinaryBatchResult:
        if not questions:
            return BinaryBatchResult(probabilities={}, requests_made=0, input_tokens=0, cost_usd=0.0)

        payload = {
            "model": self.model,
            "state": _as_state_object(state),
            "questions": {
                qid: {
                    "type": "noul",
                    "instructions": q_text,
                    "criteria": {"true": yes_desc, "false": no_desc},
                }
                for qid, q_text in questions.items()
            },
        }
        data = await self._post_json(payload)
        answers = data.get("answers")
        if not isinstance(answers, Mapping):
            raise DecisionJudgeError(self.judge_id, 502, "OpenRouter response missing 'answers' object.")

        probs: dict[str, float] = {}
        for qid in questions:
            entry = answers.get(qid)
            if isinstance(entry, Mapping):
                raw_val = entry.get("noul")
                if raw_val is None:
                    raw_val = entry.get("probability")
            else:
                raw_val = entry
            probs[qid] = clamp_probability(raw_val, judge_id=self.judge_id, field_name=qid)

        tokens, cost = extract_usage(data)
        return BinaryBatchResult(
            probabilities=probs,
            requests_made=1,
            input_tokens=tokens,
            cost_usd=cost,
        )

    async def achoice(
        self,
        state: str | dict[str, Any],
        instruction: str,
        options: Mapping[str, str],
    ) -> ChoiceResult:
        if not options or len(options) > 255:
            raise ValueError(f"achoice requires between 1 and 255 options, got {len(options)}.")

        payload = {
            "model": self.model,
            "state": _as_state_object(state),
            "questions": {
                "best": {
                    "type": "choice",
                    "instructions": instruction,
                    "criteria": dict(options),
                }
            },
        }
        data = await self._post_json(payload)
        answers = data.get("answers")
        if not isinstance(answers, Mapping) or "best" not in answers:
            raise DecisionJudgeError(self.judge_id, 502, "OpenRouter choice response missing 'answers.best'.")

        best_entry = answers["best"]
        if isinstance(best_entry, Mapping) and isinstance(best_entry.get("probabilities"), Mapping):
            raw_map = best_entry["probabilities"]
        elif isinstance(best_entry, Mapping):
            raw_map = best_entry
        else:
            raise DecisionJudgeError(self.judge_id, 502, "Malformed 'answers.best' in OpenRouter response.")

        option_ids = list(options.keys())
        probs, winner = normalize_choice_probabilities(raw_map, option_ids, judge_id=self.judge_id)
        tokens, cost = extract_usage(data)
        return ChoiceResult(
            probabilities=probs,
            winner=winner,
            requests_made=1,
            input_tokens=tokens,
            cost_usd=cost,
        )