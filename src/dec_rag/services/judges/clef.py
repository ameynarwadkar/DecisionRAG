from collections.abc import Mapping
from typing import Any

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


def resolve_clef_endpoint(raw_url: str) -> str:
    cleaned = raw_url.strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1/systemone"):
        return cleaned
    return f"{cleaned}/v1/systemone"


def _as_state_object(state: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(state, str):
        return {"context": state}
    return dict(state)


class ClefDecisionJudge(DecisionJudge):
    """Colab Cloudflare Clef-Flash 9B (/v1/systemone) adapter."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "Cloudflare/clef-flash",
        max_concurrency: int = 8,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            judge_id="clef",
            model=model,
            endpoint=resolve_clef_endpoint(base_url),
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
            "state": _as_state_object(state),
            "questions": {
                qid: {
                    "type": "choice",
                    "instructions": q_text,
                    "criteria": {"yes": yes_desc, "no": no_desc},
                }
                for qid, q_text in questions.items()
            },
        }
        data = await self._post_json(payload)
        decisions = data.get("decisions")
        if not isinstance(decisions, Mapping):
            raise DecisionJudgeError(self.judge_id, 502, "Clef response missing 'decisions' object.")

        probs: dict[str, float] = {}
        for qid in questions:
            entry = decisions.get(qid)
            if not isinstance(entry, Mapping):
                raise DecisionJudgeError(self.judge_id, 502, f"Clef response missing decision for '{qid}'.")
            raw_yes = entry.get("yes")
            if raw_yes is None and isinstance(entry.get("probabilities"), Mapping):
                raw_yes = entry["probabilities"].get("yes")
            probs[qid] = clamp_probability(raw_yes, judge_id=self.judge_id, field_name=qid)

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
        decisions = data.get("decisions")
        if not isinstance(decisions, Mapping) or "best" not in decisions:
            raise DecisionJudgeError(self.judge_id, 502, "Clef choice response missing 'decisions.best'.")

        best_entry = decisions["best"]
        if isinstance(best_entry, Mapping) and isinstance(best_entry.get("probabilities"), Mapping):
            raw_map = best_entry["probabilities"]
        elif isinstance(best_entry, Mapping):
            raw_map = best_entry
        else:
            raise DecisionJudgeError(self.judge_id, 502, "Malformed 'decisions.best' in Clef response.")

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