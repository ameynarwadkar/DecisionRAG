import asyncio
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

MAX_PREDICATES_PER_REQUEST = 400


class OpenAILunaDecisionJudge(DecisionJudge):
    """OpenAI Direct Decisions API (/v1/decisions, model='gpt-6-luna') adapter."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str = "gpt-6-luna",
        max_concurrency: int = 8,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            judge_id="openai_luna",
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            max_concurrency=max_concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            transport=transport,
        )

    async def _abinary_chunk(
        self,
        state: str | dict[str, Any],
        chunk_items: list[tuple[str, str]],
    ) -> BinaryBatchResult:
        payload = {
            "model": self.model,
            "input": state,
            "questions": [
                {"type": "predicate", "name": qid, "instructions": q_text}
                for qid, q_text in chunk_items
            ],
        }
        data = await self._post_json(payload)
        raw_container = data.get("answers") or data.get("decisions") or data.get("output")
        lookup: dict[str, Any] = {}

        if isinstance(raw_container, list):
            for item in raw_container:
                if isinstance(item, Mapping) and "name" in item:
                    lookup[str(item["name"])] = item.get("probability")
        elif isinstance(raw_container, Mapping):
            for k, v in raw_container.items():
                lookup[str(k)] = v.get("probability") if isinstance(v, Mapping) else v
        else:
            raise DecisionJudgeError(
                self.judge_id,
                502,
                "OpenAI Decisions response missing 'answers'/'decisions'/'output'.",
            )

        probs: dict[str, float] = {}
        for qid, _ in chunk_items:
            if qid not in lookup:
                raise DecisionJudgeError(
                    self.judge_id,
                    502,
                    f"OpenAI Decisions response missing predicate '{qid}'.",
                )
            probs[qid] = clamp_probability(lookup[qid], judge_id=self.judge_id, field_name=qid)

        tokens, cost = extract_usage(data)
        return BinaryBatchResult(
            probabilities=probs,
            requests_made=1,
            input_tokens=tokens,
            cost_usd=cost,
        )

    async def abinary(
        self,
        state: str | dict[str, Any],
        questions: Mapping[str, str],
        *,
        yes_desc: str = "Yes, relevant.",
        no_desc: str = "No, not relevant.",
    ) -> BinaryBatchResult:
        del yes_desc, no_desc
        if not questions:
            return BinaryBatchResult(probabilities={}, requests_made=0, input_tokens=0, cost_usd=0.0)

        items = list(questions.items())
        chunks = [
            items[i : i + MAX_PREDICATES_PER_REQUEST]
            for i in range(0, len(items), MAX_PREDICATES_PER_REQUEST)
        ]
        results = await asyncio.gather(*(self._abinary_chunk(state, chunk) for chunk in chunks))

        merged_probs: dict[str, float] = {}
        total_reqs = 0
        total_tokens = 0
        total_cost = 0.0
        for res in results:
            merged_probs.update(res.probabilities)
            total_reqs += res.requests_made
            total_tokens += res.input_tokens
            total_cost += res.cost_usd

        return BinaryBatchResult(
            probabilities=merged_probs,
            requests_made=total_reqs,
            input_tokens=total_tokens,
            cost_usd=total_cost,
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
            "input": state,
            "questions": [
                {
                    "type": "choice",
                    "name": "best",
                    "instructions": instruction,
                    "choices": [
                        {"value": opt_id, "description": label}
                        for opt_id, label in options.items()
                    ],
                }
            ],
        }
        data = await self._post_json(payload)
        raw_container = data.get("answers") or data.get("decisions") or data.get("output")
        best_entry: Any = None

        if isinstance(raw_container, list):
            for item in raw_container:
                if isinstance(item, Mapping) and item.get("name") == "best":
                    best_entry = item
                    break
        elif isinstance(raw_container, Mapping):
            best_entry = raw_container.get("best")

        if not isinstance(best_entry, Mapping):
            raise DecisionJudgeError(
                self.judge_id,
                502,
                "OpenAI Decisions choice response missing 'best' entry.",
            )

        raw_probs_field = best_entry.get("probabilities", best_entry)
        raw_map: dict[str, Any] = {}
        if isinstance(raw_probs_field, list):
            for item in raw_probs_field:
                if isinstance(item, Mapping) and "value" in item:
                    raw_map[str(item["value"])] = item.get("probability")
        elif isinstance(raw_probs_field, Mapping):
            raw_map = dict(raw_probs_field)
        else:
            raise DecisionJudgeError(
                self.judge_id,
                502,
                "Malformed 'probabilities' in OpenAI Decisions choice response.",
            )

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