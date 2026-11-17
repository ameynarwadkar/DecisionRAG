import asyncio
from dataclasses import dataclass, field
import json
import time


@dataclass
class _RunState:
    step: str = ""
    done: bool = False
    updated_at: float = field(default_factory=time.monotonic)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class RunProgressBroker:
    """Ephemeral in-memory tracker broadcasting live pipeline step names over SSE."""

    def __init__(self, ttl_seconds: float = 120.0) -> None:
        self._runs: dict[str, _RunState] = {}
        self._ttl = ttl_seconds

    def _prune_stale(self) -> None:
        now = time.monotonic()
        stale = [
            rid
            for rid, st in self._runs.items()
            if (now - st.updated_at) > self._ttl
        ]
        for rid in stale:
            self._runs.pop(rid, None)

    def _get_or_create(self, run_id: str) -> _RunState:
        self._prune_stale()
        st = self._runs.get(run_id)
        if st is None:
            st = _RunState()
            self._runs[run_id] = st
        return st

    async def publish(self, run_id: str | None, step: str, *, done: bool = False) -> None:
        if not run_id:
            return
        st = self._get_or_create(run_id)
        async with st.condition:
            st.step = step
            st.done = done
            st.updated_at = time.monotonic()
            st.condition.notify_all()

    async def subscribe(self, run_id: str):
        st = self._get_or_create(run_id)
        last_sent = ""
        try:
            while True:
                async with st.condition:
                    if st.step == last_sent and not st.done:
                        try:
                            await asyncio.wait_for(st.condition.wait(), timeout=25.0)
                        except TimeoutError:
                            yield ": keepalive\n\n"
                            continue

                    current_step = st.step
                    is_done = st.done

                if current_step and current_step != last_sent:
                    last_sent = current_step
                    payload = json.dumps({"step": current_step, "done": is_done})
                    yield f"data: {payload}\n\n"

                if is_done:
                    if last_sent != "done":
                        yield f"data: {json.dumps({'step': 'done', 'done': True})}\n\n"
                    break
        finally:
            if st.done:
                self._runs.pop(run_id, None)


progress_broker = RunProgressBroker()
