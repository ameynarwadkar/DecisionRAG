from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from dec_rag.services.progress import progress_broker

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/{run_id}/events", include_in_schema=False)
async def stream_run_events(run_id: str) -> StreamingResponse:
    return StreamingResponse(
        progress_broker.subscribe(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
