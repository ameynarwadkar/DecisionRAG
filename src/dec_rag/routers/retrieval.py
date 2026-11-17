from collections.abc import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.auth import AuthContext, require_api_key
from dec_rag.config import JudgeId, Settings, get_settings
from dec_rag.database import get_db
from dec_rag.models import ConversationSession, Namespace
from dec_rag.schemas import (
    ChatRequest,
    ChatResponse,
    CompareJudgesRequest,
    CompareJudgesResponse,
    DialogueTurn,
    JudgeComparisonEntry,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    apply_runtime_credentials,
)
from dec_rag.services.generator import generate_grounded_answer
from dec_rag.services.judges import DecisionJudge, create_judge
from dec_rag.services.paper_pipeline import navigate_corpus
from dec_rag.services.progress import progress_broker

JudgeFactory = Callable[[JudgeId, Settings], DecisionJudge]

router = APIRouter(prefix="/namespaces/{ns_id}", tags=["retrieval"])


def get_judge_factory() -> JudgeFactory:
    def _factory(judge_id: JudgeId, settings: Settings) -> DecisionJudge:
        return create_judge(judge_id, settings)

    return _factory


async def _load_namespace(db: AsyncSession, ns_id: UUID) -> Namespace:
    ns = (await db.execute(select(Namespace).where(Namespace.id == ns_id))).scalar_one_or_none()
    if ns is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Namespace not found.")
    return ns


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_passages(
    ns_id: UUID,
    body: RetrieveRequest,
    x_run_id: str | None = Header(default=None, alias="X-Run-Id"),
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    judge_factory: JudgeFactory = Depends(get_judge_factory),
) -> RetrieveResponse:
    auth.ensure_namespace_access(ns_id)
    ns = await _load_namespace(db, ns_id)

    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    judge_id = body.judge or effective_settings.DEFAULT_JUDGE
    judge = judge_factory(judge_id, effective_settings)
    try:
        nav_out = await navigate_corpus(
            db,
            namespace=ns,
            judge=judge,
            query=body.query,
            scenario=body.scenario,
            overrides=body,
            on_stage=lambda step: progress_broker.publish(x_run_id, step),
        )
        return RetrieveResponse(
            namespace_id=ns.id,
            query=body.query,
            passages=nav_out.passages,
            telemetry=nav_out.telemetry,
        )
    finally:
        await progress_broker.publish(x_run_id, "done", done=True)


@router.post("/query", response_model=QueryResponse)
async def query_namespace(
    ns_id: UUID,
    body: QueryRequest,
    x_run_id: str | None = Header(default=None, alias="X-Run-Id"),
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    judge_factory: JudgeFactory = Depends(get_judge_factory),
) -> QueryResponse:
    auth.ensure_namespace_access(ns_id)
    ns = await _load_namespace(db, ns_id)

    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    gen_key = body.runtime_credentials.generator_api_key if body.runtime_credentials else None
    judge_id = body.judge or effective_settings.DEFAULT_JUDGE
    judge = judge_factory(judge_id, effective_settings)
    try:
        nav_out = await navigate_corpus(
            db,
            namespace=ns,
            judge=judge,
            query=body.query,
            scenario=body.scenario,
            overrides=body,
            on_stage=lambda step: progress_broker.publish(x_run_id, step),
        )
        await progress_broker.publish(x_run_id, "answer")
        gen = await generate_grounded_answer(
            passages=nav_out.passages,
            settings=effective_settings,
            query=body.query,
            scenario=body.scenario,
            system_prompt=ns.system_prompt,
            top_k_context=body.top_k_context,
            api_key=gen_key,
        )
        return QueryResponse(
            namespace_id=ns.id,
            query=body.query,
            answer=gen.answer,
            passages=nav_out.passages,
            telemetry=nav_out.telemetry,
            generator_latency_ms=gen.latency_ms,
        )
    finally:
        await progress_broker.publish(x_run_id, "done", done=True)


@router.post("/chat", response_model=ChatResponse)
async def chat_namespace(
    ns_id: UUID,
    body: ChatRequest,
    x_run_id: str | None = Header(default=None, alias="X-Run-Id"),
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    judge_factory: JudgeFactory = Depends(get_judge_factory),
) -> ChatResponse:
    auth.ensure_namespace_access(ns_id)
    ns = await _load_namespace(db, ns_id)

    session: ConversationSession | None = None
    if body.session_id is not None:
        session = (
            await db.execute(
                select(ConversationSession).where(
                    ConversationSession.id == body.session_id,
                    ConversationSession.namespace_id == ns.id,
                )
            )
        ).scalar_one_or_none()
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation session not found in this namespace.",
            )
    else:
        session = ConversationSession(
            namespace_id=ns.id,
            turns_json=[t.model_dump() for t in (body.turns or [])],
            notepad_page_ids=[],
        )
        db.add(session)
        await db.flush()

    prior_turns: list[DialogueTurn] = (
        list(body.turns)
        if (body.turns is not None and body.session_id is None)
        else [DialogueTurn.model_validate(t) for t in session.turns_json]
    )
    active_turns = prior_turns + [DialogueTurn(role="user", content=body.message)]

    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    gen_key = body.runtime_credentials.generator_api_key if body.runtime_credentials else None
    judge_id = body.judge or effective_settings.DEFAULT_JUDGE
    judge = judge_factory(judge_id, effective_settings)
    try:
        nav_out = await navigate_corpus(
            db,
            namespace=ns,
            judge=judge,
            dialogue_turns=active_turns,
            notepad_page_ids=session.notepad_page_ids,
            overrides=body,
            on_stage=lambda step: progress_broker.publish(x_run_id, step),
        )
        await progress_broker.publish(x_run_id, "answer")
        gen = await generate_grounded_answer(
            passages=nav_out.passages,
            settings=effective_settings,
            dialogue_turns=active_turns,
            system_prompt=ns.system_prompt,
            top_k_context=body.top_k_context,
            api_key=gen_key,
        )

        updated_turns = active_turns + [DialogueTurn(role="assistant", content=gen.answer)]
        session.turns_json = [t.model_dump() for t in updated_turns]
        session.notepad_page_ids = list(nav_out.updated_notepad_page_ids)
        await db.commit()
        await db.refresh(session)

        return ChatResponse(
            namespace_id=ns.id,
            session_id=session.id,
            answer=gen.answer,
            notepad_page_ids=session.notepad_page_ids,
            passages=nav_out.passages,
            telemetry=nav_out.telemetry,
            generator_latency_ms=gen.latency_ms,
        )
    finally:
        await progress_broker.publish(x_run_id, "done", done=True)


@router.post("/compare-judges", response_model=CompareJudgesResponse)
async def compare_judges_endpoint(
    ns_id: UUID,
    body: CompareJudgesRequest,
    x_run_id: str | None = Header(default=None, alias="X-Run-Id"),
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    judge_factory: JudgeFactory = Depends(get_judge_factory),
) -> CompareJudgesResponse:
    auth.ensure_namespace_access(ns_id)
    ns = await _load_namespace(db, ns_id)

    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    entries: list[JudgeComparisonEntry] = []
    try:
        for j_id in body.judges:
            judge = judge_factory(j_id, effective_settings)
            nav_out = await navigate_corpus(
                db,
                namespace=ns,
                judge=judge,
                query=body.query,
                scenario=body.scenario,
                overrides=body,
                on_stage=lambda step: progress_broker.publish(x_run_id, step),
            )
            entries.append(
                JudgeComparisonEntry(
                    judge=j_id,
                    passages=nav_out.passages,
                    telemetry=nav_out.telemetry,
                )
            )

        return CompareJudgesResponse(
            namespace_id=ns.id,
            query=body.query,
            results=entries,
        )
    finally:
        await progress_broker.publish(x_run_id, "done", done=True)
