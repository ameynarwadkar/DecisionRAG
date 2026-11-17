from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.auth import AuthContext, require_api_key
from dec_rag.config import Settings, get_settings
from dec_rag.database import get_db
from dec_rag.models import CategoryNode, DocumentPage, Namespace, Passage
from dec_rag.schemas import DocumentPageRead, DocumentUploadResponse, PassageRead
from dec_rag.services.ingestion import ingest_document, refresh_category_auto_description

router = APIRouter(prefix="/namespaces/{ns_id}/documents", tags=["documents"])


async def _ensure_namespace(db: AsyncSession, ns_id: UUID) -> Namespace:
    ns = (await db.execute(select(Namespace).where(Namespace.id == ns_id))).scalar_one_or_none()
    if ns is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Namespace not found.")
    return ns


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    ns_id: UUID,
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    category_description: str | None = Form(default=None),
    merge_part_suffix: bool = Form(default=False),
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DocumentUploadResponse:
    auth.ensure_namespace_access(ns_id)
    await _ensure_namespace(db, ns_id)

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        page, merged = await ingest_document(
            db,
            namespace_id=ns_id,
            filename=file.filename or "document.txt",
            raw_bytes=raw_bytes,
            content_type=file.content_type,
            category_name=category,
            category_description=category_description,
            merge_part_suffix=merge_part_suffix,
            max_passage_chars=settings.MAX_PASSAGE_CHARS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentUploadResponse(
        page=DocumentPageRead.model_validate(page),
        merged_into_existing=merged,
    )


@router.get("", response_model=list[DocumentPageRead])
async def list_documents(
    ns_id: UUID,
    category_id: UUID | None = Query(default=None),
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentPageRead]:
    auth.ensure_namespace_access(ns_id)
    await _ensure_namespace(db, ns_id)

    stmt = (
        select(DocumentPage)
        .where(DocumentPage.namespace_id == ns_id)
        .order_by(DocumentPage.doc_order.asc())
    )
    if category_id is not None:
        stmt = stmt.where(DocumentPage.category_id == category_id)
    pages = list((await db.execute(stmt)).scalars().all())
    return [DocumentPageRead.model_validate(p) for p in pages]


@router.get("/{page_id}", response_model=DocumentPageRead)
async def get_document_page(
    ns_id: UUID,
    page_id: UUID,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> DocumentPageRead:
    auth.ensure_namespace_access(ns_id)
    page = (
        await db.execute(
            select(DocumentPage).where(
                DocumentPage.id == page_id,
                DocumentPage.namespace_id == ns_id,
            )
        )
    ).scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document page not found.")
    return DocumentPageRead.model_validate(page)


@router.get("/{page_id}/passages", response_model=list[PassageRead])
async def list_page_passages(
    ns_id: UUID,
    page_id: UUID,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> list[PassageRead]:
    auth.ensure_namespace_access(ns_id)
    page = (
        await db.execute(
            select(DocumentPage).where(
                DocumentPage.id == page_id,
                DocumentPage.namespace_id == ns_id,
            )
        )
    ).scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document page not found.")

    passages = list(
        (
            await db.execute(
                select(Passage)
                .where(Passage.page_id == page_id)
                .order_by(Passage.passage_index.asc())
            )
        ).scalars().all()
    )
    return [PassageRead.model_validate(p) for p in passages]


@router.delete("/{page_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_page(
    ns_id: UUID,
    page_id: UUID,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> Response:
    auth.ensure_namespace_access(ns_id)
    page = (
        await db.execute(
            select(DocumentPage).where(
                DocumentPage.id == page_id,
                DocumentPage.namespace_id == ns_id,
            )
        )
    ).scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document page not found.")

    parent_cat_id = page.category_id
    await db.delete(page)
    await db.flush()

    if parent_cat_id is not None:
        cat = (
            await db.execute(select(CategoryNode).where(CategoryNode.id == parent_cat_id))
        ).scalar_one_or_none()
        if cat is not None:
            await refresh_category_auto_description(db, cat)

    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)