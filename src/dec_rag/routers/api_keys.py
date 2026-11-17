from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.auth import AuthContext, generate_api_key, require_api_key
from dec_rag.database import get_db
from dec_rag.models import ApiKey, Namespace
from dec_rag.schemas import ApiKeyCreate, ApiKeyCreatedResponse, ApiKeyRead

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


@router.post("", response_model=ApiKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key_endpoint(
    body: ApiKeyCreate,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> ApiKeyCreatedResponse:
    if auth.namespace_id is not None:
        if body.namespace_id != auth.namespace_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Namespace-scoped API keys can only create keys for their own namespace.",
            )

    if body.namespace_id is not None:
        ns = (
            await db.execute(select(Namespace).where(Namespace.id == body.namespace_id))
        ).scalar_one_or_none()
        if ns is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Namespace not found.")

    raw_key, key_prefix, key_hash = generate_api_key()
    record = ApiKey(
        name=body.name.strip(),
        key_prefix=key_prefix,
        key_hash=key_hash,
        namespace_id=body.namespace_id,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return ApiKeyCreatedResponse(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        namespace_id=record.namespace_id,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
        api_key=raw_key,
    )


@router.get("", response_model=list[ApiKeyRead])
async def list_api_keys(
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> list[ApiKeyRead]:
    stmt = select(ApiKey).order_by(ApiKey.created_at.desc())
    if auth.namespace_id is not None:
        stmt = stmt.where(ApiKey.namespace_id == auth.namespace_id)
    records = list((await db.execute(stmt)).scalars().all())
    return [ApiKeyRead.model_validate(r) for r in records]


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: UUID,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> Response:
    record = (await db.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found.")
    if auth.namespace_id is not None and record.namespace_id != auth.namespace_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    await db.delete(record)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)