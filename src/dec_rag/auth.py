from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import secrets
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.config import Settings, get_settings
from dec_rag.database import get_db
from dec_rag.models import ApiKey


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Return (raw_key, key_prefix, key_hash)."""
    raw_key = f"dr_live_{secrets.token_urlsafe(24)}"
    key_prefix = raw_key[:12]
    return raw_key, key_prefix, hash_api_key(raw_key)


@dataclass(frozen=True)
class AuthContext:
    api_key_id: UUID | None
    namespace_id: UUID | None
    bypassed: bool = False

    def ensure_namespace_access(self, target_namespace_id: UUID) -> None:
        if self.namespace_id is not None and self.namespace_id != target_namespace_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key is not authorized for this namespace.",
            )


def _extract_token(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            return token
    return None


async def verify_api_key_token(
    raw_token: str | None,
    db: AsyncSession,
    settings: Settings,
) -> AuthContext:
    if raw_token is None:
        if settings.DEV_BYPASS_AUTH:
            return AuthContext(api_key_id=None, namespace_id=None, bypassed=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Supply X-API-Key or Authorization: Bearer <key>.",
        )

    token_hash = hash_api_key(raw_token)
    stmt = select(ApiKey).where(ApiKey.key_hash == token_hash)
    record = (await db.execute(stmt)).scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    record.last_used_at = datetime.now(UTC)
    await db.commit()
    return AuthContext(
        api_key_id=record.id,
        namespace_id=record.namespace_id,
        bypassed=False,
    )


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthContext:
    raw_token = _extract_token(x_api_key, authorization)
    return await verify_api_key_token(raw_token, db, settings)