import importlib
import importlib.util
from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from dec_rag.config import get_settings


class Base(DeclarativeBase):
    """Base declarative class for all DecisionRAG ORM models."""


def create_db_engine(database_url: str) -> AsyncEngine:
    engine = create_async_engine(database_url, future=True)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine.sync_engine, "connect")
        def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine: AsyncEngine = create_db_engine(get_settings().DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db(target_engine: AsyncEngine | None = None) -> None:
    if importlib.util.find_spec("dec_rag.models") is not None:
        importlib.import_module("dec_rag.models")
    active_engine = target_engine or engine
    async with active_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session