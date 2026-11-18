from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

JudgeId = Literal["clef", "openrouter_jev", "openrouter_luna", "openai_luna"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_env_file() -> Path | None:
    root_env = PROJECT_ROOT / ".env"
    if root_env.is_file():
        return root_env
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        return cwd_env
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server & Database
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite+aiosqlite:///./dec_rag.db"
    DEV_BYPASS_AUTH: bool = False

    # Decision Judge Execution Limits (App. F: <=8 concurrent requests per query)
    DEFAULT_JUDGE: JudgeId = "openrouter_jev"
    DECISION_CONCURRENCY: int = Field(default=8, ge=1, le=8)
    DECISION_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0.0)
    DECISION_MAX_RETRIES: int = Field(default=3, ge=0)

    # 1. Colab Cloudflare Clef-Flash 9B
    CLEF_API_URL: str = ""
    CLEF_API_KEY: str = ""
    CLEF_MODEL: str = "Cloudflare/clef-flash"

    # 2. OpenRouter Decisions API
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/alpha/decisions"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_JEV_MODEL: str = "typesafe/jev-1.13"
    OPENROUTER_LUNA_MODEL: str = "openai/gpt-6-luna-decisions"

    # 3. OpenAI Direct Decisions API
    OPENAI_DECISIONS_API_URL: str = "https://api.openai.com/v1/decisions"
    OPENAI_API_KEY: str = ""
    OPENAI_LUNA_MODEL: str = "gpt-6-luna"

    # Stage 4 Answer Generator (App. C.1) — supports OpenAI, Azure OpenAI (azure/<deployment>), etc.
    GENERATOR_MODEL: str = "openai/gpt-4o-mini"
    GENERATOR_API_KEY: str = ""
    GENERATOR_API_BASE: str = ""
    GENERATOR_API_VERSION: str = ""

    # Azure OpenAI convenience variables (accepted interchangeably)
    AZURE_API_KEY: str = ""
    AZURE_API_BASE: str = ""
    AZURE_API_VERSION: str = "2024-10-21"
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = ""

    # Offline Clustering for >650 Flat Pages (App. A.4 & B.5)
    CLUSTER_EMBEDDING_MODEL: str = "text-embedding-3-small"
    CLUSTER_SUMMARY_MODEL: str = "openai/gpt-4o-mini"

    # Parser & Demo Seeding
    MAX_PASSAGE_CHARS: int = Field(default=1200, ge=200)
    SEED_DEMO_CORPUS: bool = True


_cached_settings: Settings | None = None
_cached_env_mtime_ns: int | None = None
_cached_env_path: Path | None = None


def get_settings() -> Settings:
    """Load Settings from .env, automatically hot-reloading whenever .env changes on disk."""
    global _cached_settings, _cached_env_mtime_ns, _cached_env_path
    env_path = resolve_env_file()
    mtime_ns = env_path.stat().st_mtime_ns if env_path is not None else None

    if (
        _cached_settings is None
        or env_path != _cached_env_path
        or mtime_ns != _cached_env_mtime_ns
    ):
        _cached_settings = Settings(_env_file=env_path)
        _cached_env_path = env_path
        _cached_env_mtime_ns = mtime_ns

    return _cached_settings
