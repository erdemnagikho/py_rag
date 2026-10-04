from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default

def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))

def _read_prompt_file(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    contents = path.read_text(encoding="utf-8").strip()
    return contents or None


@dataclass(frozen=True)
class Config:
    openai_base_url: str | None
    openai_api_key: str
    chat_model: str
    embed_model: str
    embed_dim: int
    vector_store: str
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_db: str
    documents_dir: str
    processed_dir: str
    chunk_size: int
    chunk_overlap: int
    system_prompt: str | None
    top_k: int
    weaviate_host: str
    weaviate_http_port: int
    weaviate_grpc_port: int
    weaviate_collection: str

    def ensure_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


def load_config() -> Config:
    base_url_raw = os.getenv("OPENAI_BASE_URL", "").strip()
    return Config(
        openai_base_url=base_url_raw or None,
        openai_api_key=_env("OPENAI_API_KEY", "unused-local"),
        chat_model=_env("CHAT_MODEL", "gemma3:latest"),
        embed_model=_env("EMBED_MODEL", "nomic-embed-text"),
        embed_dim=_env_int("EMBED_DIM", 768),
        vector_store=_env("VECTOR_STORE", "postgres"),
        pg_host=_env("POSTGRES_HOST", "localhost"),
        pg_port=_env_int("POSTGRES_PORT", 5432),
        pg_user=_env("POSTGRES_USER", "pyrag"),
        pg_password=_env("POSTGRES_PASSWORD", "pyrag"),
        pg_db=_env("POSTGRES_DB", "pyrag"),
        documents_dir=Path(_env("DOCUMENTS_DIR", "./documents")),
        processed_dir=Path(_env("PROCESSED_DIR", "./documents/processed")),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 5),
        system_prompt=_read_prompt_file(os.getenv("SYSTEM_PROMPT_FILE")),
        weaviate_host=_env("WEAVIATE_HOST", "localhost"),
        weaviate_http_port=_env_int("WEAVIATE_HTTP_PORT", 8080),
        weaviate_grpc_port=_env_int("WEAVIATE_GRPC_PORT", 50051),
        weaviate_collection=_env("WEAVIATE_COLLECTION", "PyRagChunk"),
    )