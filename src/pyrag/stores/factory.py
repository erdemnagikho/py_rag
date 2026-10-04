from __future__ import annotations

from ..config import Config
from .base import VectorStore
from .postgres import PostgresStore

def make_store(config: Config) -> VectorStore:
    kind = config.vector_store.lower()
    if kind == "postgres":
        return PostgresStore(dsn=config.pg_dsn)
    if kind == "weaviate":
        from .weaviate import WeaviateStore
        return WeaviateStore(
            host=config.weaviate_host,
            http_port=config.weaviate_http_port,
            grpc_port=config.weaviate_grpc_port,
            collection=config.weaviate_collection,
        )
    raise ValueError(f"Unknown VECTOR_STORE: {config.vector_store!r}")
    