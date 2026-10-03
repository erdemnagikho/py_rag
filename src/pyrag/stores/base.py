from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class StoredChunk:
    index: int
    text: str
    embedding: list[float]
    metadata: dict[str, Any]

@dataclass
class SearchHit:
    source_path: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]

class VectorStore(ABC):
    @abstractmethod
    def has_document(self, source_path: str, content_hash: str) -> bool:
        """Return True if this exact (path, hash) is already stored."""

    @abstractmethod
    def upsert_document(
        self,
        source_path: str,
        content_hash: str,
        chunks: list[StoredChunk],
    ) -> None:
        """Replace any existing chunks for `source_path` with the given ones."""

    @abstractmethod
    def search(self, query_text: str, query_embedding: list[float], k: int) -> list[SearchHit]:
        """Return top-k chunks for the query."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources."""
    
    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
        
        
