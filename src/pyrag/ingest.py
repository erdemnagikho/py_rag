from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .chunking import chunk_text
from .config import Config
from . embeddings import Embedder
from .stores.base import StoredChunk, VectorStore

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class Ingestor:
    def __init__(
            self,
            config: Config,
            store: VectorStore,
            embedder: Embedder,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder

    def ingest_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            log.info("Skipping %s, unsupported suffix %s", path, suffix)
            return

        try:
            data = path.read_bytes()
        except FileNotFoundError:
            log.warning("File vanished before read: %s", path)
            return

        content_hash = _hash_bytes(data)
        source_path = str(path.resolve())

        if self.store.has_document(source_path, content_hash):
            log.info("Skipping %s, already ingested: %s", path, content_hash)
            return

        text = data.decode("utf-8", errors="replace")
        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        if not chunks:
            log.warning("No chunks created from %s", path.name)
            return

        log.info("Embedding %d chunks from %s", len(chunks), path.name)
        embeddings = self.embedder.embed([c.text for c in chunks])

        stored = [
            StoredChunk(
                index= c.index,
                text= c.text,
                embedding= emb,
                metadata= {},
            )
            for c, emb in zip(chunks, embeddings, strict=True)
        ]

        self.store.upsert_document(source_path, content_hash, stored)
        log.info("Ingested %s (%d chunks)", path.name, len(stored))


    