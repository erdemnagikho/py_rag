from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .chunking import chunk_text
from .config import Config
from . embeddings import Embedder
from .stores.base import StoredChunk, VectorStore

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _is_under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())

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
            log.info("Unchanged %s, skipping embed: %s", path, content_hash)
            self._move_to_processed(path)
            return

        text = data.decode("utf-8", errors="replace")
        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        if not chunks:
            log.warning("No content to ingest in %s", path.name)
            self._move_to_processed(path)
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
        self._move_to_processed(path)


    def _move_to_processed(self, path: Path) -> None:
        processed = self.config.processed_dir
        processed.mkdir(parents=True, exist_ok=True)
        target = processed / path.name
        if target.exists():
            stem, suffix = path.stem, path.suffix
            ts = time.strftime("%Y%m%d-%H%M%S")
            target = processed / f"{stem}.{ts}{suffix}"
        shutil.move(str(path), str(target))
        log.info("Moved -> %s", target)

class _DebounceHandler(FileSystemEventHandler):
    def __init__(self, ingestor: Ingestor, debounce_seconds: float = 0.75) -> None:
        self._ingestor = ingestor
        self._debounce = debounce_seconds
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._processed_dir = ingestor.config.processed_dir
    
    def _schedule(self, raw_path: Path) -> None:
        path = Path(raw_path)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return
        
        if _is_under(path, self._processed_dir):
            return

        with self._lock:
            existing = self._timers.pop(raw_path, None)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(self._debounce, self._fire, args=(raw_path,))
            timer.daemon = True
            self._timers[raw_path] = timer
            timer.start()

    def _fire(self, raw_path: str) -> None:
        with self._lock:
            self._timers.pop(raw_path, None)
        
        path = Path(raw_path)
        if not path.exists():
            return
        
        try:
            self._ingestor.ingest_file(path)
        except Exception:
            log.exception("Failed to ingest %s after debounce timeout", path)
    
    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(str(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(str(event.src_path))
    
    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(str(event.src_path))

def initial_scan(ingestor: Ingestor) -> None:
    docs = ingestor.config.documents_dir
    processed = ingestor.config.processed_dir
    for path in sorted(docs.iterdir()):
        if path.is_dir():
            continue
        if _is_under(path, processed):
            continue
        try:
            ingestor.ingest_file(path)
        except Exception as e:
            log.exception("Failed to ingest %s during initial scan", path)

def watch(ingestor: Ingestor) -> None:
    ingestor.config.ensure_dirs()
    docs = ingestor.config.documents_dir

    log.info("Initial scan of %s", docs)
    initial_scan(ingestor)

    handler = _DebounceHandler(ingestor)
    observer = Observer()
    observer.schedule(handler, str(docs), recursive=False)
    observer.start()
    log.info("Watching for changes in %s (Ctrl+C to stop)", docs)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping watcher")
    finally:
        observer.stop()
        observer.join()
        log.info("Stopped")
    
        

    