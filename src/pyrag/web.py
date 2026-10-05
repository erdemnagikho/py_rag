from __future__ import annotations

import json
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, final

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .embeddings import Embedder
from .ingest import IMAGE_DESCRIBE_PROMPT, IMAGE_SUFFIXES, TEXT_SUFFIXES, Ingestor
from .llm import ChatClient, Message
from .query import build_user_message, initial_messages, retrieve, rewrite_query
from .stores.factory import make_store

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    k: int | None = None


def create_app() -> FastAPI:
    cfg = load_config()
    cfg.ensure_dirs()

    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    chat = ChatClient.from_config(cfg)
    ingestor = Ingestor(cfg, store, embedder, chat=chat)
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="pyrag", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.embedder = embedder
    app.state.chat = chat
    app.state.ingestor = ingestor
    app.state.lock = lock

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/files/{filename}")
    def serve_file(filename: str) -> FileResponse:
        if (
            "/" in filename
            or "\\" in filename
            or filename in ("", ".", "..")
            or filename. startswith(".")
        ):
            raise HTTPException(status_code=400, detail="invalid filename")

        target = cfg.processed_dir / filename
        try:
            resolved = target.resolve()
            resolved.relative_to(cfg.processed_dir.resolve())
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="invalid path") from exc

        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(resolved)

    @app.post("/api/chat")
    def chat_endpoint(req: ChatRequest) -> StreamingResponse:
        question = req.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="empty message")
        top_k = req.k if req.k is not None else cfg.top_k

        def event_stream():
            try:
                raw_history: list[Message] = [
                    {"role": t.role, "content": t.content} for t in req.history
                ]

                search_query = (
                    rewrite_query(chat, raw_history, question)
                    if raw_history else question
                )

                if search_query != question:
                    yield _sse("rewrite", {"query": search_query})

                with lock:
                    ctx = retrieve(store, embedder, search_query, top_k)

                raw_history: list[Message] = [
                    {"role": t.role, "content": t.content} for t in req.history
                ]
                messages: list[Message] = initial_messages(cfg.system_prompt)
                messages.extend(raw_history)
                messages.append(
                    {"role": "user", "content": build_user_message(
                        question, ctx)}
                )

                for piece in chat.stream(messages):
                    yield _sse("token", {"text": piece})

                yield _sse("sources", {"hits": _hits_payload(ctx.hits)})
                yield _sse("done", {})

            except Exception as exc:
                log.exception("chat endpoint failed")
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    def _handle_upload(
            file: UploadFile,
            allowed: set[str],
            label: str,
            description: str | None = None,
    ) -> dict[str, Any]:
        name = Path(file.filename or "").name
        if not name:
            raise HTTPException(status_code=400, detail="missing filename")
        suffix = Path(name).suffix.lower()
        if suffix not in allowed:
            raise HTTPException(
                status_code=400, detail=f"unsupported {label} type: {suffix or '(no extension)'}")

        target = cfg.documents_dir / name
        if target.exists():
            raise HTTPException(
                status_code=409, detail=f"{name} exists in documents")

        prior_source_path = str(target.resolve())
        with lock:
            store.delete_document(prior_source_path)
        prior_processed = cfg.processed_dir / name
        if prior_processed.exists():
            prior_processed.unlink()

        data = file.file.read()
        target.write_bytes(data)

        with lock:
            try:
                ingestor.ingest_file(target, description=description)
            except Exception as exc:
                log.exception("ingest failed for %s", target)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {"filename": name, "bytes": len(data)}

    @app.post("/api/upload")
    def upload(file: UploadFile) -> dict[str, Any]:
        return _handle_upload(file, TEXT_SUFFIXES, "document")

    @app.post("/api/upload-image")
    def upload_image(
        file: UploadFile,
        description: str = Form(""),
    ) -> dict[str, Any]:
        text = description.strip() or None
        return _handle_upload(file, IMAGE_SUFFIXES, "image", description=text)

    @app.post("/api/describe-image")
    def describe_image(file: UploadFile) -> dict[str, str]:
        name = Path(file.filename or "").name
        if not name:
            raise HTTPException(status_code=400, detail="missing filename")
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise HTTPException(
                status_code=400, detail="unsupported image type")

        data = file.file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            description = chat.describe(
                cfg.vision_model, IMAGE_DESCRIBE_PROMPT, tmp_path)
        except Exception as exc:
            log.exception("describe failed for %s", name)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return {"description": description}

    return app


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _hits_payload(hits: list) -> list[dict[str, Any]]:
    return [
        {
            "filename": Path(h.source_path).name,
            "chunk_index": h.chunk_index,
            "score": round(float(h.score), 3),
        }
        for h in hits
    ]


app = create_app()
