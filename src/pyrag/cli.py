from __future__ import annotations

import itertools
import logging
import sys
import threading
import time
from pathlib import Path

import typer

from .config import load_config
from .embeddings import Embedder
from .ingest import Ingestor, watch
from .llm import ChatClient
from .query import build_user_message, initial_messages, retrieve
from .stores.factory import make_store

app = typer.Typer(add_completion=False, help="Local RAG CLI.")

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

def _build_ingestor() -> Ingestor:
    cfg = load_config()
    cfg.ensure_dirs()
    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    return Ingestor(cfg, store, embedder)

def _print_sources(hits: list) -> None:
    if not hits:
        return
    typer.echo("\nsources:")
    for h in hits:
        name = Path(h.source_path).name
        typer.echo(
            f"   - {name} (chunk {h.chunk_index}), score {h.score:.2f}" 
        )

class _Spinner:
    FRAMES = "|/-|\\"

    def __init__(self, message="thinking", interval: float = 0.1) -> None:
        self._message = message
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._clear = "\r" + " " * (len(message) + 5) + "\r"

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{frame} {self._message}...")
            sys.stdout.flush()
            time.sleep(self._interval)
    
    def start(self) -> None:
        self._thread.start()
    
    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join()
        sys.stdout.write(self._clear)
        sys.stdout.flush()


@app.command("ingest")
def ingest_cmd(path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    _setup_logging()
    ingestor = _build_ingestor()
    try:
        ingestor.ingest_file(path)
    finally:
        ingestor.store.close()

@app.command("scan")
def scan_cmd() -> None:
    _setup_logging()
    ingestor = _build_ingestor()
    try:
        docs = ingestor.config.documents_dir
        for path in sorted(docs.iterdir()):
            if path.is_file():
                ingestor.ingest_file(path)
    finally:
        ingestor.store.close()

@app.command("watch")
def watch_cmd() -> None:
    _setup_logging()
    ingestor = _build_ingestor()
    try:
        watch(ingestor)
    finally:
        ingestor.store.close()

@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="One-shot question to ask"),
    k: int = typer.Option(None, "--k", help="Override TOP_K"),
) -> None:
    _setup_logging()
    cfg = load_config()
    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    chat = ChatClient.from_config(cfg)
    top_k = k if k is not None else cfg.top_k
    try:
        ctx = retrieve(store, embedder, question, top_k)
        messages = initial_messages(cfg.system_prompt)
        messages.append(
            {"role": "user", "content": build_user_message(question, ctx)}
        )
        for piece in chat.stream(messages):
            typer.echo(piece, nl=False)
        typer.echo("")
        _print_sources(ctx.hits)
    finally:
        store.close()

@app.command("chat")
def chat_cmd(
    k: int = typer.Option(None, "--k", help="Override TOP_K"),
) -> None:
    _setup_logging()
    cfg = load_config()
    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    chat = ChatClient.from_config(cfg)
    top_k = k if k is not None else cfg.top_k

    typer.echo(
        f"pyrag chat - model={cfg.chat_model}\n"
        "Type your questions. Commands: /reset to clear history, /exit to quit."
    )

    messages = initial_messages(cfg.system_prompt)

    try:
        while True:
            try:
                question = typer.prompt("\nyou", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                typer.echo("")
                break

            q = question.strip()
            if not q:
                continue
            if q in {"/exit", "/quit"}:
                break
            if q == "/reset":
                messages = initial_messages(cfg.system_prompt)
                typer.echo("Chat history cleared.")
                continue

            ctx = retrieve(store, embedder, q, top_k)
            messages.append(
                {"role": "user", "content": build_user_message(q, ctx)}
            )

            typer.echo("\nassistant> ", nl=False)
            answer_parts: list[str] = []
            spinner = _Spinner()
            spinner.start()
            try:
                for piece in chat.stream(messages):
                    if not answer_parts:
                        spinner.stop()
                        typer.echo("\nassistant> ", nl=False)
                    typer.echo(piece, nl=False)
                    answer_parts.append(piece)
            finally:
                spinner.stop()
            typer.echo("")
            messages.append({"role":"assistant", "content":"".join(answer_parts)})
            _print_sources(ctx.hits)
    finally:
        store.close()

@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code change"),
) -> None:
    _setup_logging()
    import uvicorn
    uvicorn.run("pyrag.web:app", host=host, port=port, reload=reload, ws="none")


if __name__ == "__main__":
    app()
        