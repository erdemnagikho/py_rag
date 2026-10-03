from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


_PARAGRAPH_RE = re.compile(r"\n\s*\n")

def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if overlap > chunk_size:
        raise ValueError("overlap  must be smaller than chunk size")
    
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in _split_paragraphs(text):
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                chunks.append(para[start:end].strip())
                if end == len(para):
                    break
                start = end - overlap
            continue

        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            flush()
            buf = para

    flush()
    return [Chunk(index=i, text=t) for i, t in enumerate(chunks) if t]