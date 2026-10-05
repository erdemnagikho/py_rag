from __future__ import annotations

import base64
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from openai import OpenAI

if TYPE_CHECKING:
    from .config import Config


_PORTABLE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_MIME_BY_SUFFIX = {".png": "image/png", ".jpeg": "image/jpeg", ".jpg": "image/jpeg"}

@contextmanager
def _as_portable_image(path: Path) -> Iterator[Path]:
    if path.suffix.lower() in _PORTABLE_IMAGE_SUFFIXES:
        yield path
        return
    
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with Image.open(path) as img:
            img.convert("RGB").save(tmp_path, format="PNG")
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def _image_data_url(path: Path) -> str:
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"
    
class Message(TypedDict):
    role: str
    content: str

class ChatClient:
    def __init__(self, base_url: str | None, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    @classmethod
    def from_config(cls, cfg: Config) -> ChatClient:
        return cls(cfg.openai_base_url, cfg.openai_api_key, cfg.chat_model)
    
    def stream(self, messages: list[Message]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece
    
    def describe(self, model: str, prompt: str, image_path: Path) -> str:
        with _as_portable_image(image_path) as readable_path:
            data_url = _image_data_url(readable_path)
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    }
                ],
                stream=False,
            )
        content = response.choices[0].message.content or ""
        return content.strip()
                