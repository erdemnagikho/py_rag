from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypedDict

from openai import OpenAI

if TYPE_CHECKING:
    from .config import Config


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