from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from .config import Config

class Embedder:
    def __init__(
            self,
            base_url: str | None,
            api_key: str,
            model: str,
            expected_dim: int,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._expected_dim = expected_dim
    
    @classmethod
    def from_config(cls, cfg: Config) -> Embedder:
        return cls(cfg.openai_base_url, cfg.openai_api_key, cfg.embed_model, cfg.embed_dim)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        resp = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [d.embedding for d in resp.data]

        if vectors and len(vectors[0]) != self._expected_dim:
            raise ValueError(
                f"Embedding dim mismatch: model returned {len(vectors[0])}, "
                f"config expects {self._expected_dim}. Update EMBED_DIM and "
                f"re-init the vector store schema."
            )
            
        return vectors
            