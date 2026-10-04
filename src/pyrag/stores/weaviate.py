from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from .base import SearchHit, StoredChunk, VectorStore


class WeaviateStore(VectorStore):

    P_SOURCE_PATH = "source_path"
    P_CONTENT_HASH = "content_hash"
    P_CHUNK_INDEX = "chunk_index"
    P_CONTENT = "content"
    P_CHUNK_METADATA = "chunk_metadata"
    P_DOCUMENT_METADATA = "document_metadata"
    P_INGESTED_AT = "ingested_at"
    P_CHUNK_COUNT = "chunk_count"

    def __init__(
            self,
            host: str,
            http_port: int,
            grpc_port: int,
            collection: str,
    ) -> None:
        self._host = host
        self._http_port = http_port
        self._grpc_port = grpc_port
        self._collection_name = collection
        self._client: weaviate.WeaviateClient | None = None


    def _connect(self) -> weaviate.WeaviateClient:
        if self._client is None or not self._client.is_connected():
            self._client = weaviate.connect_to_local(
                host=self._host,
                port=self._http_port,
                grpc_port=self._grpc_port,
            )
            self._ensure_collection()
        return self._client
    
    def _ensure_collection(self) -> None:
        assert self._client is not None
        if self._client.collections.exists(self._collection_name):
            return
        
        self._client.collections.create(
            name=self._collection_name,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name=self.P_SOURCE_PATH, data_type=DataType.TEXT),
                Property(name=self.P_CONTENT_HASH, data_type=DataType.TEXT),
                Property(name=self.P_CHUNK_INDEX, data_type=DataType.INT),
                Property(name=self.P_CONTENT, data_type=DataType.TEXT),
                Property(name=self.P_CHUNK_METADATA, data_type=DataType.TEXT),
                Property(name=self.P_DOCUMENT_METADATA, data_type=DataType.TEXT),
                Property(name=self.P_INGESTED_AT, data_type=DataType.DATE),
                Property(name=self.P_CHUNK_COUNT, data_type=DataType.INT),
            ],
        )

    def has_document(self, source_path: str, content_hash: str) -> bool:
        col = self._connect().collections.get(self._collection_name)
        result = col.aggregate.over_all(
            filters=(
                Filter.by_property(self.P_SOURCE_PATH).equal(source_path)
                & Filter.by_property(self.P_CONTENT_HASH).equal(content_hash)
            ),
            total_count=True,
        )
        return (result.total_count or 0) > 0
    
    def upsert_document(
            self,
            source_path: str,
            content_hash: str,
            chunks: list[StoredChunk],
            metadata: dict[str, Any] | None = None,
    ) -> None:
        col = self._connect().collections.get(self._collection_name)

        col.data.delete_many(
            where=Filter.by_property(self.P_SOURCE_PATH).equal(source_path)
        )

        if not chunks:
            return
        
        ingested_at = datetime.now(timezone.utc)
        doc_metadata_json = json.dumps(metadata or {})
        chunk_count = len(chunks)

        with col.batch.dynamic() as batch:
            for c in chunks:
                batch.add_object(
                    properties = {
                        self.P_SOURCE_PATH: source_path,
                        self.P_CONTENT_HASH: content_hash,
                        self.P_CHUNK_INDEX: c.index,
                        self.P_CONTENT: c.text,
                        self.P_CHUNK_METADATA: json.dumps(c.metadata or {}),
                        self.P_DOCUMENT_METADATA: doc_metadata_json,
                        self.P_INGESTED_AT: ingested_at,
                        self.P_CHUNK_COUNT: chunk_count,
                    },
                    vector=c.embedding,
                )

        failed = col.batch.failed_objects
        if failed:
            raise RuntimeError(
                f"Weaviate batch insert failed for {len(failed)} object(s); "
                f"first error: {failed[0].message}"
            )
        
    def delete_document(self, source_path: str) -> None:
        col = self._connect().collections.get(self._collection_name)
        col.data.delete_many(
            where=Filter.by_property(self.P_SOURCE_PATH).equal(source_path)
        )

    def search(
            self, query_text: str, query_embedding: list[float], k: int
    ) -> list[SearchHit]:
        col = self._connect().collections.get(self._collection_name)
        result = col.query.hybrid(
            query=query_text,
            vector=query_embedding,
            limit=k,
            fusion_type=HybridFusion.RANKED,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )

        hits: list[SearchHit] = []
        for obj in result.objects:
            props = obj.properties

            chunk_md = _safe_json(props.get(self.P_CHUNK_METADATA))
            doc_md = _safe_json(props.get(self.P_DOCUMENT_METADATA))

            chunk_md = dict(chunk_md)
            chunk_md["hybrid_score"] = (
                float(obj.metadata.score) if obj.metadata.score is not None else None
            )

            hits.append(
                SearchHit(
                    source_path=str(props[self.P_SOURCE_PATH]),
                    chunk_index=int(props[self.P_CHUNK_INDEX]),
                    text=str(props[self.P_CONTENT]),
                    score=float(obj.metadata.score or 0.0),
                    metadata=chunk_md,
                    document_metadata=doc_md,
                    ingested_at=_as_datetime(props.get(self.P_INGESTED_AT)),
                )
            )
        return hits

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client.close()
                

def _as_datetime(v: object) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    
    return None


def _safe_json(blob: object) -> dict[str, Any]:
    if not blob:
        return {}
    if isinstance(blob, dict):
        return blob
    
    try:
        parsed = json.loads(str(blob))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}