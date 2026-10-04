from __future__ import annotations

import json
from typing import Any
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .base import SearchHit, StoredChunk, VectorStore

class PostgresStore(VectorStore):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: psycopg.Connection[Any] | None = None

    def _connect(self) -> psycopg.Connection[Any]:
        if self._conn is None or self._conn.closed:
            conn = psycopg.connect(self._dsn, autocommit=False)
            register_vector(conn)
            self._conn = conn
        
        return self._conn

    def has_document(self, source_path: str, content_hash: str) -> bool:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE source_path = %s AND content_hash = %s",
                (source_path, content_hash),
            )
            found = cur.fetchone() is not None
        conn.commit()
        return found

    def upsert_document(
        self,
        source_path: str,
        content_hash: str,
        chunks: list[StoredChunk],
    ) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (source_path, content_hash, chunk_count)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_path) DO UPDATE
                        SET content_hash = EXCLUDED.content_hash,
                            chunk_count = EXCLUDED.chunk_count,
                            ingested_at = NOW()
                    RETURNING id
                    """,
                    (source_path, content_hash, len(chunks))
                )
                doc_id = cur.fetchone()[0]

                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                    
                if chunks:
                    cur.executemany(
                        """
                        INSERT INTO chunks
                            (document_id, chunk_index, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                doc_id,
                                c.index,
                                c.text,
                                c.embedding,
                                json.dumps(c.metadata),
                            )
                            for c in chunks
                        ],    
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def delete_document(self, source_path: str) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE source_path = %s", (source_path,))
            conn.commit()

    def search(
            self, query_text: str, query_embedding: list[float], k: int        
    ) -> list[SearchHit]:
        conn = self._connect()
        candidates = max(20, k*4)
        rrf_k = 60
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SET LOCAL hnsw.ef_search = 100")
            cur.execute(
                """
                WITH semantic AS (
                    SELECT id, dist,
                        ROW_NUMBER() OVER (ORDER BY dist) AS rank
                    FROM (
                        SELECT c.id, c.embedding <=> %(vec)s::vector AS dist
                        FROM chunks c
                        ORDER BY c.embedding <=> %(vec)s::vector
                        LIMIT %(cand)s
                    ) s
                ),

                lexical AS (
                    SELECT id,
                        ROW_NUMBER() OVER (ORDER BY score DESC) as rank
                    FROM (
                        SELECT c.id, ts_rank(c.content_tsv, q.query) AS score
                        FROM chunks c,
                            websearch_to_tsquery('english', %(qtext)s) AS q(query)
                        WHERE c.content_tsv @@ q.query
                        ORDER BY ts_rank(c.content_tsv, q.query) DESC
                        LIMIT %(cand)s
                    ) l
                ),

                fused AS (
                    SELECT id, SUM(1.0 / (%(rrf)s + rank)) AS rrf_score
                    FROM (
                        SELECT id, rank FROM semantic
                        UNION ALL
                        SELECT id, rank FROM lexical
                    ) ranks
                    GROUP BY id
                )

                SELECT d.source_path,
                       c.chunk_index,
                       c.content,
                       c.metadata,
                       f.rrf_score,
                       CASE WHEN s.dist IS NOT NULL
                           THEN 1 - s.dist
                           ELSE 1 - (c.embedding <=> %(vec)s::vector)
                       END AS cosine,
                       s.rank AS sem_rank,
                       l.rank AS lex_rank
                FROM fused f
                JOIN chunks c ON c.id = f.id
                JOIN documents d ON d.id = c.document_id
                LEFT JOIN semantic s ON s.id = c.id
                LEFT JOIN lexical l ON l.id = c.id
                ORDER BY f.rrf_score DESC
                LIMIT %(k)s
                """,
                {
                    "vec": query_embedding,
                    "qtext": query_text,
                    "cand": candidates,
                    "rrf": rrf_k,
                    "k": k,
                }
            )
            hits = []
            for r in cur.fetchall():
                meta = dict(r["metadata"] or {})
                meta["rrf_score"] = float(r["rrf_score"])
                meta["semantic_rank"] = r["sem_rank"]
                meta["lexical_rank"] = r["lex_rank"]
                hits.append(
                    SearchHit(
                        source_path=r["source_path"],
                        chunk_index=int(r["chunk_index"]),
                        text=r["content"],
                        score=float(r["cosine"]),
                        metadata=meta,
                    )
                )
                
            conn.commit()
            return hits

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None       
        