from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .embeddings import Embedder
from .stores.base import SearchHit, VectorStore
from .llm import ChatClient, Message
from .stores.base import SearchHit, VectorStore


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about mythological creatures, "
    "folklore, mythology and related topics. You have documents available to you to use to "
    "answer questions. Use the context provided and your general knowledge from outside "
    "the documents to answer each question. Do not invent facts. When you use a fact "
    "from the context, do NOT cite the source filename in parentheses. Just write the "
    "response as a part of the normal conversation. "
    "NEVER use phrasing like 'According to the doucments...' or anything similar. "
    "You ARE allowed to use your general knowledge to answer questions, provided "
    "that the question is related to your area of expertise. Politely decline to "
    "answer questions not related to mythological creatures, folklore, mythology, "
    "and related topics. "
    "Keep your responses friendly and conversational."
)

IMAGE_INSTRUCTIONS = (
    "Format your answers in plain Markdown. "
    "IMAGES: an image is ONLY available to you when a context chunk above "
    "contains an explicit 'Image URL: <url>' line. In that case, and only "
    "in that case, you MAY embed the image inline using markdown image "
    "syntax: ![short alt text](URL), copying the URL verbatim from the "
    "'Image URL:' line. If no 'Image URL:' line is present in the context, "
    "you MUST NOT emit any markdown image syntax (no ![...](...) and no "
    "![...] at all). Just answer in words. The user can ask for a URL "
    "that doesn't exist; in that case, say so plainly. Never invent a URL "
    "and never guess one from a filename."
)


@dataclass
class RetrievedContext:
    hits: list[SearchHit]

    def to_prompt_block(self) -> str:
        if not self.hits:
            return "(no relevant context found)"
        parts = []
        for h in self.hits:
            name = Path(h.source_path).name
            is_image = h.metadata.get("type") == "image"
            if is_image:
                url = f"/files/{quote(name)}"
                parts.append(
                    f"[source: {name} | chunk {h.chunk_index} | "
                    f"score {h.score:.2f} | image]\n"
                    f"Image URL: {url}\n"
                    f"{h.text}"
                )
            else:
                parts.append(
                    f"[source: {name}] | chunk {h.chunk_index} | score {h.score:.2f}\n"
                    f"{h.text}"
                )
        return "\n\n---\n\n".join(parts)


def retrieve(
        store: VectorStore, embedder: Embedder, question: str, k: int
) -> RetrievedContext:
    [vec] = embedder.embed(question)
    return RetrievedContext(hits=store.search(question, vec, k))


def build_user_message(question: str, ctx: RetrievedContext) -> str:
    has_image = any(h.metadata.get("type") == "image" for h in ctx.hits)
    suffix = ""
    if has_image:
        suffix = (
            "\n\n"
            "Reminder: one or more of the context chunks above is an image. "
            "Each image chunk has an 'Image URL: ...' line. If an image "
            "matches what the user is asking about, embed it inline using "
            "markdown image syntax: ![short alt text](URL), copying the URL "
            "verbatim from that line. The 'no filenames in parentheses' rule "
            "does not apply to markdown image URLs — embedding the image is "
            "expected and correct."
        )
    return (
        "Context: \n"
        f"{ctx.to_prompt_block()}\n\n"
        "Question: \n"
        f"{question}"
        f"{suffix}"
    )


def initial_messages(system_prompt: str | None) -> list[Message]:
    base = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    return [{"role": "system", "content": base + IMAGE_INSTRUCTIONS}]


REWRITE_SYSTEM = (
    "You rewrite the user's latest message into a standalone search query "
    "that captures everything the search index needs to know. USe the prior "
    "conversation to resolve pronouns and implicit references. "
    "Return ONLY the rewritten query -- no preamble, no quotes, no explanation. "
    "If the latest message is already a complete standalone question, return it unchanged."
)


def rewrite_query(
        chat: ChatClient, history: list[Message], question: str
) -> str:
    if not history:
        return question

    messages: list[Message] = [
        {"role": "system", "content": REWRITE_SYSTEM},
        *history,
        {"role": "user", "content": question},
    ]

    rewritten = "".join(chat.stream(messages)).strip()

    if (
        len(rewritten) >= 2
        and rewritten[0] == rewritten[-1]
        and rewritten[0] in {'"', "'"}
    ):
        rewritten = rewritten[1:-1].strip()

    return rewritten or question
