from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Chunk:
    index: int
    text: str

class RecursiveCharacterTextSplitter:
    """
    Splits text recursively using a list of separators. 
    It tries to split on the largest semantic boundary (like paragraphs),
    and falls back to sentences, words, and characters only if needed.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # List of separators from highest semantic importance to lowest
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        separator = separators[-1]
        new_separators = []

        # Find the first valid separator
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        splits = [s for s in splits if s != ""]

        docs = []
        current_doc = []
        total = 0

        for s in splits:
            len_to_add = len(s) + (len(separator) if current_doc else 0)
            
            # Flush if adding the next split exceeds chunk_size
            if total + len_to_add > self.chunk_size and current_doc:
                doc_text = separator.join(current_doc)
                if separator == ". " and not doc_text.endswith("."):
                    doc_text += "."
                docs.append(doc_text)
                
                # Build overlap for the next chunk
                overlap_doc = []
                overlap_length = 0
                for c in reversed(current_doc):
                    len_c = len(c) + (len(separator) if overlap_doc else 0)
                    if overlap_length + len_c > self.chunk_overlap:
                        break
                    overlap_doc.insert(0, c)
                    overlap_length += len_c
                
                current_doc = overlap_doc
                total = overlap_length
            
            # If a single split is still larger than chunk_size
            if len(s) > self.chunk_size:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    for sub in sub_splits:
                        sub_len = len(sub) + (len(separator) if current_doc else 0)
                        if total + sub_len > self.chunk_size and current_doc:
                            doc_text = separator.join(current_doc)
                            if separator == ". " and not doc_text.endswith("."):
                                doc_text += "."
                            docs.append(doc_text)
                            current_doc = []
                            total = 0
                        
                        if len(sub) > self.chunk_size:
                            # Absolute fallback: character splitting
                            start = 0
                            while start < len(sub):
                                end = min(start + self.chunk_size, len(sub))
                                docs.append(sub[start:end])
                                start = end - self.chunk_overlap if end < len(sub) else end
                        else:
                            current_doc.append(sub)
                            total += len(sub) + (len(separator) if len(current_doc) > 1 else 0)
                else:
                    # No more separators, character slice fallback
                    start = 0
                    while start < len(s):
                        end = min(start + self.chunk_size, len(s))
                        docs.append(s[start:end])
                        start = end - self.chunk_overlap if end < len(s) else end
            else:
                current_doc.append(s)
                total += len(s) + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_text = separator.join(current_doc)
            if separator == ". " and not doc_text.endswith("."):
                doc_text += "."
            docs.append(doc_text)

        return docs

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if overlap > chunk_size:
        raise ValueError("overlap must be smaller than chunk size")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    string_chunks = splitter.split_text(text)
    
    return [Chunk(index=i, text=t.strip()) for i, t in enumerate(string_chunks) if t.strip()]
