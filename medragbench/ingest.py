"""
Stage 0-1 of the MedRAGBench pipeline: ingest uploaded PDFs and build the
two indexes used later to find supporting evidence for gold answers.

  * Extract text + metadata from each PDF (pdfplumber, PyPDF2 fallback).
  * Split into overlapping word-window chunks, keeping provenance.
  * Build a dense vector index in ChromaDB (text-embedding-3-small).
  * Build a sparse keyword index with rank_bm25 over the same chunks.

The resulting `Corpus` object holds the chunk list and both indexes and is
passed to the search module in Stage 3.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import config, llm


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    text: str
    order: int  # position of chunk within its paper


@dataclass
class Corpus:
    chunks: List[Chunk] = field(default_factory=list)
    papers: Dict[str, str] = field(default_factory=dict)  # paper_id -> title
    _chroma_collection: object = None
    _bm25: object = None
    _bm25_tokens: List[List[str]] = field(default_factory=list)

    def chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None


# --------------------------------------------------------------------------
# PDF text extraction
# --------------------------------------------------------------------------
def _extract_pdf_text(path: str) -> str:
    """Extract text from a PDF, preferring pdfplumber, falling back to PyPDF2."""
    text_parts: List[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
    except Exception:
        text_parts = []

    if not text_parts:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(path)
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
        except Exception:
            text_parts = []

    return "\n".join(text_parts)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)          # join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)         # collapse whitespace
    return text.strip()


def _derive_title(path: str, text: str) -> str:
    """Best-effort human-readable title: try a metadata title, else filename."""
    try:
        from PyPDF2 import PdfReader

        meta = PdfReader(path).metadata
        if meta and meta.title and meta.title.strip():
            return meta.title.strip()
    except Exception:
        pass
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace("_", " ").strip()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def _chunk_words(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------
# Tokeniser for BM25
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# --------------------------------------------------------------------------
# Build the corpus
# --------------------------------------------------------------------------
def build_corpus(
    pdf_paths: List[str],
    progress: Optional[Callable[[str], None]] = None,
) -> Corpus:
    """Run Stage 0-1: extract, chunk, and index the uploaded PDFs."""

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    if len(pdf_paths) > config.MAX_PDFS:
        raise ValueError(
            f"{len(pdf_paths)} PDFs provided but MAX_PDFS={config.MAX_PDFS}. "
            "Raise MAX_PDFS in config.py to scale up."
        )

    corpus = Corpus()

    # ---- Extract + chunk --------------------------------------------------
    for path in pdf_paths:
        log(f"Extracting: {os.path.basename(path)}")
        raw = _extract_pdf_text(path)
        cleaned = _clean_text(raw)
        if not cleaned:
            log(f"  (no extractable text, skipping {os.path.basename(path)})")
            continue
        paper_id = uuid.uuid4().hex[:8]
        title = _derive_title(path, cleaned)
        corpus.papers[paper_id] = title

        pieces = _chunk_words(
            cleaned, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS
        )
        for i, piece in enumerate(pieces):
            corpus.chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}_{i}",
                    paper_id=paper_id,
                    paper_title=title,
                    text=piece,
                    order=i,
                )
            )
        log(f"  -> {len(pieces)} chunks")

    if not corpus.chunks:
        raise RuntimeError("No extractable text found in the provided PDFs.")

    # ---- Dense index (ChromaDB) ------------------------------------------
    log("Embedding chunks (text-embedding-3-small)...")
    import chromadb

    client = chromadb.PersistentClient(path=config.PATHS.chroma_dir)
    collection_name = f"corpus_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    # Embed in batches to respect API limits.
    batch = 64
    for start in range(0, len(corpus.chunks), batch):
        sub = corpus.chunks[start : start + batch]
        vectors = llm.embed_texts([c.text for c in sub])
        collection.add(
            ids=[c.chunk_id for c in sub],
            embeddings=vectors,
            documents=[c.text for c in sub],
            metadatas=[
                {"paper_id": c.paper_id, "paper_title": c.paper_title, "order": c.order}
                for c in sub
            ],
        )
        log(f"  embedded {min(start + batch, len(corpus.chunks))}/{len(corpus.chunks)}")

    corpus._chroma_collection = collection

    # ---- Sparse index (BM25) ---------------------------------------------
    log("Building BM25 keyword index...")
    from rank_bm25 import BM25Okapi

    corpus._bm25_tokens = [tokenize(c.text) for c in corpus.chunks]
    corpus._bm25 = BM25Okapi(corpus._bm25_tokens)

    log(
        f"Corpus ready: {len(corpus.papers)} papers, "
        f"{len(corpus.chunks)} chunks indexed."
    )
    return corpus
