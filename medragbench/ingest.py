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

import hashlib
import json
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
def _table_to_markdown(table: List[List]) -> str:
    """Convert a pdfplumber table (list of rows) into a markdown table string."""
    if not table or len(table) < 2:
        return ""
    clean = []
    for row in table:
        clean.append([str(cell).strip() if cell else "" for cell in row])
    header = "| " + " | ".join(clean[0]) + " |"
    sep = "| " + " | ".join("---" for _ in clean[0]) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in clean[1:])
    return f"{header}\n{sep}\n{body}"


def _extract_figures(path: str, progress=None) -> List[str]:
    """Extract images from PDF pages and describe them with a vision model."""
    import base64
    descriptions = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                images = page.images
                if not images:
                    continue
                page_img = page.to_image(resolution=200)
                import io
                buf = io.BytesIO()
                page_img.original.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                prompt = (
                    "You are a medical research assistant. Describe this figure "
                    "from a medical paper in detail. Include all data points, "
                    "labels, axes, legends, and any conclusions that can be drawn. "
                    "If it contains a chart or graph, describe the trends and "
                    "key values. Be thorough and specific."
                )
                if progress:
                    progress(f"  Describing figure on page {page_num}...")
                desc = llm.describe_image(img_b64, prompt)
                if desc:
                    descriptions.append(
                        f"[Figure from page {page_num}] {desc}"
                    )
    except Exception:
        pass
    return descriptions


def _extract_pdf_text(path: str, progress=None) -> str:
    """Extract text, tables, and figures from a PDF."""
    text_parts: List[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)

                tables = page.extract_tables() or []
                for table in tables:
                    md = _table_to_markdown(table)
                    if md:
                        text_parts.append(md)
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

    figure_descs = _extract_figures(path, progress=progress)
    text_parts.extend(figure_descs)

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
# Persistence helpers
# --------------------------------------------------------------------------
_CORPUS_META_FILE = "corpus_meta.json"
_CHROMA_COLLECTION_NAME = "medragbench_corpus"


def _corpus_fingerprint(pdf_paths: List[str]) -> str:
    """Stable hash of the set of PDF filenames + sizes to detect changes."""
    parts = []
    for p in sorted(pdf_paths):
        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0
        parts.append(f"{os.path.basename(p)}:{size}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _save_corpus_meta(corpus: Corpus, fingerprint: str) -> None:
    """Persist chunk metadata + papers to disk so we can reload without re-ingesting."""
    meta = {
        "fingerprint": fingerprint,
        "papers": corpus.papers,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "paper_title": c.paper_title,
                "text": c.text,
                "order": c.order,
            }
            for c in corpus.chunks
        ],
    }
    path = os.path.join(config.PATHS.workdir, _CORPUS_META_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def _load_corpus_if_cached(
    fingerprint: str, progress=None
) -> Optional[Corpus]:
    """Try to load a previously saved corpus. Returns None if cache miss."""
    meta_path = os.path.join(config.PATHS.workdir, _CORPUS_META_FILE)
    if not os.path.exists(meta_path):
        return None

    def log(m):
        if progress:
            progress(m)

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    if meta.get("fingerprint") != fingerprint:
        log("Corpus cache fingerprint mismatch, re-ingesting...")
        return None

    import chromadb
    client = chromadb.PersistentClient(path=config.PATHS.chroma_dir)
    try:
        collection = client.get_collection(_CHROMA_COLLECTION_NAME)
    except Exception:
        return None

    if collection.count() == 0:
        return None

    corpus = Corpus()
    corpus.papers = meta["papers"]
    for cd in meta["chunks"]:
        corpus.chunks.append(
            Chunk(
                chunk_id=cd["chunk_id"],
                paper_id=cd["paper_id"],
                paper_title=cd["paper_title"],
                text=cd["text"],
                order=cd["order"],
            )
        )
    corpus._chroma_collection = collection

    from rank_bm25 import BM25Okapi
    corpus._bm25_tokens = [tokenize(c.text) for c in corpus.chunks]
    corpus._bm25 = BM25Okapi(corpus._bm25_tokens)

    log(
        f"Loaded cached corpus: {len(corpus.papers)} papers, "
        f"{len(corpus.chunks)} chunks."
    )
    return corpus


def load_corpus_from_dir(directory: str) -> Optional[Corpus]:
    """Load a corpus from a given directory containing corpus_meta.json and chroma/."""
    meta_path = os.path.join(directory, _CORPUS_META_FILE)
    chroma_path = os.path.join(directory, "chroma")

    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    import chromadb
    if not os.path.isdir(chroma_path):
        chroma_path = config.PATHS.chroma_dir
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection(_CHROMA_COLLECTION_NAME)
    except Exception:
        return None

    if collection.count() == 0:
        return None

    corpus = Corpus()
    corpus.papers = meta.get("papers", {})
    for cd in meta.get("chunks", []):
        corpus.chunks.append(
            Chunk(
                chunk_id=cd["chunk_id"],
                paper_id=cd["paper_id"],
                paper_title=cd["paper_title"],
                text=cd["text"],
                order=cd["order"],
            )
        )
    corpus._chroma_collection = collection

    from rank_bm25 import BM25Okapi
    corpus._bm25_tokens = [tokenize(c.text) for c in corpus.chunks]
    corpus._bm25 = BM25Okapi(corpus._bm25_tokens)

    return corpus


# --------------------------------------------------------------------------
# Build the corpus
# --------------------------------------------------------------------------
def build_corpus(
    pdf_paths: List[str],
    progress: Optional[Callable[[str], None]] = None,
) -> Corpus:
    """Extract, chunk, and index PDFs. Reuses cached corpus if PDFs haven't changed."""

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    if len(pdf_paths) > config.MAX_PDFS:
        raise ValueError(
            f"{len(pdf_paths)} PDFs provided but MAX_PDFS={config.MAX_PDFS}. "
            "Raise MAX_PDFS in config.py to scale up."
        )

    fingerprint = _corpus_fingerprint(pdf_paths)
    cached = _load_corpus_if_cached(fingerprint, progress=log)
    if cached is not None:
        return cached

    corpus = Corpus()

    # ---- Extract + chunk --------------------------------------------------
    for path in pdf_paths:
        log(f"Extracting: {os.path.basename(path)}")
        raw = _extract_pdf_text(path, progress=log)
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
    try:
        client.delete_collection(_CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=_CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

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

    # ---- Save to disk for next time --------------------------------------
    _save_corpus_meta(corpus, fingerprint)

    log(
        f"Corpus ready: {len(corpus.papers)} papers, "
        f"{len(corpus.chunks)} chunks indexed."
    )
    return corpus
