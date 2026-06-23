"""
Stage 3 of the MedRAGBench pipeline: find the supporting evidence for a
question by hybrid search over the indexed corpus.

  1. Dense search (ChromaDB cosine) -> ranked chunk list.
  2. BM25 keyword search             -> ranked chunk list.
  3. Reciprocal Rank Fusion (RRF) merges the two ranked lists.
  4. Cross-encoder reranks the fused candidates.
  5. Sufficiency check: if the best reranked score is below the threshold,
     the corpus lacks evidence -> the item should be treated as unanswerable.

These passages are used to ASSEMBLE GOLD ANSWERS for the benchmark; this is
not a chatbot retrieval path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import config, llm
from .ingest import Corpus, Chunk, tokenize


@dataclass
class RetrievedPassage:
    chunk: Chunk
    rerank_score: float


# Lazy singleton for the cross-encoder (loaded once, reused).
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
    return _cross_encoder


# --------------------------------------------------------------------------
# Individual searches
# --------------------------------------------------------------------------
def _dense_search(corpus: Corpus, question: str, top_k: int) -> List[str]:
    """Return chunk_ids ranked by dense cosine similarity."""
    qvec = llm.embed_text(question)
    res = corpus._chroma_collection.query(
        query_embeddings=[qvec],
        n_results=min(top_k, len(corpus.chunks)),
    )
    ids = res.get("ids", [[]])
    return ids[0] if ids else []


def _bm25_search(corpus: Corpus, question: str, top_k: int) -> List[str]:
    """Return chunk_ids ranked by BM25 keyword score."""
    scores = corpus._bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = ranked[: min(top_k, len(ranked))]
    return [corpus.chunks[i].chunk_id for i in top]


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------
def _rrf(ranked_lists: List[List[str]], k: int, top_k: int) -> List[str]:
    """
    Merge several ranked lists of chunk_ids with Reciprocal Rank Fusion.

    RRF score(d) = sum over lists of 1 / (k + rank(d)), rank 0-based.
    Scale-independent: a chunk ranked highly in BOTH lists beats one ranked
    highly in only one. Returns the top_k chunk_ids by fused score.
    """
    fused: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [cid for cid, _ in ordered[:top_k]]


# --------------------------------------------------------------------------
# Cross-encoder rerank
# --------------------------------------------------------------------------
def _rerank(
    corpus: Corpus, question: str, candidate_ids: List[str], top_k: int
) -> List[RetrievedPassage]:
    if not candidate_ids:
        return []
    chunks = [corpus.chunk_by_id(cid) for cid in candidate_ids]
    chunks = [c for c in chunks if c is not None]
    pairs = [(question, c.text) for c in chunks]

    encoder = _get_cross_encoder()
    scores = encoder.predict(pairs)

    scored = sorted(
        zip(chunks, scores), key=lambda cs: float(cs[1]), reverse=True
    )
    return [
        RetrievedPassage(chunk=c, rerank_score=float(s))
        for c, s in scored[:top_k]
    ]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def find_evidence(
    corpus: Corpus, question: str
) -> Tuple[List[RetrievedPassage], bool]:
    """
    Run the full hybrid search for one question.

    Returns (passages, sufficient) where `sufficient` is True iff the best
    reranked passage scores at or above SUFFICIENCY_THRESHOLD. When False,
    the corpus is judged to lack evidence and the caller should mark the
    item unanswerable / abstain.
    """
    dense_ids = _dense_search(corpus, question, config.DENSE_TOP_K)
    bm25_ids = _bm25_search(corpus, question, config.BM25_TOP_K)
    fused_ids = _rrf([dense_ids, bm25_ids], config.RRF_K, config.RRF_TOP_K)
    passages = _rerank(corpus, question, fused_ids, config.RERANK_TOP_K)

    sufficient = bool(passages) and (
        passages[0].rerank_score >= config.SUFFICIENCY_THRESHOLD
    )
    return passages, sufficient
