"""
Stage 5 of the MedRAGBench pipeline: automated quality filter.

Runs before clinician review to protect reviewer time. Four checks:

  1. Recall-risk: ask an LLM the question WITHOUT any retrieved context. If
     its from-memory answer is highly similar to the gold answer, the item
     likely tests parametric recall rather than retrieval -> flag it.
  2. Schema completeness: required fields populated for the item's type.
  3. Semantic dedup: drop questions that are near-duplicates of an earlier,
     kept question (by embedding cosine similarity).
  4. Unanswerable confirmation: an item typed/decided unanswerable must have
     no supporting passages and the abstain behavior.

Items can be flagged (kept, with a warning the clinician sees) or dropped
(near-duplicates). Nothing is silently discarded except exact dedup hits.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from . import config, llm
from .generate import BenchmarkItem


# --------------------------------------------------------------------------
# Cosine similarity helper
# --------------------------------------------------------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# Check 1: recall-risk via no-retrieval answer
# --------------------------------------------------------------------------
_NORETRIEVAL_SYSTEM = (
    "You are a medical question answerer. Answer the question concisely from "
    "your own knowledge. If you do not know, say you do not know."
)


def _recall_risk(item: BenchmarkItem) -> bool:
    """True if a no-retrieval LLM answer closely matches the gold answer."""
    if item.expected_behavior == "abstain" or not item.gold_answer:
        return False
    memory_answer = llm.chat(_NORETRIEVAL_SYSTEM, item.question)
    if not memory_answer:
        return False
    embs = llm.embed_texts([item.gold_answer, memory_answer])
    if len(embs) < 2:
        return False
    sim = _cosine(embs[0], embs[1])
    return sim >= config.RECALL_RISK_SIMILARITY


# --------------------------------------------------------------------------
# Check 2: schema completeness
# --------------------------------------------------------------------------
def _schema_complete(item: BenchmarkItem) -> bool:
    if not item.question.strip():
        return False
    if item.type not in config.QUESTION_TYPES:
        return False
    if item.category not in config.PKD_CATEGORIES:
        return False
    if item.difficulty not in ("easy", "moderate", "hard"):
        return False
    if not item.expected_behavior:
        return False
    if item.expected_behavior == "abstain":
        # Unanswerable items must have no evidence and a non-empty note.
        return item.gold_answer.strip() != "" and not item.supporting_passages
    # Answerable items must have an answer and at least one supporting passage.
    return bool(item.gold_answer.strip()) and bool(item.supporting_passages)


# --------------------------------------------------------------------------
# Check 4: unanswerable confirmation
# --------------------------------------------------------------------------
def _unanswerable_consistent(item: BenchmarkItem) -> bool:
    if item.expected_behavior != "abstain":
        return True  # not an unanswerable item; nothing to confirm
    return not item.supporting_passages and not item.retrieval_targets


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def quality_filter(
    items: List[BenchmarkItem], progress=None
) -> Tuple[List[BenchmarkItem], List[BenchmarkItem]]:
    """
    Apply the four checks. Returns (kept, dropped).

    Kept items may carry flags (visible to clinicians). Dropped items are
    semantic near-duplicates only.
    """

    def log(m):
        if progress:
            progress(m)

    kept: List[BenchmarkItem] = []
    dropped: List[BenchmarkItem] = []

    # Pre-embed all questions once for dedup.
    log("Quality filter: embedding questions for dedup...")
    q_embs = llm.embed_texts([it.question for it in items])

    kept_embs: List[List[float]] = []
    for i, item in enumerate(items):
        log(f"Quality filter: checking {i + 1}/{len(items)}")

        # Check 3: semantic dedup against already-kept questions.
        is_dup = False
        for ke in kept_embs:
            if _cosine(q_embs[i], ke) >= config.DEDUP_SIMILARITY:
                is_dup = True
                break
        if is_dup:
            item.flags.append("near_duplicate")
            dropped.append(item)
            continue

        # Check 2: schema completeness.
        if not _schema_complete(item):
            item.flags.append("schema_incomplete")

        # Check 4: unanswerable consistency.
        if not _unanswerable_consistent(item):
            item.flags.append("unanswerable_inconsistent")

        # Check 1: recall-risk (only meaningful for answerable items).
        try:
            if _recall_risk(item):
                item.flags.append("recall_risk")
        except Exception:
            # A flaky API call should not crash the whole filter.
            item.flags.append("recall_check_failed")

        kept.append(item)
        kept_embs.append(q_embs[i])

    log(f"Quality filter done: {len(kept)} kept, {len(dropped)} dropped (dedup).")
    return kept, dropped
