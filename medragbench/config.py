"""
Central configuration for MedRAGBench.

Every tunable knob lives here so that scaling, model choices, and thresholds
can be changed in one place. Notably:

  * MAX_PDFS         -- raise from 10 to 100 (or higher) when you are ready.
  * LLM_PROVIDER     -- "openai" today; see README "Switching to Claude".
  * Generation/retrieval thresholds -- documented inline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------
# Support up to 10 PDFs for now. To scale to 100, change ONLY this number.
MAX_PDFS: int = 10

# How many benchmark questions to draft in total (spread across categories
# and types). With ~100 papers you might raise this to ~100.
TARGET_QUESTION_COUNT: int = 25


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------
# "openai" or "anthropic". The README explains exactly what to change to
# switch to Claude (you also flip this flag).
LLM_PROVIDER: str = os.environ.get("MEDRAGBENCH_PROVIDER", "openai")

# OpenAI model names.
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
OPENAI_CHAT_MODEL: str = "gpt-5"

# Anthropic model name (used only when LLM_PROVIDER == "anthropic").
ANTHROPIC_CHAT_MODEL: str = "claude-opus-4-1"

# API keys are read from the environment. Set OPENAI_API_KEY (and, if you
# switch providers, ANTHROPIC_API_KEY) before launching the app.
OPENAI_API_KEY_ENV: str = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV: str = "ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE_WORDS: int = 220          # approx words per chunk
CHUNK_OVERLAP_WORDS: int = 40        # overlap between neighbouring chunks


# --------------------------------------------------------------------------
# Retrieval / evidence search
# --------------------------------------------------------------------------
DENSE_TOP_K: int = 12                # candidates from the dense index
BM25_TOP_K: int = 12                 # candidates from the BM25 index
RRF_K: int = 60                      # RRF constant (standard default = 60)
RRF_TOP_K: int = 12                  # candidates kept after fusion, before rerank
RERANK_TOP_K: int = 6                # passages kept after cross-encoder rerank

# Cross-encoder model for reranking (CPU-friendly).
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Sufficiency / unanswerability threshold.
# The cross-encoder produces a relevance logit per (question, passage) pair.
# If the best reranked passage scores BELOW this, we treat the corpus as
# lacking evidence -> the item becomes "unanswerable / abstain".
# ms-marco-MiniLM logits are roughly in [-11, +11]; ~0 is a reasonable
# "weakly relevant" boundary. Tune against your own corpus.
SUFFICIENCY_THRESHOLD: float = 0.0


# --------------------------------------------------------------------------
# Quality filter
# --------------------------------------------------------------------------
# Two answers (gold vs. a no-retrieval LLM answer) whose embedding cosine
# similarity exceeds this are considered "answerable from memory" -> flagged
# as a recall-risk item (it tests parametric knowledge, not retrieval).
RECALL_RISK_SIMILARITY: float = 0.82

# Two questions whose embedding cosine similarity exceeds this are treated
# as near-duplicates; the later one is dropped.
DEDUP_SIMILARITY: float = 0.92


# --------------------------------------------------------------------------
# PKD categories and question types
# --------------------------------------------------------------------------
PKD_CATEGORIES: List[str] = [
    "Disease mechanisms",
    "Diagnosis and imaging",
    "Treatment and medication",
    "Diet and lifestyle",
    "Progression and prognosis",
]

QUESTION_TYPES: List[str] = [
    "Standard factual",
    "Context-dependent",
    "False-premise",
    "Safety-critical",
    "Unanswerable",
]

# Expected system behaviour per question type. This is written into every
# dataset record so a downstream evaluator knows what a correct RAG system
# should do.
EXPECTED_BEHAVIOR_BY_TYPE = {
    "Standard factual": "answer",
    "Context-dependent": "answer_with_synthesis",
    "False-premise": "correct_the_premise",
    "Safety-critical": "answer_with_safety_caveat_or_defer",
    "Unanswerable": "abstain",
}

# Which types require supporting evidence to be retrieved from the corpus.
ANSWERABLE_TYPES = {
    "Standard factual",
    "Context-dependent",
    "False-premise",
    "Safety-critical",
}


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
@dataclass
class Paths:
    workdir: str = os.path.join(os.path.expanduser("~"), ".medragbench")
    chroma_dir: str = field(init=False)
    export_default: str = field(init=False)

    def __post_init__(self) -> None:
        self.chroma_dir = os.path.join(self.workdir, "chroma")
        self.export_default = os.path.join(self.workdir, "benchmark.json")
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)


PATHS = Paths()
