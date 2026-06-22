"""
Pipeline orchestrator for MedRAGBench.

Wires the stages together so the GUI can call a single function on a worker
thread:

    Stage 0-1  ingest.build_corpus
    Stage 2    generate.generate_questions
    Stage 3    search.find_evidence            (per question)
    Stage 4    generate.assemble_gold_answer   / mark_unanswerable
    Stage 5    quality.quality_filter

Stage 6 (clinician review) happens interactively in the GUI.
Stage 7 (export) is `export_dataset` below.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional

from . import config, ingest, generate, search, quality
from .generate import BenchmarkItem


def run_generation(
    pdf_paths: List[str],
    progress: Optional[Callable[[str], None]] = None,
) -> List[BenchmarkItem]:
    """Run Stages 0-5 and return items ready for clinician review."""

    def log(m: str) -> None:
        if progress:
            progress(m)

    # Stage 0-1
    log("=== Stage 0-1: Ingest & index corpus ===")
    corpus = ingest.build_corpus(pdf_paths, progress=log)

    # Stage 2
    log("=== Stage 2: Generate questions ===")
    items = generate.generate_questions(corpus, progress=log)

    # Stage 3-4
    log("=== Stage 3-4: Find evidence & assemble gold answers ===")
    for i, item in enumerate(items):
        log(f"  ({i + 1}/{len(items)}) {item.type} / {item.category}")
        if item.type in config.ANSWERABLE_TYPES:
            passages, sufficient = search.find_evidence(corpus, item.question)
            if sufficient:
                generate.assemble_gold_answer(item, passages)
            else:
                # Corpus lacks evidence -> downgrade to unanswerable.
                item.flags.append("auto_unanswerable_low_evidence")
                item.type = "Unanswerable"
                generate.mark_unanswerable(item)
        else:
            # Unanswerable by design: confirm absence of evidence via scores.
            passages, sufficient = search.find_evidence(corpus, item.question)
            if sufficient:
                # Evidence unexpectedly exists; flag for clinician attention.
                item.flags.append("expected_unanswerable_but_evidence_found")
            generate.mark_unanswerable(item)

    # Stage 5
    log("=== Stage 5: Automated quality filter ===")
    kept, _dropped = quality.quality_filter(items, progress=log)

    log(f"=== Generation complete: {len(kept)} items for review ===")
    return kept


def export_dataset(items: List[BenchmarkItem], path: str) -> int:
    """
    Stage 7: write APPROVED items to a JSON file.

    Returns the number of records written.
    """
    approved = [it for it in items if it.approved]
    payload = {
        "benchmark": "MedRAGBench",
        "version": 1,
        "categories": config.PKD_CATEGORIES,
        "question_types": config.QUESTION_TYPES,
        "count": len(approved),
        "records": [it.to_record() for it in approved],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return len(approved)
