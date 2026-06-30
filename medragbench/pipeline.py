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
import os
from typing import Callable, List, Optional

from . import config, ingest, generate, search, quality
from .generate import BenchmarkItem

_RESULTS_FILE = os.path.join(config.PATHS.workdir, "pipeline_results.json")


def save_results(items: List[BenchmarkItem]) -> None:
    """Persist pipeline results to disk."""
    records = []
    for it in items:
        records.append({
            "question": it.question,
            "type": it.type,
            "category": it.category,
            "difficulty": it.difficulty,
            "expected_behavior": it.expected_behavior,
            "gold_answer": it.gold_answer,
            "supporting_passages": it.supporting_passages,
            "source_papers": it.source_papers,
            "retrieval_targets": it.retrieval_targets,
            "flags": it.flags,
            "approved": it.approved,
        })
    with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def load_results() -> Optional[List[BenchmarkItem]]:
    """Load previously saved pipeline results. Returns None if no saved results."""
    if not os.path.exists(_RESULTS_FILE):
        return None
    try:
        with open(_RESULTS_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return None
    if not isinstance(records, list) or not records:
        return None
    items = []
    for r in records:
        items.append(BenchmarkItem(
            question=r.get("question", ""),
            type=r.get("type", ""),
            category=r.get("category", ""),
            difficulty=r.get("difficulty", "moderate"),
            expected_behavior=r.get("expected_behavior", ""),
            gold_answer=r.get("gold_answer", ""),
            supporting_passages=r.get("supporting_passages", []),
            source_papers=r.get("source_papers", []),
            retrieval_targets=r.get("retrieval_targets", []),
            flags=r.get("flags", []),
            approved=r.get("approved", False),
        ))
    return items


def run_generation(
    pdf_paths: List[str],
    progress: Optional[Callable[[str], None]] = None,
    preloaded_corpus=None,
) -> List[BenchmarkItem]:
    """Run Stages 0-5 and return items ready for clinician review."""

    def log(m: str) -> None:
        if progress:
            progress(m)

    if preloaded_corpus is not None:
        log("=== Using pre-loaded corpus ===")
        corpus = preloaded_corpus
    else:
        log("=== Ingesting & indexing corpus ===")
        corpus = ingest.build_corpus(pdf_paths, progress=log)

    log("=== Classifying papers ===")
    category_breakdown = generate.classify_papers(corpus, progress=log)

    log("=== Generating questions ===")
    items = generate.generate_questions(corpus, category_breakdown, progress=log)

    log("=== Finding evidence & assembling gold answers ===")
    for i, item in enumerate(items):
        log(f"  ({i + 1}/{len(items)}) {item.type} / {item.category}")
        if item.type in config.ANSWERABLE_TYPES:
            passages, sufficient = search.find_evidence(corpus, item.question)
            if sufficient:
                generate.assemble_gold_answer(item, passages)
            else:
                item.flags.append("auto_unanswerable_low_evidence")
                item.type = "Unanswerable"
                generate.mark_unanswerable(item)
        else:
            passages, sufficient = search.find_evidence(corpus, item.question)
            if sufficient:
                item.flags.append("expected_unanswerable_but_evidence_found")
            generate.mark_unanswerable(item)

    log("=== Running quality checks ===")
    kept, _dropped = quality.quality_filter(items, progress=log)

    log(f"=== Complete: {len(kept)} items ready for review ===")

    existing = load_results() or []
    merged = kept + existing
    save_results(merged)
    log(f"Appended {len(kept)} new items to {len(existing)} existing → {len(merged)} total.")
    return merged


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
