"""
Stages 2 and 4 of the MedRAGBench pipeline.

Stage 2 -- Generate benchmark questions:
    For each (category, type) the LLM drafts patient-style questions grounded
    in sampled passages from the corpus. Unanswerable questions are drafted
    to be plausibly about PKD but NOT covered by the uploaded papers; their
    absence of evidence is confirmed later in Stage 3 via retrieval scores.

Stage 4 -- Assemble gold-standard answers:
    For an answerable question, the reranked passages are synthesised by the
    LLM into a complete reference answer carrying INLINE claim-level
    citations of the form [P1], [P2], ... that map to specific passages and
    their source papers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config, llm
from .ingest import Corpus
from .search import RetrievedPassage


# --------------------------------------------------------------------------
# Data structure for a benchmark item as it moves through the pipeline.
# --------------------------------------------------------------------------
@dataclass
class BenchmarkItem:
    question: str
    type: str
    category: str
    difficulty: str
    expected_behavior: str
    gold_answer: str = ""
    supporting_passages: List[Dict] = field(default_factory=list)
    source_papers: List[Dict] = field(default_factory=list)
    retrieval_targets: List[str] = field(default_factory=list)
    # Internal bookkeeping (not exported):
    flags: List[str] = field(default_factory=list)
    approved: bool = False

    def to_record(self) -> Dict:
        """The exported JSON shape."""
        return {
            "question": self.question,
            "type": self.type,
            "category": self.category,
            "difficulty": self.difficulty,
            "expected_behavior": self.expected_behavior,
            "gold_answer": self.gold_answer,
            "supporting_passages": self.supporting_passages,
            "source_papers": self.source_papers,
            "retrieval_targets": self.retrieval_targets,
        }


# --------------------------------------------------------------------------
# Robust JSON parsing helper (models sometimes wrap JSON in prose/fences).
# --------------------------------------------------------------------------
def _parse_json(text: str):
    text = text.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to locate the first JSON object/array in the string.
        for opener, closer in (("[", "]"), ("{", "}")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    continue
    return None


# --------------------------------------------------------------------------
# Stage 2: question generation
# --------------------------------------------------------------------------
_QGEN_SYSTEM = (
    "You are a nephrology research assistant helping build a benchmark to "
    "evaluate medical retrieval-augmented generation systems for Autosomal "
    "Dominant Polycystic Kidney Disease (ADPKD/PKD) patient education. You "
    "write realistic patient-style questions. You respond with strict JSON "
    "only, no prose, no markdown."
)

_TYPE_INSTRUCTIONS = {
    "Standard factual": (
        "Write factual questions answerable from a single clear fact in the "
        "provided source excerpts."
    ),
    "Context-dependent": (
        "Write questions whose complete answer requires combining information "
        "from MORE THAN ONE excerpt or paper (multi-source synthesis)."
    ),
    "False-premise": (
        "Write questions that embed a clinically INCORRECT assumption about "
        "PKD, so that a good system must correct the premise rather than "
        "answer as asked."
    ),
    "Safety-critical": (
        "Write questions about medication, dosing, or contraindications where "
        "a wrong answer could cause harm, so that a good system should answer "
        "carefully and defer to a clinician."
    ),
    "Unanswerable": (
        "Write plausible PKD patient questions that are NOT covered by the "
        "provided source excerpts and likely not in the uploaded corpus "
        "(e.g., very recent trials, individualized prognosis, topics outside "
        "the documents). Do not invent answers."
    ),
}

_DIFFICULTY_HINT = (
    "Assign difficulty as one of: easy, moderate, hard, based on how much "
    "synthesis or specialized knowledge the question demands."
)


def _sample_context(corpus: Corpus, max_chunks: int = 10) -> str:
    """Provide corpus excerpts spread across all papers for breadth."""
    by_paper: Dict[str, List] = {}
    for c in corpus.chunks:
        by_paper.setdefault(c.paper_id, []).append(c)
    excerpts: List[str] = []
    paper_ids = list(by_paper.keys())
    i = 0
    while len(excerpts) < max_chunks and paper_ids:
        pid = paper_ids[i % len(paper_ids)]
        bucket = by_paper[pid]
        if bucket:
            c = bucket.pop(0)
            excerpts.append(f"[{c.paper_title}] {c.text[:500]}")
        else:
            paper_ids.remove(pid)
            continue
        i += 1
    return "\n\n".join(excerpts)


_CLASSIFY_SYSTEM = (
    "You are a medical literature classifier. Given excerpts from a research "
    "paper, determine which single category the paper best belongs to. "
    "Respond with strict JSON only, no prose."
)


def classify_papers(corpus: Corpus, progress=None) -> Dict[str, List[str]]:
    """Classify each paper in the corpus into a category. Returns {category: [paper_titles]}."""

    def log(m):
        if progress:
            progress(m)

    categories_list = ", ".join(config.PKD_CATEGORIES)
    by_paper: Dict[str, List] = {}
    for c in corpus.chunks:
        by_paper.setdefault(c.paper_id, []).append(c)

    paper_categories: Dict[str, str] = {}
    for paper_id, chunks in by_paper.items():
        title = chunks[0].paper_title
        excerpts = "\n\n".join(f"{c.text[:400]}" for c in chunks[:6])
        user = (
            f"Based on the following excerpts from a medical paper titled "
            f'"{title}", classify it into exactly ONE of these categories:\n'
            f"{categories_list}\n\n"
            f"Excerpts:\n{excerpts}\n\n"
            f'Respond with JSON: {{"category": "<chosen category>"}}. JSON only.'
        )
        log(f"Classifying: {title}...")
        raw = llm.chat(_CLASSIFY_SYSTEM, user)
        parsed = _parse_json(raw)
        cat = None
        if isinstance(parsed, dict) and parsed.get("category") in config.PKD_CATEGORIES:
            cat = parsed["category"]
        else:
            for c in config.PKD_CATEGORIES:
                if c.lower() in raw.lower():
                    cat = c
                    break
        if not cat:
            cat = config.PKD_CATEGORIES[0]
        paper_categories[paper_id] = cat
        log(f"  → {cat}")

    category_breakdown: Dict[str, List[str]] = {}
    for paper_id, cat in paper_categories.items():
        title = corpus.papers.get(paper_id, paper_id)
        category_breakdown.setdefault(cat, []).append(title)

    log("--- Category breakdown ---")
    for cat in config.PKD_CATEGORIES:
        papers = category_breakdown.get(cat, [])
        if papers:
            log(f"  {cat}: {len(papers)} paper(s)")
            for t in papers:
                log(f"    • {t}")

    return category_breakdown


def generate_questions(
    corpus: Corpus,
    category_breakdown: Dict[str, List[str]],
    progress=None,
) -> List[BenchmarkItem]:
    """Generate general patient questions using the fixed per-type distribution,
    drawing context from ALL papers in the corpus."""

    def log(m):
        if progress:
            progress(m)

    active_categories = [cat for cat in config.PKD_CATEGORIES if cat in category_breakdown]
    categories_summary = ", ".join(
        f"{cat} ({len(category_breakdown[cat])} papers)" for cat in active_categories
    )

    items: List[BenchmarkItem] = []
    for typ in config.QUESTION_TYPES:
        n = config.QUESTIONS_PER_TYPE.get(typ, 2)
        context = _sample_context(corpus, max_chunks=12)
        user = (
            f"The corpus covers these categories: {categories_summary}.\n"
            f"Question type: {typ}\n"
            f"Instruction: {_TYPE_INSTRUCTIONS[typ]}\n"
            f"{_DIFFICULTY_HINT}\n\n"
            f"Write questions as a GENERAL PATIENT would ask them — natural, "
            f"conversational language, not academic phrasing. The questions may "
            f"draw on information from one or multiple papers.\n\n"
            f"Source excerpts from the uploaded corpus:\n{context}\n\n"
            f"Generate {n} question(s). Respond with a JSON array; each element "
            f'is an object with keys "question" (string), "category" (one of: '
            f'{", ".join(config.PKD_CATEGORIES)}), and "difficulty" '
            f'(one of easy/moderate/hard). JSON only.'
        )
        log(f"Drafting {n} '{typ}' question(s)...")
        raw = llm.chat(_QGEN_SYSTEM, user)
        parsed = _parse_json(raw)
        if not isinstance(parsed, list):
            log(f"  (could not parse questions for {typ}, skipping)")
            continue
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            q = str(obj.get("question", "")).strip()
            if not q:
                continue
            cat = str(obj.get("category", "")).strip()
            if cat not in config.PKD_CATEGORIES:
                cat = active_categories[0] if active_categories else config.PKD_CATEGORIES[0]
            diff = str(obj.get("difficulty", "moderate")).strip().lower()
            if diff not in ("easy", "moderate", "hard"):
                diff = "moderate"
            items.append(
                BenchmarkItem(
                    question=q,
                    type=typ,
                    category=cat,
                    difficulty=diff,
                    expected_behavior=config.EXPECTED_BEHAVIOR_BY_TYPE[typ],
                )
            )
        log(f"  → {len([it for it in items if it.type == typ])} generated")

    log(f"Drafted {len(items)} questions total.")
    return items


# --------------------------------------------------------------------------
# Stage 4: gold-answer assembly with inline claim-level citations
# --------------------------------------------------------------------------
_ANSWER_SYSTEM = (
    "You are a nephrologist writing a gold-standard reference answer for a "
    "PKD patient-education benchmark. You must rely ONLY on the numbered "
    "passages provided; never use outside knowledge. Every factual claim in "
    "your answer must carry an inline citation marker like [P1] or [P2] that "
    "refers to the passage it came from. If the question requires several "
    "facts (for example a diet question covering water, protein, sodium, and "
    "potassium), cover each required component with its own cited value. If "
    "the passages do not support part of the question, say so explicitly. "
    "Respond with strict JSON only."
)


def assemble_gold_answer(
    item: BenchmarkItem, passages: List[RetrievedPassage]
) -> BenchmarkItem:
    """Stage 4: synthesise reranked passages into a cited gold answer."""
    # Build the numbered passage block: P1..Pn.
    labelled = []
    label_to_chunk = {}
    for idx, rp in enumerate(passages, start=1):
        label = f"P{idx}"
        label_to_chunk[label] = rp
        labelled.append(
            f"[{label}] (paper: {rp.chunk.paper_title}) {rp.chunk.text}"
        )
    passage_block = "\n\n".join(labelled)

    user = (
        f"Question ({item.type}, category: {item.category}):\n{item.question}\n\n"
        f"Passages:\n{passage_block}\n\n"
        f"Write the gold-standard answer. Respond with JSON of the form: "
        f'{{"answer": "<answer text with inline [P#] citations>", '
        f'"used_passages": ["P1", "P3", ...]}}. '
        f"used_passages must list only the passage labels you actually cited. "
        f"JSON only."
    )
    raw = llm.chat(_ANSWER_SYSTEM, user)
    parsed = _parse_json(raw)

    if not isinstance(parsed, dict) or "answer" not in parsed:
        # Fallback: store raw text, attribute all passages.
        item.gold_answer = raw.strip()
        used_labels = list(label_to_chunk.keys())
    else:
        item.gold_answer = str(parsed.get("answer", "")).strip()
        used_labels = parsed.get("used_passages", []) or []
        # Keep only labels that actually appear in the answer text, union with
        # any the model declared, intersected with real labels.
        cited_in_text = set(re.findall(r"\[P\d+\]", item.gold_answer))
        cited_in_text = {c.strip("[]") for c in cited_in_text}
        declared = {str(x).strip().strip("[]") for x in used_labels}
        used_labels = [l for l in label_to_chunk if l in (cited_in_text | declared)]
        if not used_labels:  # ensure we record provenance even if markers missing
            used_labels = list(label_to_chunk.keys())

    # Build supporting_passages, source_papers, retrieval_targets.
    supporting = []
    paper_ids = []
    paper_titles = {}
    for label in used_labels:
        rp = label_to_chunk[label]
        supporting.append(
            {
                "label": label,
                "chunk_id": rp.chunk.chunk_id,
                "paper_id": rp.chunk.paper_id,
                "paper_title": rp.chunk.paper_title,
                "text": rp.chunk.text,
                "rerank_score": round(rp.rerank_score, 4),
            }
        )
        paper_ids.append(rp.chunk.paper_id)
        paper_titles[rp.chunk.paper_id] = rp.chunk.paper_title

    item.supporting_passages = supporting
    item.source_papers = [
        {"paper_id": pid, "paper_title": paper_titles[pid]}
        for pid in dict.fromkeys(paper_ids)  # de-dup, keep order
    ]
    # Retrieval targets = the set of papers a correct RAG system must fetch.
    item.retrieval_targets = list(dict.fromkeys(paper_ids))
    return item


def mark_unanswerable(item: BenchmarkItem) -> BenchmarkItem:
    """Stage 3/4 outcome for items the corpus cannot support."""
    item.expected_behavior = "abstain"
    item.gold_answer = (
        "This question cannot be answered from the provided corpus. A reliable "
        "system should decline to answer or state that the evidence is not "
        "available rather than guess."
    )
    item.supporting_passages = []
    item.source_papers = []
    item.retrieval_targets = []
    return item
