"""
FastAPI wrapper exposing the MedRAGBench pipeline over HTTP.

This is the network boundary the desktop GUI (app.py) never had: it lets a
web/mobile frontend drive the same Stage 0-7 pipeline. It reuses pipeline.py
unchanged — the only new concerns here are (a) accepting PDF *uploads* instead
of local file paths, (b) running the multi-minute generation on a background
thread and streaming progress, and (c) holding review state between requests.

Run it with:

    export OPENAI_API_KEY="sk-..."
    uvicorn medragbench.api:app --reload --port 8000

NOTE: job/review state lives in an in-memory dict, so it is lost on restart and
does not survive multiple server processes. For production, back the JobStore
with a database or object store (see the `JobStore` comment below).
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load medragbench/.env so OPENAI_API_KEY / ANTHROPIC_API_KEY are available no
# matter where uvicorn is launched from. The path is anchored to this file's
# directory rather than the CWD. Harmless if the file or python-dotenv is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from . import config, ingest, pipeline
from .generate import BenchmarkItem


# ==========================================================================
# In-memory job/review store
# --------------------------------------------------------------------------
# Each generation run is a "job". A job owns: the live progress log, the
# generated BenchmarkItems (mutated in place as the clinician edits/approves),
# and a scratch dir holding the uploaded PDFs.
#
# Swap this for a real datastore in production: persist Job rows + items keyed
# by job_id, and store uploads in object storage rather than a temp dir.
# ==========================================================================
class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.pending
    log: List[str] = field(default_factory=list)
    items: List[BenchmarkItem] = field(default_factory=list)
    error: Optional[str] = None
    workdir: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


_JOBS: Dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def _get_job(job_id: str) -> Job:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
    return job


def _get_item(job: Job, item_id: int) -> BenchmarkItem:
    # item_id is the item's index within the job (stable: the list is never
    # reordered after generation). If you move to a DB, give items real UUIDs.
    with job.lock:
        if not (0 <= item_id < len(job.items)):
            raise HTTPException(status_code=404, detail=f"Unknown item {item_id}")
        return job.items[item_id]


# ==========================================================================
# API <-> domain serialization
# ==========================================================================
class ItemModel(BaseModel):
    """A BenchmarkItem as the frontend sees it during Stage 6 review.

    Unlike the exported record (BenchmarkItem.to_record), this also surfaces
    `flags` and `approved` so the reviewer knows what to scrutinize.
    """

    id: int
    question: str
    type: str
    category: str
    difficulty: str
    expected_behavior: str
    gold_answer: str
    flags: List[str]
    approved: bool
    supporting_passages: List[dict]
    source_papers: List[dict]
    retrieval_targets: List[str]


def _to_model(idx: int, it: BenchmarkItem) -> ItemModel:
    return ItemModel(
        id=idx,
        question=it.question,
        type=it.type,
        category=it.category,
        difficulty=it.difficulty,
        expected_behavior=it.expected_behavior,
        gold_answer=it.gold_answer,
        flags=it.flags,
        approved=it.approved,
        supporting_passages=it.supporting_passages,
        source_papers=it.source_papers,
        retrieval_targets=it.retrieval_targets,
    )


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusModel(BaseModel):
    job_id: str
    status: JobStatus
    log: List[str]          # log lines from `since` onward
    log_cursor: int         # pass back as ?since= to get only new lines
    item_count: int
    error: Optional[str] = None


class ItemEdit(BaseModel):
    question: Optional[str] = None
    gold_answer: Optional[str] = None


# ==========================================================================
# App
# ==========================================================================
app = FastAPI(title="MedRAGBench API", version="1.0")

# Allow a separately-served frontend (e.g. Vite dev server) to call the API.
# Restrict allow_origins to your real frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/saved-results")
def get_saved_results() -> dict:
    """Load previously saved pipeline results if available."""
    saved = pipeline.load_results()
    if saved is None:
        return {"has_results": False, "items": []}
    return {
        "has_results": True,
        "items": [_to_model(i, it).model_dump() for i, it in enumerate(saved)],
    }


@app.get("/api/corpus-status")
def get_corpus_status() -> dict:
    """Check if a cached corpus exists that can be reused."""
    meta_path = os.path.join(config.PATHS.workdir, "corpus_meta.json")
    if not os.path.exists(meta_path):
        return {"has_corpus": False}
    try:
        import json as _json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = _json.load(f)
        papers = meta.get("papers", {})
        chunks = meta.get("chunks", [])
        return {
            "has_corpus": True,
            "paper_count": len(papers),
            "chunk_count": len(chunks),
            "papers": list(papers.values()),
        }
    except Exception:
        return {"has_corpus": False}


@app.post("/api/jobs/from-corpus", response_model=JobCreated, status_code=202)
async def create_job_from_corpus() -> JobCreated:
    """Run pipeline using the cached corpus (no PDF upload needed)."""
    corpus = ingest.load_corpus_from_dir(config.PATHS.workdir)
    if corpus is None:
        raise HTTPException(status_code=404, detail="No cached corpus found.")

    job = Job(id=uuid.uuid4().hex)
    job.workdir = config.PATHS.workdir

    with _JOBS_LOCK:
        _JOBS[job.id] = job

    threading.Thread(
        target=_run_job_with_corpus, args=(job, corpus), daemon=True
    ).start()
    return JobCreated(job_id=job.id, status=job.status)


def _run_job_with_corpus(job: Job, corpus) -> None:
    """Background worker using a pre-loaded corpus."""

    def progress(msg: str) -> None:
        with job.lock:
            job.log.append(msg)

    with job.lock:
        job.status = JobStatus.running
    try:
        items = pipeline.run_generation([], progress=progress, preloaded_corpus=corpus)
        with job.lock:
            job.items = items
            job.status = JobStatus.done
    except Exception as exc:
        progress(f"ERROR: {exc}")
        with job.lock:
            job.error = str(exc)
            job.status = JobStatus.error


@app.get("/api/config")
def get_config() -> dict:
    """Taxonomy + limits the frontend needs to render the review UI."""
    return {
        "max_pdfs": config.MAX_PDFS,
        "target_question_count": config.TARGET_QUESTION_COUNT,
        "categories": config.PKD_CATEGORIES,
        "question_types": config.QUESTION_TYPES,
        "questions_per_type": config.QUESTIONS_PER_TYPE,
        "expected_behavior_by_type": config.EXPECTED_BEHAVIOR_BY_TYPE,
    }


# ---- Stage 0-5: upload PDFs and launch generation ------------------------
def _run_job(job: Job, pdf_paths: List[str]) -> None:
    """Background worker: runs Stages 0-5, streaming progress into job.log."""

    def progress(msg: str) -> None:
        with job.lock:
            job.log.append(msg)

    with job.lock:
        job.status = JobStatus.running
    try:
        items = pipeline.run_generation(pdf_paths, progress=progress)
        with job.lock:
            job.items = items
            job.status = JobStatus.done
    except Exception as exc:  # surface to the client instead of crashing
        progress(f"ERROR: {exc}")
        with job.lock:
            job.error = str(exc)
            job.status = JobStatus.error


@app.post("/api/jobs", response_model=JobCreated, status_code=202)
async def create_job(files: List[UploadFile] = File(...)) -> JobCreated:
    """Upload up to MAX_PDFS PDFs and kick off Stages 0-5 in the background."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > config.MAX_PDFS:
        raise HTTPException(
            status_code=400,
            detail=f"{len(files)} PDFs uploaded but MAX_PDFS={config.MAX_PDFS}.",
        )

    job = Job(id=uuid.uuid4().hex)
    job.workdir = tempfile.mkdtemp(prefix=f"medrag_{job.id}_")

    pdf_paths: List[str] = []
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"{f.filename!r} is not a .pdf"
            )
        dest = os.path.join(job.workdir, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        pdf_paths.append(dest)

    with _JOBS_LOCK:
        _JOBS[job.id] = job

    threading.Thread(
        target=_run_job, args=(job, pdf_paths), daemon=True
    ).start()
    return JobCreated(job_id=job.id, status=job.status)


# ---- Progress: poll or stream --------------------------------------------
@app.get("/api/jobs/{job_id}", response_model=JobStatusModel)
def get_job(job_id: str, since: int = 0) -> JobStatusModel:
    """Poll job status. Pass ?since=<log_cursor> to fetch only new log lines."""
    job = _get_job(job_id)
    with job.lock:
        return JobStatusModel(
            job_id=job.id,
            status=job.status,
            log=job.log[since:],
            log_cursor=len(job.log),
            item_count=len(job.items),
            error=job.error,
        )


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream of progress lines until the job finishes."""
    job = _get_job(job_id)

    async def gen():
        sent = 0
        while True:
            with job.lock:
                new = job.log[sent:]
                sent = len(job.log)
                status = job.status
            for line in new:
                yield f"data: {json.dumps({'log': line})}\n\n"
            if status in (JobStatus.done, JobStatus.error):
                yield f"data: {json.dumps({'status': status.value})}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---- Stage 6: review (list / get / edit / approve / reject) --------------
@app.get("/api/jobs/{job_id}/items", response_model=List[ItemModel])
def list_items(job_id: str) -> List[ItemModel]:
    job = _get_job(job_id)
    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409, detail=f"Job is {job.status.value}, not done."
        )
    with job.lock:
        return [_to_model(i, it) for i, it in enumerate(job.items)]


@app.get("/api/jobs/{job_id}/items/{item_id}", response_model=ItemModel)
def get_item(job_id: str, item_id: int) -> ItemModel:
    job = _get_job(job_id)
    return _to_model(item_id, _get_item(job, item_id))


@app.patch("/api/jobs/{job_id}/items/{item_id}", response_model=ItemModel)
def edit_item(job_id: str, item_id: int, edit: ItemEdit) -> ItemModel:
    """Stage 6 edit: clinician revises the question and/or gold answer."""
    job = _get_job(job_id)
    it = _get_item(job, item_id)
    with job.lock:
        if edit.question is not None:
            it.question = edit.question.strip()
        if edit.gold_answer is not None:
            it.gold_answer = edit.gold_answer.strip()
        pipeline.save_results(job.items)
    return _to_model(item_id, it)


@app.post("/api/jobs/{job_id}/items/{item_id}/approve", response_model=ItemModel)
def approve_item(job_id: str, item_id: int) -> ItemModel:
    job = _get_job(job_id)
    it = _get_item(job, item_id)
    with job.lock:
        it.approved = True
        pipeline.save_results(job.items)
    return _to_model(item_id, it)


@app.post("/api/jobs/{job_id}/items/{item_id}/reject", response_model=ItemModel)
def reject_item(job_id: str, item_id: int) -> ItemModel:
    job = _get_job(job_id)
    it = _get_item(job, item_id)
    with job.lock:
        it.approved = False
        pipeline.save_results(job.items)
    return _to_model(item_id, it)


# ---- Stage 7: export approved items --------------------------------------
@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str) -> dict:
    """Return the benchmark JSON of approved items (same shape pipeline writes)."""
    job = _get_job(job_id)
    with job.lock:
        approved = [it for it in job.items if it.approved]
        if not approved:
            raise HTTPException(status_code=400, detail="No approved items.")
        return {
            "benchmark": "MedRAGBench",
            "version": 1,
            "categories": config.PKD_CATEGORIES,
            "question_types": config.QUESTION_TYPES,
            "count": len(approved),
            "records": [it.to_record() for it in approved],
        }


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    """Drop a job and clean up its uploaded PDFs."""
    with _JOBS_LOCK:
        job = _JOBS.pop(job_id, None)
    if job and job.workdir and os.path.isdir(job.workdir):
        shutil.rmtree(job.workdir, ignore_errors=True)


# ==========================================================================
# Static frontend (production)
# --------------------------------------------------------------------------
# Serve the built React app (frontend/dist, produced by `npm run build`) at the
# root, so the whole app runs from a single `uvicorn` command and the API is
# same-origin. Mounted LAST so the /api/* routes above take precedence over the
# catch-all static handler.
#
# In DEVELOPMENT you don't need this: run `npm run dev` in frontend/ (Vite on
# :5173) which hot-reloads and proxies /api to this server on :8000. The dist
# dir only exists after a build, so this mount is simply skipped until then.
# ==========================================================================
from fastapi.staticfiles import StaticFiles  # noqa: E402

_DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_DIST_DIR):
    app.mount(
        "/", StaticFiles(directory=_DIST_DIR, html=True), name="frontend"
    )
