# MedRAGBench

A desktop/web application that **creates a medical RAG benchmark dataset** from a
set of uploaded PDF papers. MedRAGBench does not answer patient questions
itself — it produces a dataset of questions, clinician-approved gold answers,
and retrieval targets that you then use to **evaluate** retrieval-augmented
generation (RAG) systems such as CysticCare, GPT, Gemini, or Claude.

It implements the Stage 0–7 pipeline:

| Stage | What it does |
|------:|--------------|
| 0 | Upload up to 100 PDFs (configurable). |
| 1 | Extract text, chunk, build a **dense index (ChromaDB)** and a **BM25 keyword index**. |
| 2 | Generate questions across 5 PKD categories and 5 question types. |
| 3 | For each answerable question, find evidence by **hybrid search → Reciprocal Rank Fusion → cross-encoder rerank**, with a **sufficiency threshold** that flags unanswerable items. |
| 4 | Assemble a **gold-standard answer with inline `[P#]` claim-level citations** linking each claim to its source passage and paper. |
| 5 | **Automated quality filter**: recall-risk detection, schema completeness, semantic dedup, unanswerable confirmation. |
| 6 | **Clinician review** in the GUI: edit, approve, or reject each item. |
| 7 | **Export** approved items to a structured JSON benchmark file. |

---

## 1. Requirements

- Python 3.9+
- An OpenAI API key (for embeddings and question/answer generation)
- ~1–2 GB free disk for the cross-encoder model download on first run

### Tkinter

The GUI uses Tkinter, which ships with most Python installers. If you get
`ModuleNotFoundError: No module named 'tkinter'`:

- **Ubuntu/Debian:** `sudo apt-get install python3-tk`
- **Fedora:** `sudo dnf install python3-tkinter`
- **macOS (Homebrew):** `brew install python-tk`
- **Windows:** reinstall Python from python.org with the "tcl/tk" option checked.

---

## 2. Install

```bash
cd medragbench
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the cross-encoder model
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 MB) automatically.

---

## 3. Set your API key

MedRAGBench reads the key from an environment variable. **Do not paste keys
into the code.**

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$Env:OPENAI_API_KEY = "sk-..."
```

---

## 4. Run

```bash
python run.py
```

Then in the window:

1. **Choose PDFs…** — select up to 10 PDF papers.
2. **Run pipeline (Stages 0–5)** — watch the log on the left. This calls the
   OpenAI API and runs the cross-encoder, so it takes a few minutes.
3. **Review** each generated item on the right (Stage 6): read the question,
   the gold answer, and the supporting passages; edit if needed; **Approve**
   or **Reject**. Flags from the quality filter (e.g. `recall_risk`,
   `near_duplicate`) are shown so you know what to scrutinize.
4. **Export approved → JSON (Stage 7)** — writes the benchmark file.

### Output format

Each exported record has exactly these fields:

```json
{
  "question": "...",
  "type": "Standard factual | Context-dependent | False-premise | Safety-critical | Unanswerable",
  "category": "Disease mechanisms | Diagnosis and imaging | Treatment and medication | Diet and lifestyle | Progression and prognosis",
  "difficulty": "easy | moderate | hard",
  "expected_behavior": "answer | answer_with_synthesis | correct_the_premise | answer_with_safety_caveat_or_defer | abstain",
  "gold_answer": "Answer text with inline [P1] [P2] citations ...",
  "supporting_passages": [
    {"label": "P1", "chunk_id": "...", "paper_id": "...", "paper_title": "...", "text": "...", "rerank_score": 5.12}
  ],
  "source_papers": [{"paper_id": "...", "paper_title": "..."}],
  "retrieval_targets": ["paper_id_1", "paper_id_2"]
}
```

`retrieval_targets` is the list of papers a correct RAG system must retrieve to
answer the question — this is what your evaluation scores against.

---

## 5. Scaling from 10 to 100 PDFs

Open `medragbench/config.py` and change **one line**:

```python
MAX_PDFS: int = 10      # change to 100
```

You will probably also want to raise the number of questions drafted:

```python
TARGET_QUESTION_COUNT: int = 25   # raise toward ~100 for a larger corpus
```

Nothing else needs to change. Embedding and indexing already run in batches.

---

## 6. Switching to an other LLM

MedRAGBench routes **all chat generation** (question drafting, gold-answer
assembly, and the no-retrieval recall check) through a single function,
`llm.chat(system, user)`. Switching the generator to different LLM, for example Anthropic Claude, is
therefore a small, localized change. Embeddings stay on OpenAI, because
Anthropic does not provide an embeddings endpoint.

**Step 1 — install the Anthropic SDK.** In `requirements.txt`, uncomment:

```
anthropic>=0.39.0
```

then `pip install -r requirements.txt`.

**Step 2 — set the provider.** In `medragbench/config.py`:

```python
LLM_PROVIDER: str = "anthropic"     # was "openai"
```

(or, without editing files, `export MEDRAGBENCH_PROVIDER=anthropic`.)

**Step 3 — set both keys.** You still need the OpenAI key for embeddings:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Step 4 — (optional) pick the model.** In `medragbench/config.py`:

```python
ANTHROPIC_CHAT_MODEL: str = "claude-opus-4-1"
```

### What actually changes under the hood

The relevant code already exists in `medragbench/llm.py`. The dispatcher:

```python
def chat(system: str, user: str, max_retries: int = 4) -> str:
    if config.LLM_PROVIDER == "anthropic":
        return _chat_anthropic(system, user, max_retries)
    return _chat_openai(system, user, max_retries)
```

The OpenAI call uses `chat.completions.create` with the system prompt as a
message:

```python
resp = client.chat.completions.create(
    model=config.OPENAI_CHAT_MODEL,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
)
return resp.choices[0].message.content
```

The Anthropic call uses `messages.create`, where the **system prompt is a
separate top-level argument** (not a message) and the **response is a list of
content blocks** you concatenate:

```python
resp = client.messages.create(
    model=config.ANTHROPIC_CHAT_MODEL,
    max_tokens=4096,
    system=system,                                  # separate argument
    messages=[{"role": "user", "content": user}],
)
return "".join(b.text for b in resp.content if b.type == "text")
```

That is the entire difference. No stage of the pipeline (generation, answer
assembly, quality filter) needs editing, because they all call `llm.chat`.

---

## 7. Project layout

```
medragbench/
  run.py                  # launcher
  requirements.txt
  README.md
  medragbench/
    __init__.py
    config.py             # all settings: scaling, provider, thresholds
    llm.py                # OpenAI + Anthropic wrappers (chat, embeddings)
    ingest.py             # Stage 0-1: extract, chunk, dense + BM25 index
    search.py             # Stage 3: hybrid search, RRF, rerank, sufficiency
    generate.py           # Stage 2 + 4: question gen, gold-answer assembly
    quality.py            # Stage 5: automated quality filter
    pipeline.py           # orchestrates 0-5, plus Stage 7 export
    app.py                # Tkinter GUI (Stage 6 review + controls)
```

---

## 8. Notes and limitations

- The sufficiency threshold (`SUFFICIENCY_THRESHOLD` in `config.py`) is the
  cross-encoder relevance score below which the corpus is judged to lack
  evidence. The default of `0.0` is a reasonable starting point for the
  ms-marco-MiniLM model; tune it against your own papers.
- Unanswerable items are confirmed by **low retrieval scores**, not by
  assertion: even questions drafted as "Unanswerable" are run through hybrid
  search, and if strong evidence is unexpectedly found, the item is flagged
  for clinician attention rather than silently kept.
- All clinician decisions are made in Stage 6; nothing is exported until you
  approve it.
```
