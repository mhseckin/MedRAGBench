"""
MedRAGBench desktop GUI (Tkinter).

Three areas:
  * Top    -- choose PDFs (up to MAX_PDFS) and run the generation pipeline.
  * Left   -- live progress log from the worker thread.
  * Right  -- Stage 6 clinician review: step through generated items, edit
              the question / gold answer, then Approve, Reject, or Skip.
  * Bottom -- Stage 7 export of approved items to JSON.

The pipeline runs on a background thread so the UI stays responsive; the
worker posts progress strings and the final item list back to the Tk main
thread through a queue.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from . import config, pipeline
from .generate import BenchmarkItem


class MedRAGBenchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MedRAGBench — Medical RAG Benchmark Builder")
        self.root.geometry("1180x760")

        self.pdf_paths: List[str] = []
        self.items: List[BenchmarkItem] = []
        self.review_index: int = 0

        self._msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._build_ui()
        self._poll_queue()

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        # Top control bar
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Choose PDFs…", command=self._choose_pdfs).pack(
            side=tk.LEFT
        )
        self.pdf_label = ttk.Label(top, text="No PDFs selected")
        self.pdf_label.pack(side=tk.LEFT, padx=10)

        self.run_btn = ttk.Button(
            top, text="Run pipeline (Stages 0–5)", command=self._run_pipeline
        )
        self.run_btn.pack(side=tk.LEFT, padx=10)

        provider = config.LLM_PROVIDER
        ttk.Label(top, text=f"Provider: {provider}").pack(side=tk.RIGHT)

        # Main split: log (left) | review (right)
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: log
        left = ttk.Frame(main)
        ttk.Label(left, text="Pipeline log").pack(anchor=tk.W)
        self.log_text = tk.Text(left, width=46, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        main.add(left, weight=1)

        # Right: review
        right = ttk.Frame(main)
        self._build_review(right)
        main.add(right, weight=2)

        # Bottom: export + status
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.export_btn = ttk.Button(
            bottom, text="Export approved → JSON (Stage 7)", command=self._export
        )
        self.export_btn.pack(side=tk.LEFT)
        self.status = ttk.Label(bottom, text="Ready.")
        self.status.pack(side=tk.LEFT, padx=12)

    def _build_review(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent, text="Stage 6 — Clinician review", font=("", 11, "bold")
        ).pack(anchor=tk.W)

        self.review_meta = ttk.Label(parent, text="No items yet.", wraplength=640)
        self.review_meta.pack(anchor=tk.W, pady=(2, 6))

        ttk.Label(parent, text="Question (editable):").pack(anchor=tk.W)
        self.q_edit = tk.Text(parent, height=3, wrap=tk.WORD)
        self.q_edit.pack(fill=tk.X)

        ttk.Label(parent, text="Gold answer (editable):").pack(
            anchor=tk.W, pady=(6, 0)
        )
        self.a_edit = tk.Text(parent, height=10, wrap=tk.WORD)
        self.a_edit.pack(fill=tk.BOTH, expand=False)

        ttk.Label(parent, text="Supporting passages / source papers:").pack(
            anchor=tk.W, pady=(6, 0)
        )
        self.evidence_text = tk.Text(
            parent, height=8, wrap=tk.WORD, state=tk.DISABLED
        )
        self.evidence_text.pack(fill=tk.BOTH, expand=True)

        # Review action buttons
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=6)
        ttk.Button(actions, text="◀ Prev", command=self._prev_item).pack(
            side=tk.LEFT
        )
        ttk.Button(actions, text="Approve ✓", command=self._approve).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Reject ✗", command=self._reject).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Save edits", command=self._save_edits).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(actions, text="Next ▶", command=self._next_item).pack(
            side=tk.LEFT, padx=4
        )

    # -------------------------------------------------------------- actions
    def _choose_pdfs(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select up to %d PDF files" % config.MAX_PDFS,
            filetypes=[("PDF files", "*.pdf")],
        )
        paths = list(paths)
        if len(paths) > config.MAX_PDFS:
            messagebox.showwarning(
                "Too many PDFs",
                f"You selected {len(paths)} files but the limit is "
                f"{config.MAX_PDFS}. Only the first {config.MAX_PDFS} will be used. "
                f"(Raise MAX_PDFS in config.py to scale up.)",
            )
            paths = paths[: config.MAX_PDFS]
        self.pdf_paths = paths
        self.pdf_label.config(text=f"{len(paths)} PDF(s) selected")

    def _run_pipeline(self) -> None:
        if not self.pdf_paths:
            messagebox.showinfo("No PDFs", "Choose at least one PDF first.")
            return
        if self._worker and self._worker.is_alive():
            return
        self.run_btn.config(state=tk.DISABLED)
        self.status.config(text="Running pipeline…")
        self._log("Starting pipeline…")

        def work():
            try:
                items = pipeline.run_generation(
                    self.pdf_paths,
                    progress=lambda m: self._msg_queue.put(("log", m)),
                )
                self._msg_queue.put(("done", items))
            except Exception as exc:  # surface errors to the UI
                self._msg_queue.put(("error", str(exc)))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self.items = payload
                    self.review_index = 0
                    self.run_btn.config(state=tk.NORMAL)
                    self.status.config(
                        text=f"Generated {len(self.items)} items. Review them."
                    )
                    self._log(f"Pipeline finished: {len(self.items)} items.")
                    self._refresh_review()
                elif kind == "error":
                    self.run_btn.config(state=tk.NORMAL)
                    self.status.config(text="Error (see log).")
                    self._log(f"ERROR: {payload}")
                    messagebox.showerror("Pipeline error", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _log(self, msg: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # --------------------------------------------------------- review logic
    def _current(self) -> Optional[BenchmarkItem]:
        if 0 <= self.review_index < len(self.items):
            return self.items[self.review_index]
        return None

    def _refresh_review(self) -> None:
        item = self._current()
        if item is None:
            self.review_meta.config(text="No items to review.")
            self.q_edit.delete("1.0", tk.END)
            self.a_edit.delete("1.0", tk.END)
            self._set_evidence("")
            return

        flags = (", ".join(item.flags)) if item.flags else "none"
        approved = "APPROVED" if item.approved else "not yet approved"
        self.review_meta.config(
            text=(
                f"Item {self.review_index + 1}/{len(self.items)}  |  "
                f"type: {item.type}  |  category: {item.category}  |  "
                f"difficulty: {item.difficulty}  |  expected: {item.expected_behavior}\n"
                f"flags: {flags}  |  status: {approved}"
            )
        )
        self.q_edit.delete("1.0", tk.END)
        self.q_edit.insert(tk.END, item.question)
        self.a_edit.delete("1.0", tk.END)
        self.a_edit.insert(tk.END, item.gold_answer)

        # Evidence display
        lines = []
        for sp in item.supporting_passages:
            lines.append(
                f"[{sp['label']}] {sp['paper_title']} "
                f"(score {sp.get('rerank_score', 'n/a')})\n{sp['text'][:400]}…\n"
            )
        if not lines:
            lines.append("(no supporting passages — unanswerable/abstain item)")
        self._set_evidence("\n".join(lines))

    def _set_evidence(self, text: str) -> None:
        self.evidence_text.config(state=tk.NORMAL)
        self.evidence_text.delete("1.0", tk.END)
        self.evidence_text.insert(tk.END, text)
        self.evidence_text.config(state=tk.DISABLED)

    def _save_edits(self) -> None:
        item = self._current()
        if item is None:
            return
        item.question = self.q_edit.get("1.0", tk.END).strip()
        item.gold_answer = self.a_edit.get("1.0", tk.END).strip()
        self.status.config(text="Edits saved to current item.")

    def _approve(self) -> None:
        item = self._current()
        if item is None:
            return
        self._save_edits()
        item.approved = True
        self.status.config(text=f"Approved item {self.review_index + 1}.")
        self._next_item()

    def _reject(self) -> None:
        item = self._current()
        if item is None:
            return
        item.approved = False
        self.status.config(text=f"Rejected item {self.review_index + 1}.")
        self._next_item()

    def _next_item(self) -> None:
        if self.review_index < len(self.items) - 1:
            self.review_index += 1
        self._refresh_review()

    def _prev_item(self) -> None:
        if self.review_index > 0:
            self.review_index -= 1
        self._refresh_review()

    # --------------------------------------------------------------- export
    def _export(self) -> None:
        approved = [it for it in self.items if it.approved]
        if not approved:
            messagebox.showinfo(
                "Nothing to export", "Approve at least one item before exporting."
            )
            return
        path = filedialog.asksaveasfilename(
            title="Export benchmark JSON",
            defaultextension=".json",
            initialfile="benchmark.json",
            initialdir=config.PATHS.workdir,
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        n = pipeline.export_dataset(self.items, path)
        self.status.config(text=f"Exported {n} approved records → {path}")
        messagebox.showinfo("Exported", f"Wrote {n} records to:\n{path}")


def main() -> None:
    root = tk.Tk()
    MedRAGBenchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
