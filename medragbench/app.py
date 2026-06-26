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
    BG = "#F0F4F8"
    CARD_BG = "#FFFFFF"
    HEADER_BG = "#1E3A5F"
    HEADER_FG = "#FFFFFF"
    ACCENT = "#2B7A78"
    ACCENT_HOVER = "#3AAFA9"
    DANGER = "#E74C3C"
    TEXT = "#2C3E50"
    TEXT_MUTED = "#7F8C8D"
    BORDER = "#DEE2E6"
    LOG_BG = "#FAFBFC"
    FIELD_BG = "#FFFFFF"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MedRAGBench — Medical RAG Benchmark Builder")
        self.root.geometry("1180x760")
        self.root.configure(bg=self.BG)

        self.pdf_paths: List[str] = []
        self.items: List[BenchmarkItem] = []
        self.review_index: int = 0

        self._msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._configure_styles()
        self._build_ui()
        self._poll_queue()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=self.BG, foreground=self.TEXT,
                         font=("Helvetica", 11))
        style.configure("TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.CARD_BG)
        style.configure("Header.TFrame", background=self.HEADER_BG)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT,
                         font=("Helvetica", 11))
        style.configure("Header.TLabel", background=self.HEADER_BG,
                         foreground=self.HEADER_FG, font=("Helvetica", 13, "bold"))
        style.configure("Section.TLabel", background=self.CARD_BG,
                         foreground=self.HEADER_BG, font=("Helvetica", 12, "bold"))
        style.configure("Field.TLabel", background=self.CARD_BG,
                         foreground=self.TEXT_MUTED, font=("Helvetica", 10))
        style.configure("Meta.TLabel", background=self.CARD_BG,
                         foreground=self.TEXT_MUTED, font=("Helvetica", 10))
        style.configure("Status.TLabel", background=self.BG,
                         foreground=self.ACCENT, font=("Helvetica", 10))
        style.configure("Card.TLabel", background=self.CARD_BG,
                         foreground=self.TEXT)

        style.configure("Accent.TButton", background=self.ACCENT,
                         foreground="#FFFFFF", font=("Helvetica", 10, "bold"),
                         padding=(14, 6))
        style.map("Accent.TButton",
                   background=[("active", self.ACCENT_HOVER)])
        style.configure("TButton", padding=(10, 5), font=("Helvetica", 10))
        style.configure("Danger.TButton", background=self.DANGER,
                         foreground="#FFFFFF", font=("Helvetica", 10, "bold"),
                         padding=(10, 5))
        style.map("Danger.TButton",
                   background=[("active", "#C0392B")])
        style.configure("Nav.TButton", padding=(8, 4), font=("Helvetica", 10))

        style.configure("TPanedwindow", background=self.BG)

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        # Header bar
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(16, 10))
        header.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(header, text="MedRAGBench", style="Header.TLabel").pack(
            side=tk.LEFT)

        ttk.Button(header, text="Choose PDFs...", style="Accent.TButton",
                    command=self._choose_pdfs).pack(side=tk.RIGHT, padx=(8, 0))
        self.run_btn = ttk.Button(
            header, text="Run Pipeline", style="Accent.TButton",
            command=self._run_pipeline)
        self.run_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self.pdf_label = ttk.Label(header, text="No PDFs selected",
                                    style="Header.TLabel",
                                    font=("Helvetica", 10))
        self.pdf_label.pack(side=tk.RIGHT, padx=(8, 0))

        # Main split: log (left) | review (right)
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Left: log card
        left_card = ttk.Frame(main, style="Card.TFrame", padding=12)
        ttk.Label(left_card, text="Pipeline Log", style="Section.TLabel").pack(
            anchor=tk.W, pady=(0, 8))
        self.log_text = tk.Text(left_card, width=40, wrap=tk.WORD,
                                 state=tk.DISABLED, bg=self.LOG_BG, fg=self.TEXT,
                                 font=("Menlo", 10), relief=tk.FLAT,
                                 borderwidth=0, padx=8, pady=8,
                                 highlightthickness=1,
                                 highlightbackground=self.BORDER)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        main.add(left_card, weight=1)

        # Right: review card
        right_card = ttk.Frame(main, style="Card.TFrame", padding=12)
        self._build_review(right_card)
        main.add(right_card, weight=2)

        # Bottom: export + status
        bottom = ttk.Frame(self.root, padding=(12, 6))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.export_btn = ttk.Button(
            bottom, text="Export Approved Results",
            style="Accent.TButton", command=self._export)
        self.export_btn.pack(side=tk.LEFT)
        self.status = ttk.Label(bottom, text="Ready.", style="Status.TLabel")
        self.status.pack(side=tk.LEFT, padx=16)

    def _build_review(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Clinician Review",
                   style="Section.TLabel").pack(anchor=tk.W)

        self.review_meta = ttk.Label(parent, text="No items yet.",
                                      wraplength=640, style="Meta.TLabel")
        self.review_meta.pack(anchor=tk.W, pady=(4, 10))

        text_opts = dict(wrap=tk.WORD, bg=self.FIELD_BG, fg=self.TEXT,
                          font=("Helvetica", 11), relief=tk.FLAT, borderwidth=0,
                          padx=8, pady=6, highlightthickness=1,
                          highlightbackground=self.BORDER,
                          insertbackground=self.ACCENT)

        ttk.Label(parent, text="QUESTION", style="Field.TLabel").pack(
            anchor=tk.W, pady=(0, 2))
        self.q_edit = tk.Text(parent, height=3, **text_opts)
        self.q_edit.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(parent, text="GOLD ANSWER", style="Field.TLabel").pack(
            anchor=tk.W, pady=(0, 2))
        self.a_edit = tk.Text(parent, height=10, **text_opts)
        self.a_edit.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        ttk.Label(parent, text="SUPPORTING PASSAGES / SOURCE PAPERS",
                   style="Field.TLabel").pack(anchor=tk.W, pady=(0, 2))
        self.evidence_text = tk.Text(parent, height=8, state=tk.DISABLED,
                                      **{**text_opts, "bg": self.LOG_BG})
        self.evidence_text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(actions, text="Prev", style="Nav.TButton",
                    command=self._prev_item).pack(side=tk.LEFT)
        ttk.Button(actions, text="Approve", style="Accent.TButton",
                    command=self._approve).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Reject", style="Danger.TButton",
                    command=self._reject).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Edits", style="TButton",
                    command=self._save_edits).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Next", style="Nav.TButton",
                    command=self._next_item).pack(side=tk.LEFT, padx=6)

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
