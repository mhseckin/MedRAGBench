import { useRef, useState } from "react";

// Stage 0: pick up to MAX_PDFS PDFs and launch the pipeline.
export default function UploadView({ config, onRun, onToast }) {
  const [files, setFiles] = useState([]);
  const [dragover, setDragover] = useState(false);
  const inputRef = useRef(null);

  if (!config) {
    return (
      <div className="card">
        <p className="muted">Loading…</p>
      </div>
    );
  }

  const accept = (incoming) => {
    let next = Array.from(incoming).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    );
    if (next.length > config.max_pdfs) {
      onToast(`Limit is ${config.max_pdfs} PDFs; keeping the first ${config.max_pdfs}.`);
      next = next.slice(0, config.max_pdfs);
    }
    setFiles(next);
  };

  return (
    <section className="card upload-card">
      <h2>1 · Upload PKD papers</h2>
      <p className="muted">
        Select up to <b>{config.max_pdfs}</b> PDF papers. The pipeline will chunk
        and index them, draft questions across <b>{config.categories.length}</b>{" "}
        categories and <b>{config.question_types.length}</b> question types, find
        supporting evidence, and assemble cited gold answers for you to review.
      </p>

      <label
        className={"dropzone" + (dragover ? " dragover" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragover(true);
        }}
        onDragLeave={() => setDragover(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragover(false);
          accept(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(e) => accept(e.target.files)}
        />
        <span className="dropzone-icon">📄</span>
        <span className="dropzone-text">Click or drop PDFs here</span>
        <span className="file-list muted">
          {files.length ? files.map((f) => f.name).join(", ") : "No files selected"}
        </span>
      </label>

      <div className="taxonomy">
        <div>
          <h3>Categories</h3>
          <ul>
            {config.categories.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Question types</h3>
          <ul>
            {config.question_types.map((t) => (
              <li key={t}>
                {t} <span className="muted">→ {config.expected_behavior_by_type[t]}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <button
        className="btn btn-primary btn-lg"
        disabled={files.length === 0}
        onClick={() => onRun(files)}
      >
        Run pipeline (Stages 0–5)
      </button>
    </section>
  );
}
