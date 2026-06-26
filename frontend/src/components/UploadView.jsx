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

  const totalQuestions = config.questions_per_type
    ? Object.values(config.questions_per_type).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <section className="card upload-card">
      <h2>1 · Upload PKD Papers</h2>
      <p className="muted">
        Select up to <b>{config.max_pdfs}</b> PDF papers. Each paper will be
        classified into a category, then <b>{totalQuestions}</b> general patient
        questions will be generated, with evidence retrieved across all papers.
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
          <h3>Paper Categories</h3>
          <ul>
            {config.categories.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Question Distribution</h3>
          <ul>
            {config.question_types.map((t) => (
              <li key={t}>
                {t}{" "}
                <span className="muted">
                  — {config.questions_per_type?.[t] || 0} questions
                </span>
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
        Run Pipeline
      </button>
    </section>
  );
}
