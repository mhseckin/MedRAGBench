import { useEffect, useRef } from "react";

// Stages 0-5: live progress log streamed from the backend job.
export default function RunningView({ logLines, status }) {
  const logRef = useRef(null);

  // Keep the log scrolled to the newest line.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  return (
    <section className="card">
      <h2>2 · Building benchmark…</h2>
      <p className="muted">
        Embedding, retrieval, and generation call the LLM and the cross-encoder,
        so this takes a few minutes. Live log below.
      </p>
      <div className="spinner-row">
        <span className="spinner" />
        <span>{status}</span>
      </div>
      <pre className="log" ref={logRef}>
        {logLines.join("\n")}
      </pre>
    </section>
  );
}
