import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import UploadView from "./components/UploadView.jsx";
import RunningView from "./components/RunningView.jsx";
import ReviewView from "./components/ReviewView.jsx";

export default function App() {
  const [config, setConfig] = useState(null);
  const [view, setView] = useState("upload"); // upload | running | review
  const [jobId, setJobId] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [runStatus, setRunStatus] = useState("Starting…");
  const [items, setItems] = useState([]);
  const [index, setIndex] = useState(0);
  const [toast, setToast] = useState(null); // { msg, error }

  const showToast = useCallback((msg, error = false) => {
    setToast({ msg, error });
  }, []);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3200);
    return () => clearTimeout(t);
  }, [toast]);

  // ---- Stage 0: load taxonomy/config -------------------------------------
  useEffect(() => {
    api("/api/config")
      .then(setConfig)
      .catch((e) => showToast("Cannot reach backend: " + e.message, true));
  }, [showToast]);

  // ---- Stages 0-5: launch generation -------------------------------------
  const startJob = useCallback(
    async (files) => {
      const form = new FormData();
      files.forEach((f) => form.append("files", f, f.name));
      setLogLines([]);
      setRunStatus("Uploading…");
      setView("running");
      try {
        const { job_id } = await api("/api/jobs", { method: "POST", body: form });
        setJobId(job_id);
        setRunStatus("Running pipeline…");
      } catch (e) {
        showToast("Failed to start: " + e.message, true);
        setView("upload");
      }
    },
    [showToast]
  );

  // ---- Poll progress while running ---------------------------------------
  const cursorRef = useRef(0);
  useEffect(() => {
    if (view !== "running" || !jobId) return;
    cursorRef.current = 0;
    let cancelled = false;

    const id = setInterval(async () => {
      let job;
      try {
        job = await api(`/api/jobs/${jobId}?since=${cursorRef.current}`);
      } catch {
        return; // transient — retry next tick
      }
      if (cancelled) return;
      if (job.log.length) {
        cursorRef.current = job.log_cursor;
        setLogLines((prev) => [...prev, ...job.log]);
      }
      if (job.status === "done") {
        clearInterval(id);
        setRunStatus("Done.");
        try {
          const list = await api(`/api/jobs/${jobId}/items`);
          setItems(list.map((it) => ({ ...it, decision: it.approved ? "approved" : null })));
          setIndex(0);
          setView("review");
        } catch (e) {
          showToast("Failed to load items: " + e.message, true);
        }
      } else if (job.status === "error") {
        clearInterval(id);
        setRunStatus("Failed.");
        showToast("Pipeline error: " + (job.error || "see log"), true);
      }
    }, 800);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [view, jobId, showToast]);

  // ---- Stage 6: review actions -------------------------------------------
  const patchItem = useCallback(
    async (item, body) => {
      const updated = await api(`/api/jobs/${jobId}/items/${item.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setItems((prev) =>
        prev.map((it) => (it.id === item.id ? { ...it, ...updated } : it))
      );
      return updated;
    },
    [jobId]
  );

  const saveEdits = useCallback(
    async (item, edits) => {
      await patchItem(item, edits);
      showToast("Edits saved.");
    },
    [patchItem, showToast]
  );

  const decide = useCallback(
    async (item, approve, edits) => {
      try {
        await patchItem(item, edits); // persist edits before deciding
        const verb = approve ? "approve" : "reject";
        const updated = await api(`/api/jobs/${jobId}/items/${item.id}/${verb}`, {
          method: "POST",
        });
        setItems((prev) =>
          prev.map((it) =>
            it.id === item.id
              ? { ...it, ...updated, decision: approve ? "approved" : "rejected" }
              : it
          )
        );
        showToast(approve ? "Approved." : "Rejected.");
        setIndex((i) => (i < items.length - 1 ? i + 1 : i));
      } catch (e) {
        showToast("Action failed: " + e.message, true);
      }
    },
    [patchItem, jobId, items.length, showToast]
  );

  // ---- Stage 7: export ----------------------------------------------------
  const exportJson = useCallback(async () => {
    try {
      const data = await api(`/api/jobs/${jobId}/export`);
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "benchmark.json";
      a.click();
      URL.revokeObjectURL(url);
      showToast(`Exported ${data.count} approved record(s).`);
    } catch (e) {
      showToast("Export failed: " + e.message, true);
    }
  }, [jobId, showToast]);

  const approvedCount = items.filter((it) => it.approved).length;

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="logo">⬡</span>
          <div>
            <h1>MedRAGBench</h1>
            <p className="subtitle">Medical RAG benchmark builder</p>
          </div>
        </div>
        <div className="topbar-right">
          {config && <span className="chip">Provider: {config.provider}</span>}
          {view === "review" && (
            <div className="review-summary">
              <span className="approved-count">{approvedCount}</span> approved of{" "}
              {items.length}
              <button className="btn btn-primary" onClick={exportJson}>
                Export JSON ↓
              </button>
              <button className="btn btn-ghost" onClick={() => window.location.reload()}>
                New run
              </button>
            </div>
          )}
        </div>
      </header>

      <main>
        {view === "upload" && (
          <UploadView config={config} onRun={startJob} onToast={showToast} />
        )}
        {view === "running" && (
          <RunningView logLines={logLines} status={runStatus} />
        )}
        {view === "review" && (
          <ReviewView
            items={items}
            index={index}
            onSelect={setIndex}
            onSave={saveEdits}
            onDecide={decide}
          />
        )}
      </main>

      {toast && (
        <div className={"toast" + (toast.error ? " error" : "")}>{toast.msg}</div>
      )}
    </>
  );
}
