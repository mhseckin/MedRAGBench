import { useState } from "react";
import { fmtScore } from "../api.js";

// Stage 6 detail: editable question + gold answer, flags, passages, actions.
// Mounted with key={item.id} in the parent, so local edit state resets cleanly
// whenever a different item is selected.
export default function ItemDetail({
  item,
  isFirst,
  isLast,
  onPrev,
  onNext,
  onSave,
  onDecide,
}) {
  const [question, setQuestion] = useState(item.question);
  const [goldAnswer, setGoldAnswer] = useState(item.gold_answer);

  const edits = () => ({ question, gold_answer: goldAnswer });

  return (
    <article className="detail">
      <div className="detail-header">
        <div className="badges">
          <span className="badge type">{item.type}</span>
          <span className="badge">{item.category}</span>
          <span className="badge">{item.difficulty}</span>
          <span className="badge">expects: {item.expected_behavior}</span>
        </div>
        <div className="flags">
          {item.flags.map((f) => (
            <span className="flag" key={f} title="quality-filter flag">
              ⚑ {f}
            </span>
          ))}
        </div>
      </div>

      <label className="field-label">
        Question <span className="muted">(editable)</span>
      </label>
      <textarea
        className="textarea"
        rows={3}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <label className="field-label">
        Gold answer{" "}
        <span className="muted">(editable — inline [P#] cite passages below)</span>
      </label>
      <textarea
        className="textarea"
        rows={9}
        value={goldAnswer}
        onChange={(e) => setGoldAnswer(e.target.value)}
      />

      <h4 className="passages-title">Supporting passages</h4>
      <div className="passages">
        {item.supporting_passages.length ? (
          item.supporting_passages.map((p) => (
            <div className="passage" key={p.label + p.chunk_id}>
              <div className="passage-head">
                <span className="passage-label">{p.label}</span>
                <span className="passage-paper">{p.paper_title || p.paper_id}</span>
                <span className="passage-score">score {fmtScore(p.rerank_score)}</span>
              </div>
              <div className="passage-text">{p.text}</div>
            </div>
          ))
        ) : (
          <div className="no-passages">
            No supporting passages — unanswerable / abstain item.
          </div>
        )}
      </div>

      <div className="detail-actions">
        <button className="btn btn-ghost" disabled={isFirst} onClick={onPrev}>
          ◀ Prev
        </button>
        <button className="btn" onClick={() => onSave(item, edits())}>
          Save edits
        </button>
        <button
          className="btn btn-danger"
          onClick={() => onDecide(item, false, edits())}
        >
          Reject ✗
        </button>
        <button
          className="btn btn-success"
          onClick={() => onDecide(item, true, edits())}
        >
          Approve ✓
        </button>
        <button className="btn btn-ghost spacer" disabled={isLast} onClick={onNext}>
          Next ▶
        </button>
      </div>
    </article>
  );
}
