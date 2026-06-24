import { shorten } from "../api.js";

// Left rail: every generated item with a status dot and flag count.
export default function ItemList({ items, index, onSelect }) {
  return (
    <ul className="item-list">
      {items.map((it, i) => {
        const status = it.decision || ""; // "approved" | "rejected" | ""
        return (
          <li
            key={it.id}
            className={i === index ? "selected" : ""}
            onClick={() => onSelect(i)}
          >
            <span className={"dot " + status} />
            <span className="li-text">
              {i + 1}. {shorten(it.question, 42)}
            </span>
            {it.flags.length > 0 && (
              <span className="li-flag" title={it.flags.join(", ")}>
                ⚑{it.flags.length}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
