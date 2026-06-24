import ItemList from "./ItemList.jsx";
import ItemDetail from "./ItemDetail.jsx";

// Stage 6: clinician review — item list + detail editor.
export default function ReviewView({ items, index, onSelect, onSave, onDecide }) {
  const current = items[index];

  return (
    <div className="review-layout">
      <aside className="item-list-pane">
        <h3>
          Items <span className="muted">({items.length})</span>
        </h3>
        <ItemList items={items} index={index} onSelect={onSelect} />
      </aside>

      <div className="item-detail-pane">
        {current ? (
          <ItemDetail
            key={current.id}
            item={current}
            isFirst={index === 0}
            isLast={index === items.length - 1}
            onPrev={() => onSelect(Math.max(0, index - 1))}
            onNext={() => onSelect(Math.min(items.length - 1, index + 1))}
            onSave={onSave}
            onDecide={onDecide}
          />
        ) : (
          <div className="muted detail-empty">No items to review.</div>
        )}
      </div>
    </div>
  );
}
