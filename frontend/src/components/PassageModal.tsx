import type { DocumentPageRead, PassageRead } from "../types";

interface PassageModalProps {
  page: DocumentPageRead | null;
  passages: PassageRead[];
  loading: boolean;
  onClose: () => void;
}

export function PassageModal({
  page,
  passages,
  loading,
  onClose,
}: PassageModalProps) {
  if (!page) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Passages"
      >
        <div className="drawer-header">
          <div>
            <h2>{page.title}</h2>
            <p className="muted-sm">
              <code>{page.filename}</code> · {page.passage_count} ¶
            </p>
          </div>
          <button className="btn-ghost" onClick={onClose} type="button">
            ×
          </button>
        </div>

        <div className="modal-body">
          {loading ? (
            <p className="muted-sm">Loading…</p>
          ) : passages.length === 0 ? (
            <p className="muted-sm">Empty.</p>
          ) : (
            <div className="passage-inspect-list">
              {passages.map((p) => (
                <div key={p.id} className="passage-inspect-item">
                  <div className="passage-inspect-top">
                    <span className="badge-rank">#{p.passage_index}</span>
                    {p.heading_path && (
                      <span className="badge-section">{p.heading_path}</span>
                    )}
                    <span className="mono-xs">{p.token_estimate}t</span>
                  </div>
                  <p className="passage-text">{p.content}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
