import type { DocumentPageRead, NavigationTelemetry } from "../types";

interface TelemetryPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  telemetry: NavigationTelemetry | null;
  generatorLatencyMs?: number | null;
  notepadPageIds: string[];
  documents: DocumentPageRead[];
  onInspectPage: (page: DocumentPageRead) => void;
}

const STAGE_LABELS: Record<string, string> = {
  L0_categories: "Topics",
  L0_clusters: "Topics",
  L1_pages: "Pages",
  L1_pages_rescore: "Page Rescore",
  L1_5_headings: "Sections",
  S2_passages: "Passages",
  S3_mc: "Ranking",
};

export function TelemetryPanel({
  collapsed,
  onToggleCollapse,
  telemetry,
  generatorLatencyMs,
  notepadPageIds,
  documents,
  onInspectPage,
}: TelemetryPanelProps) {
  const docMap = new Map(documents.map((d) => [d.id, d]));

  if (collapsed) {
    const summary = telemetry
      ? `Telemetry · ${telemetry.total_latency_ms.toFixed(0)}ms`
      : "Telemetry";
    return (
      <aside
        className="telemetry-panel sidebar-rail"
        onClick={onToggleCollapse}
        role="button"
        tabIndex={0}
        aria-label="Expand telemetry panel"
        title="Expand telemetry panel"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggleCollapse();
          }
        }}
      >
        <span className="rail-toggle-glyph" aria-hidden="true">
          &laquo;
        </span>
        <span className="rail-label">{summary}</span>
      </aside>
    );
  }

  return (
    <aside className="telemetry-panel">
      <div className="panel-collapse-bar">
        <span className="eyebrow">Telemetry</span>
        <button
          type="button"
          className="btn-ghost btn-xs"
          aria-label="Collapse telemetry panel"
          title="Collapse telemetry panel"
          onClick={onToggleCollapse}
        >
          &raquo;
        </button>
      </div>
      <div className="panel-block">
        <span className="eyebrow">Stages</span>
        {!telemetry ? (
          <p className="empty-hint">Idle.</p>
        ) : (
          <>
            <div className="kpi-grid">
              <div className="kpi-box">
                <span className="kpi-label">Judge</span>
                <strong className="mono-xs">{telemetry.judge}</strong>
              </div>
              <div className="kpi-box">
                <span className="kpi-label">Nav</span>
                <strong>{telemetry.total_latency_ms.toFixed(0)} ms</strong>
              </div>
              <div className="kpi-box">
                <span className="kpi-label">Reqs</span>
                <strong>{telemetry.total_requests}</strong>
              </div>
              <div className="kpi-box">
                <span className="kpi-label">Mode</span>
                <strong>
                  {telemetry.profile_used}/{telemetry.mc_mode}
                </strong>
              </div>
            </div>

            {typeof generatorLatencyMs === "number" && (
              <div className="gen-latency-pill">
                Gen: <strong>{generatorLatencyMs.toFixed(0)} ms</strong>
              </div>
            )}

            <div className="stage-funnel-list">
              {telemetry.stages.map((st) => {
                const ratio =
                  st.candidates_in > 0
                    ? Math.max(6, Math.round((st.candidates_out / st.candidates_in) * 100))
                    : 100;
                return (
                  <div key={st.stage} className="stage-card">
                    <div className="stage-card-top">
                      <strong>{STAGE_LABELS[st.stage] || st.stage}</strong>
                      <span className="mono-xs">{st.latency_ms.toFixed(0)}ms</span>
                    </div>
                    <div className="funnel-bar-track">
                      <div
                        className="funnel-bar-fill"
                        style={{ width: `${Math.min(100, ratio)}%` }}
                      />
                    </div>
                    <div className="stage-card-meta">
                      <span>
                        {st.candidates_in} &rarr; <strong>{st.candidates_out}</strong>
                      </span>
                      <span>{st.requests_made}r</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>

      <div className="panel-block">
        <span className="eyebrow">Notepad</span>
        {notepadPageIds.length === 0 ? (
          <p className="empty-hint">Empty.</p>
        ) : (
          <div className="notepad-list">
            {notepadPageIds.map((pid, idx) => {
              const doc = docMap.get(pid);
              return (
                <div
                  key={pid}
                  className="notepad-item"
                  onClick={() => doc && onInspectPage(doc)}
                >
                  <span className="badge-rank">#{idx + 1}</span>
                  <div className="notepad-item-info">
                    <strong>{doc ? doc.title : pid.slice(0, 8)}</strong>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
