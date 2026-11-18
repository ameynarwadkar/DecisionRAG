export function PixelTreeGlyph({ active = false }: { active?: boolean }) {
  return (
    <svg
      className={`pixel-glyph ${active ? "pixel-glyph-active" : ""}`}
      width="18"
      height="18"
      viewBox="0 0 9 9"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect className="px-node-root" x="3" y="0" width="3" height="2" />
      <rect className="px-branch" x="4" y="2" width="1" height="2" />
      <rect className="px-branch" x="1" y="4" width="7" height="1" />
      <rect className="px-leaf-1" x="1" y="5" width="1" height="2" />
      <rect className="px-leaf-2" x="4" y="5" width="1" height="2" />
      <rect className="px-leaf-3" x="7" y="5" width="1" height="2" />
      <rect className="px-dot-1" x="0" y="7" width="3" height="2" />
      <rect className="px-dot-2" x="4" y="7" width="1" height="2" />
      <rect className="px-dot-3" x="6" y="7" width="3" height="2" />
    </svg>
  );
}

export function PixelMatrixSpinner({ label }: { label?: string }) {
  return (
    <span className="loader-bars-wrap" role="status">
      <span className="loader-bars" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      {label && <span>{label}</span>}
    </span>
  );
}

export function PixelScanBar() {
  return (
    <div className="top-progress-track" aria-hidden="true">
      <div className="top-progress-glider" />
    </div>
  );
}
