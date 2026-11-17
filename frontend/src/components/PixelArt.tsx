/**
 * 1-bit pixel art & stepped frame-by-frame animations (strictly var(--fg) & var(--bg-base)).
 */

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
      {/* Root node */}
      <rect className="px-node-root" x="3" y="0" width="3" height="2" />
      {/* Stem */}
      <rect className="px-branch" x="4" y="2" width="1" height="2" />
      <rect className="px-branch" x="1" y="4" width="7" height="1" />
      {/* Left & Right mid branches */}
      <rect className="px-leaf-1" x="1" y="5" width="1" height="2" />
      <rect className="px-leaf-2" x="4" y="5" width="1" height="2" />
      <rect className="px-leaf-3" x="7" y="5" width="1" height="2" />
      {/* Leaf pixels */}
      <rect className="px-dot-1" x="0" y="7" width="3" height="2" />
      <rect className="px-dot-2" x="4" y="7" width="1" height="2" />
      <rect className="px-dot-3" x="6" y="7" width="3" height="2" />
    </svg>
  );
}

export function PixelMatrixSpinner({ label }: { label?: string }) {
  return (
    <span className="pixel-spinner-wrap" role="status">
      <span className="pixel-matrix" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
        <i />
        <i />
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
    <div className="pixel-scanbar" aria-hidden="true">
      <div className="pixel-scanbar-runner" />
    </div>
  );
}
