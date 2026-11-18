import { type CSSProperties, useRef, useState } from "react";
import {
  ARCH_ASPECT_RATIO,
  clampArchHeight,
  loadLayoutPrefs,
  updateLayoutPrefs,
} from "../layoutPrefs";
import type { JudgeId, PipelineStepId } from "../types";

interface PipelineBarProps {
  activeStep: PipelineStepId | null;
  judge: JudgeId;
  hasNotepad: boolean;
}

const JUDGE_SHORT: Record<JudgeId, string> = {
  clef: "Clef-Flash 9B",
  openrouter_jev: "Jev 1.13",
  openrouter_luna: "GPT-6 Luna (OR)",
  openai_luna: "GPT-6 Luna",
};

const DECISION_STEPS: Set<PipelineStepId> = new Set([
  "categories",
  "clusters",
  "pages",
  "rescore",
  "sections",
  "passages",
  "ranking",
]);

interface ArchBoxProps {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sublabel?: string;
  active: boolean;
  dashed?: boolean;
}

function ArchBox({ x, y, w, h, label, sublabel, active, dashed }: ArchBoxProps) {
  const cx = x + w / 2;
  const cy = y + h / 2;
  return (
    <g className={`arch-node ${active ? "arch-node-active" : ""}`}>
      {active && (
        <rect
          x={x + 3}
          y={y + 3}
          width={w}
          height={h}
          className="arch-shadow"
        />
      )}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={3}
        className={`arch-rect ${dashed ? "arch-rect-dashed" : ""}`}
      />
      {active && (
        <rect
          x={x + 7}
          y={cy - 3}
          width={6}
          height={6}
          className="arch-active-pixel"
        />
      )}
      <text
        x={active ? cx + 4 : cx}
        y={sublabel ? cy - 3 : cy + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        className="arch-label"
      >
        {label}
      </text>
      {sublabel && (
        <text
          x={active ? cx + 4 : cx}
          y={cy + 9}
          textAnchor="middle"
          dominantBaseline="middle"
          className="arch-sublabel"
        >
          {sublabel}
        </text>
      )}
    </g>
  );
}

function ArrowHead({
  x,
  y,
  dir = "right",
}: {
  x: number;
  y: number;
  dir?: "right" | "down" | "up" | "left";
}) {
  if (dir === "down") {
    return <polygon points={`${x},${y} ${x - 4},${y - 6} ${x + 4},${y - 6}`} className="arch-arrow" />;
  }
  if (dir === "up") {
    return <polygon points={`${x},${y} ${x - 4},${y + 6} ${x + 4},${y + 6}`} className="arch-arrow" />;
  }
  if (dir === "left") {
    return <polygon points={`${x},${y} ${x + 6},${y - 4} ${x + 6},${y + 4}`} className="arch-arrow" />;
  }
  return <polygon points={`${x},${y} ${x - 6},${y - 4} ${x - 6},${y + 4}`} className="arch-arrow" />;
}

export function PipelineBar({ activeStep, judge, hasNotepad }: PipelineBarProps) {
  const [archHeight, setArchHeight] = useState<number | null>(
    () => loadLayoutPrefs().archHeight
  );
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragOriginRef = useRef<{
    startY: number;
    startHeight: number;
    moved: boolean;
  } | null>(null);

  const handleResetSize = () => {
    setArchHeight(null);
    updateLayoutPrefs({ archHeight: null });
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    const measuredHeight =
      archHeight ??
      Math.round(svgRef.current?.getBoundingClientRect().height || 180);
    dragOriginRef.current = {
      startY: e.clientY,
      startHeight: measuredHeight,
      moved: false,
    };
    setDragging(true);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const origin = dragOriginRef.current;
    if (!origin) return;
    const deltaY = e.clientY - origin.startY;
    if (!origin.moved && Math.abs(deltaY) < 2) return;
    origin.moved = true;
    const next = clampArchHeight(origin.startHeight + deltaY);
    if (next !== null) {
      setArchHeight(next);
    }
  };

  const finishDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    const origin = dragOriginRef.current;
    if (!origin) return;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    dragOriginRef.current = null;
    setDragging(false);
    if (origin.moved) {
      const deltaY = e.clientY - origin.startY;
      const finalHeight = clampArchHeight(origin.startHeight + deltaY);
      if (finalHeight !== null) {
        setArchHeight(finalHeight);
        updateLayoutPrefs({ archHeight: finalHeight });
      }
    }
  };

  const judgeActive = activeStep !== null && DECISION_STEPS.has(activeStep);
  const notepadActive =
    hasNotepad &&
    (activeStep === "categories" ||
      activeStep === "pages" ||
      activeStep === "passages");

  const customSvgStyle: CSSProperties | undefined =
    archHeight !== null
      ? {
          height: `${archHeight}px`,
          width: `${Math.round(archHeight * ARCH_ASPECT_RATIO)}px`,
          maxWidth: "none",
          maxHeight: "none",
          flexShrink: 0,
        }
      : undefined;

  return (
    <div
      className={`arch-shell ${dragging ? "arch-shell-dragging" : ""}`}
      aria-label="DecisionRAG Architecture"
    >
      {archHeight !== null && (
        <div className="arch-toolbar">
          <span className="mono-xs">{archHeight}px</span>
          <button
            type="button"
            className="btn-ghost btn-xs"
            onClick={handleResetSize}
          >
            Reset size
          </button>
        </div>
      )}

      <div className="arch-viewport">
        <svg
          ref={svgRef}
          className="arch-svg"
          style={customSvgStyle}
          viewBox="0 0 1040 226"
          shapeRendering="geometricPrecision"
        >
        {/* ===============================================================
            1. SESSION NOTEPAD FEEDBACK LOOP (Top Lane)
            =============================================================== */}
        {/* Ranked Passages (x=890, y=96) -> up to y=20 -> left into Notepad (x=596, y=20) */}
        <path
          d="M 890 96 L 890 20 L 598 20"
          className="arch-line arch-line-dashed"
        />
        <ArrowHead x={596} y={20} dir="left" />

        {/* Notepad (x=420, y=20) dashed feeds into Categories (x=174), Pages (x=316), Passages (x=605) */}
        <path
          d="M 420 20 L 174 20 L 174 50"
          className="arch-line arch-line-dashed"
        />
        <ArrowHead x={174} y={52} dir="down" />

        <path
          d="M 316 20 L 316 94"
          className="arch-line arch-line-dashed"
        />
        <ArrowHead x={316} y={96} dir="down" />

        <path
          d="M 555 34 L 555 42 L 605 42 L 605 94"
          className="arch-line arch-line-dashed"
        />
        <ArrowHead x={605} y={96} dir="down" />

        {/* ===============================================================
            2. MAIN HIERARCHICAL NAVIGATION GRAPH (Left -> Right)
            =============================================================== */}
        {/* Query (x=90, y=114) -> 3-way fork at x=104 (Categories / Flat / Clusters) */}
        <path d="M 90 114 L 258 114" className="arch-line" />
        <path d="M 104 114 L 104 69 L 118 69" className="arch-line" />
        <ArrowHead x={120} y={69} dir="right" />
        <path d="M 104 114 L 104 159 L 118 159" className="arch-line" />
        <ArrowHead x={120} y={159} dir="right" />

        {/* Categories & Clusters merge at x=244 -> Pages (x=260, y=114) */}
        <path d="M 228 69 L 244 69 L 244 114" className="arch-line" />
        <path d="M 228 159 L 244 159 L 244 114" className="arch-line" />
        <ArrowHead x={260} y={114} dir="right" />

        {/* Pages <-> Top-50 Rescore vertical loop */}
        <path d="M 300 132 L 300 146" className="arch-line arch-line-dashed" />
        <ArrowHead x={300} y={148} dir="down" />
        <path d="M 332 148 L 332 134" className="arch-line arch-line-dashed" />
        <ArrowHead x={332} y={132} dir="up" />

        {/* Pages (x=372) -> Section Filter fork (>60 passages) or direct -> Passages (x=544) */}
        <path d="M 372 114 L 542 114" className="arch-line" />
        <path d="M 388 114 L 388 69 L 402 69" className="arch-line" />
        <ArrowHead x={404} y={69} dir="right" />
        <path d="M 512 69 L 528 69 L 528 114" className="arch-line" />
        <ArrowHead x={544} y={114} dir="right" />

        {/* Passages (x=666) -> Choice Rerank fork or direct -> Ranked (x=848) */}
        <path d="M 666 114 L 846 114" className="arch-line" />
        <path d="M 682 114 L 682 69 L 696 69" className="arch-line" />
        <ArrowHead x={698} y={69} dir="right" />
        <path d="M 816 69 L 832 69 L 832 114" className="arch-line" />
        <ArrowHead x={848} y={114} dir="right" />

        {/* Ranked (x=932) -> Answer (x=952) */}
        <path d="M 932 114 L 950 114" className="arch-line" />
        <ArrowHead x={952} y={114} dir="right" />

        {/* ===============================================================
            3. DECISION MODEL BUS TICKS (Bottom Lane y=194)
            =============================================================== */}
        <path d="M 174 176 L 174 194" className="arch-line arch-line-dotted" />
        <path d="M 316 178 L 316 194" className="arch-line arch-line-dotted" />
        <path d="M 458 86 L 458 194" className="arch-line arch-line-dotted" />
        <path d="M 605 132 L 605 194" className="arch-line arch-line-dotted" />
        <path d="M 757 86 L 757 194" className="arch-line arch-line-dotted" />

        {/* ===============================================================
            4. ARCHITECTURE NODES (Off-white idle -> Solid Black active)
            =============================================================== */}
        <ArchBox
          x={420}
          y={6}
          w={176}
          h={28}
          label="Session Notepad"
          active={notepadActive}
          dashed
        />

        <ArchBox
          x={12}
          y={96}
          w={78}
          h={36}
          label="Query"
          active={activeStep === "query"}
        />

        <ArchBox
          x={120}
          y={52}
          w={108}
          h={34}
          label="Categories"
          active={activeStep === "categories"}
        />
        <ArchBox
          x={120}
          y={142}
          w={108}
          h={34}
          label="Clusters"
          active={activeStep === "clusters"}
        />

        <ArchBox
          x={260}
          y={96}
          w={112}
          h={36}
          label="Pages"
          active={activeStep === "pages"}
        />
        <ArchBox
          x={260}
          y={148}
          w={112}
          h={30}
          label="Top-50 Rescore"
          active={activeStep === "rescore"}
          dashed
        />

        <ArchBox
          x={404}
          y={52}
          w={108}
          h={34}
          label="Sections"
          sublabel=">60 passages"
          active={activeStep === "sections"}
        />

        <ArchBox
          x={544}
          y={96}
          w={122}
          h={36}
          label="Passages"
          active={activeStep === "passages"}
        />

        <ArchBox
          x={698}
          y={52}
          w={118}
          h={34}
          label="Choice Rerank"
          active={activeStep === "ranking"}
        />

        <ArchBox
          x={848}
          y={96}
          w={84}
          h={36}
          label="Ranked"
          active={activeStep === "ranked"}
        />

        <ArchBox
          x={952}
          y={96}
          w={78}
          h={36}
          label="Answer"
          active={activeStep === "answer"}
        />

        <ArchBox
          x={120}
          y={194}
          w={696}
          h={26}
          label={`Decision Model · ${JUDGE_SHORT[judge]}`}
          active={judgeActive}
        />
      </svg>
      </div>

      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label="Drag to resize architecture diagram (double-click to reset)"
        title="Drag to resize architecture diagram · Double-click to reset"
        className="arch-resize-handle"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={finishDrag}
        onPointerCancel={finishDrag}
        onDoubleClick={handleResetSize}
      >
        <span className="arch-resize-grip" aria-hidden="true" />
      </div>
    </div>
  );
}
