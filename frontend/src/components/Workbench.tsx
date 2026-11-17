import { useRef, useState } from "react";
import type {
  ChatMessageEntry,
  JudgeComparisonEntry,
  JudgeId,
  LocalClientSettings,
  PipelineStepId,
  RankedPassageItem,
  WorkbenchMode,
} from "../types";
import { ApiFallbackCard } from "./ApiFallbackCard";
import { PipelineBar } from "./PipelineBar";
import { PixelMatrixSpinner, PixelScanBar } from "./PixelArt";

interface WorkbenchProps {
  mode: WorkbenchMode;
  onChangeMode: (mode: WorkbenchMode) => void;
  activePipelineStep: PipelineStepId | null;
  activeJudge: JudgeId;
  hasNotepad: boolean;
  hasNamespace: boolean;
  hasDocuments: boolean;
  busyUpload: boolean;
  busyRun: boolean;
  errorBanner: string | null;
  onOpenSettings: () => void;
  onQuickUpload: (files: FileList) => Promise<void>;
  localSettings: LocalClientSettings;
  verifyingApis: boolean;
  needsJudgeInput: boolean;
  judgeIssue: string | null;
  needsGeneratorInput: boolean;
  generatorIssue: string | null;
  onSelectJudge: (j: JudgeId) => void;
  onApplyFallbackApis: (
    next: LocalClientSettings,
    judgeToVerify: JudgeId
  ) => Promise<void>;
  onRecheckEnv: () => Promise<void>;
  onDismissGenerator: () => void;
  lastAnswer: string | null;
  rankedPassages: RankedPassageItem[];
  chatMessages: ChatMessageEntry[];
  chatSessionId: string | null;
  onResetChat: () => void;
  compareEntries: JudgeComparisonEntry[];
  selectedCompareJudges: JudgeId[];
  onToggleCompareJudge: (j: JudgeId) => void;
  onRunQueryOrRetrieve: (query: string, scenario: string) => Promise<void>;
  onSendChat: (message: string) => Promise<void>;
  onRunCompare: (query: string, scenario: string) => Promise<void>;
}

const COMPARE_JUDGE_OPTIONS: JudgeId[] = [
  "clef",
  "openrouter_jev",
  "openrouter_luna",
  "openai_luna",
];

function renderCitedText(
  text: string,
  onSelectRank: (rank: number) => void
): React.ReactNode[] {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((chunk, idx) => {
    const match = /^\[(\d+)\]$/.exec(chunk);
    if (match) {
      const rankNum = parseInt(match[1], 10);
      return (
        <button
          key={idx}
          type="button"
          className="citation-chip"
          onClick={() => onSelectRank(rankNum)}
        >
          [{rankNum}]
        </button>
      );
    }
    return <span key={idx}>{chunk}</span>;
  });
}

function PassageScoreCard({
  item,
  highlighted,
}: {
  item: RankedPassageItem;
  highlighted: boolean;
}) {
  return (
    <div
      id={`passage-rank-${item.rank}`}
      className={`passage-card ${highlighted ? "passage-card-highlight" : ""}`}
    >
      <div className="passage-card-header">
        <div className="passage-title-group">
          <span className="badge-rank">#{item.rank}</span>
          <strong>{item.page_title}</strong>
          {item.heading_path && (
            <span className="badge-section">{item.heading_path}</span>
          )}
        </div>
        <div className="score-pills">
          <span className="score-pill">
            {(item.p_para_bin * 100).toFixed(1)}%
          </span>
          {item.p_mul !== null && (
            <span className="score-pill score-pill-mc">
              {(item.p_mul * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      <div className="prob-bar-track">
        <div
          className="prob-bar-fill"
          style={{ width: `${Math.round(item.stage2_score * 100)}%` }}
        />
      </div>

      <p className="passage-text">{item.content}</p>
    </div>
  );
}

export function Workbench({
  mode,
  onChangeMode,
  activePipelineStep,
  activeJudge,
  hasNotepad,
  hasNamespace,
  hasDocuments,
  busyUpload,
  busyRun,
  errorBanner,
  onOpenSettings,
  onQuickUpload,
  localSettings,
  verifyingApis,
  needsJudgeInput,
  judgeIssue,
  needsGeneratorInput,
  generatorIssue,
  onSelectJudge,
  onApplyFallbackApis,
  onRecheckEnv,
  onDismissGenerator,
  lastAnswer,
  rankedPassages,
  chatMessages,
  chatSessionId,
  onResetChat,
  compareEntries,
  selectedCompareJudges,
  onToggleCompareJudge,
  onRunQueryOrRetrieve,
  onSendChat,
  onRunCompare,
}: WorkbenchProps) {
  const [queryText, setQueryText] = useState("");
  const [scenarioText, setScenarioText] = useState("");
  const [showScenario, setShowScenario] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [highlightedRank, setHighlightedRank] = useState<number | null>(null);
  const centerFileRef = useRef<HTMLInputElement | null>(null);

  const jumpToRank = (rank: number) => {
    setHighlightedRank(rank);
    const el = document.getElementById(`passage-rank-${rank}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  const handleMainSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryText.trim() || busyRun) return;
    setHighlightedRank(null);
    if (mode === "compare") {
      await onRunCompare(queryText, scenarioText);
    } else {
      await onRunQueryOrRetrieve(queryText, scenarioText);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || busyRun) return;
    const msg = chatInput.trim();
    setChatInput("");
    await onSendChat(msg);
  };

  return (
    <main className="workbench">
      <PipelineBar
        activeStep={activePipelineStep}
        judge={activeJudge}
        hasNotepad={hasNotepad}
      />

      <div className="mode-tabs">
        <button
          type="button"
          className={`mode-tab ${mode === "query" ? "active" : ""}`}
          onClick={() => onChangeMode("query")}
        >
          Query
        </button>
        <button
          type="button"
          className={`mode-tab ${mode === "retrieve" ? "active" : ""}`}
          onClick={() => onChangeMode("retrieve")}
        >
          Retrieve
        </button>
        <button
          type="button"
          className={`mode-tab ${mode === "chat" ? "active" : ""}`}
          onClick={() => onChangeMode("chat")}
        >
          Chat
        </button>
        <button
          type="button"
          className={`mode-tab ${mode === "compare" ? "active" : ""}`}
          onClick={() => onChangeMode("compare")}
        >
          Compare
        </button>
      </div>

      {errorBanner && (
        <div className="error-banner">
          <span>{errorBanner}</span>
          <button
            type="button"
            className="btn-secondary btn-xs"
            onClick={onOpenSettings}
          >
            Keys
          </button>
        </div>
      )}

      <ApiFallbackCard
        needsJudge={needsJudgeInput}
        judgeReason={judgeIssue}
        needsGenerator={needsGeneratorInput}
        generatorReason={generatorIssue}
        selectedJudge={activeJudge}
        settings={localSettings}
        verifying={verifyingApis}
        onSelectJudge={onSelectJudge}
        onApplyAndVerify={onApplyFallbackApis}
        onRecheckEnv={onRecheckEnv}
        onDismissGenerator={onDismissGenerator}
      />

      {!hasNamespace || !hasDocuments ? (
        <div
          className="onboarding-card"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
              void onQuickUpload(e.dataTransfer.files);
            }
          }}
        >
          <input
            ref={centerFileRef}
            type="file"
            multiple
            accept=".pdf,.md,.markdown,.html,.htm,.txt"
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                void onQuickUpload(e.target.files);
                e.target.value = "";
              }
            }}
          />
          <button
            type="button"
            className="dropzone-btn"
            disabled={busyUpload}
            onClick={() => centerFileRef.current?.click()}
          >
            {busyUpload ? (
              <PixelMatrixSpinner label="Ingesting" />
            ) : (
              "+ Upload Documents (.pdf, .md, .html, .txt)"
            )}
          </button>
        </div>
      ) : mode === "chat" ? (
        <div className="chat-container">
          <div className="chat-toolbar">
            <span className="mono-xs">
              {chatSessionId ? chatSessionId.slice(0, 8) : "new"}
            </span>
            <button type="button" className="btn-ghost btn-xs" onClick={onResetChat}>
              Reset
            </button>
          </div>

          <div className="chat-messages">
            {chatMessages.length === 0 ? (
              <p className="empty-hint">No messages.</p>
            ) : (
              chatMessages.map((m, idx) => (
                <div key={idx} className={`chat-bubble chat-${m.role}`}>
                  <div className="chat-role-label">
                    {m.role === "user" ? "User" : "Assistant"}
                  </div>
                  <div className="chat-content">
                    {m.role === "assistant"
                      ? renderCitedText(m.content, jumpToRank)
                      : m.content}
                  </div>
                </div>
              ))
            )}
          </div>

          <form className="chat-input-bar" onSubmit={handleChatSubmit}>
            <input
              type="text"
              placeholder={hasDocuments ? "Message…" : "Upload files first"}
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              disabled={!hasDocuments || busyRun}
            />
            <button
              type="submit"
              className="btn-primary"
              disabled={!hasDocuments || busyRun || !chatInput.trim()}
            >
              {busyRun ? <PixelMatrixSpinner /> : "Send"}
            </button>
          </form>

          {rankedPassages.length > 0 && (
            <div className="results-section">
              <span className="eyebrow">Passages ({rankedPassages.length})</span>
              <div className="passage-list">
                {rankedPassages.map((item) => (
                  <PassageScoreCard
                    key={item.passage_id}
                    item={item}
                    highlighted={highlightedRank === item.rank}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="query-container">
          <form className="query-form-card" onSubmit={handleMainSubmit}>
            {mode === "compare" && (
              <div className="compare-judge-picker">
                {COMPARE_JUDGE_OPTIONS.map((jid) => (
                  <label key={jid} className="chip-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedCompareJudges.includes(jid)}
                      onChange={() => onToggleCompareJudge(jid)}
                    />
                    <span>{jid}</span>
                  </label>
                ))}
              </div>
            )}

            <textarea
              rows={2}
              placeholder={hasDocuments ? "Question…" : "Upload files first"}
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              disabled={!hasDocuments || busyRun}
            />

            {showScenario && (
              <input
                type="text"
                placeholder="Scenario…"
                value={scenarioText}
                onChange={(e) => setScenarioText(e.target.value)}
              />
            )}

            <div className="row-between">
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={() => setShowScenario((s) => !s)}
              >
                {showScenario ? "− Scenario" : "+ Scenario"}
              </button>

              <button
                type="submit"
                className="btn-primary"
                disabled={
                  !hasDocuments ||
                  busyRun ||
                  !queryText.trim() ||
                  (mode === "compare" && selectedCompareJudges.length === 0)
                }
              >
                {busyRun ? <PixelMatrixSpinner label="Navigating" /> : "Run"}
              </button>
            </div>
            {busyRun && <PixelScanBar />}
          </form>

          {mode === "compare" ? (
            compareEntries.length > 0 && (
              <div className="compare-grid">
                {compareEntries.map((entry) => (
                  <div key={entry.judge} className="compare-col">
                    <div className="compare-col-header">
                      <strong>{entry.judge}</strong>
                      <span className="mono-xs">
                        {entry.telemetry.total_latency_ms.toFixed(0)}ms ·{" "}
                        {entry.telemetry.total_requests}r
                      </span>
                    </div>
                    <div className="passage-list">
                      {entry.passages.slice(0, 5).map((p) => (
                        <PassageScoreCard
                          key={p.passage_id}
                          item={p}
                          highlighted={false}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )
          ) : (
            <>
              {mode === "query" && lastAnswer && (
                <div className="answer-card">
                  <span className="eyebrow">Answer</span>
                  <div className="answer-body">
                    {renderCitedText(lastAnswer, jumpToRank)}
                  </div>
                </div>
              )}

              {rankedPassages.length > 0 && (
                <div className="results-section">
                  <span className="eyebrow">
                    Passages ({rankedPassages.length})
                  </span>
                  <div className="passage-list">
                    {rankedPassages.map((item) => (
                      <PassageScoreCard
                        key={item.passage_id}
                        item={item}
                        highlighted={highlightedRank === item.rank}
                      />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </main>
  );
}
