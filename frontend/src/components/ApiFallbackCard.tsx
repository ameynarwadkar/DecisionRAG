import { useEffect, useState } from "react";
import type { JudgeId, LocalClientSettings } from "../types";
import { PixelMatrixSpinner } from "./PixelArt";

interface ApiFallbackCardProps {
  needsJudge: boolean;
  judgeReason: string | null;
  needsGenerator: boolean;
  generatorReason: string | null;
  selectedJudge: JudgeId;
  settings: LocalClientSettings;
  verifying: boolean;
  onSelectJudge: (j: JudgeId) => void;
  onApplyAndVerify: (next: LocalClientSettings, judgeToVerify: JudgeId) => Promise<void>;
  onRecheckEnv: () => Promise<void>;
  onDismissGenerator: () => void;
}

export function ApiFallbackCard({
  needsJudge,
  judgeReason,
  needsGenerator,
  generatorReason,
  selectedJudge,
  settings,
  verifying,
  onSelectJudge,
  onApplyAndVerify,
  onRecheckEnv,
  onDismissGenerator,
}: ApiFallbackCardProps) {
  const [draft, setDraft] = useState<LocalClientSettings>(settings);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  if (!needsJudge && !needsGenerator) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onApplyAndVerify(draft, selectedJudge);
  };

  return (
    <div
      className={`api-fallback-card ${
        expanded ? "api-fallback-expanded" : "api-fallback-collapsed"
      }`}
    >
      <div className="api-fallback-header">
        <button
          type="button"
          className="api-fallback-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
        >
          <span className="mono-xs" aria-hidden="true">
            {expanded ? "▾" : "▸"}
          </span>
          <span className="eyebrow">
            API Input (.env missing or unreachable)
          </span>
          {!expanded && (
            <span className="api-fallback-pills">
              {needsJudge && (
                <span className="score-pill">
                  Judge: {judgeReason || "Needed"}
                </span>
              )}
              {needsGenerator && (
                <span className="score-pill">
                  Gen: {generatorReason || "Needed"}
                </span>
              )}
            </span>
          )}
        </button>

        <div className="api-fallback-actions">
          <button
            type="button"
            className="btn-ghost btn-xs"
            disabled={verifying}
            onClick={() => void onRecheckEnv()}
          >
            Re-read .env
          </button>
          <button
            type="button"
            className="btn-ghost btn-xs"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "Collapse" : "Configure"}
          </button>
        </div>
      </div>

      {expanded && (
        <form className="api-fallback-body" onSubmit={handleSubmit}>
          {needsJudge && (
        <div className="fallback-group">
          <div className="row-between">
            <strong>Decision Judge</strong>
            {judgeReason && <span className="mono-xs">{judgeReason}</span>}
          </div>

          <select
            value={selectedJudge}
            onChange={(e) => onSelectJudge(e.target.value as JudgeId)}
          >
            <option value="clef">Clef-Flash 9B (Colab)</option>
            <option value="openrouter_jev">OpenRouter · Jev 1.13</option>
            <option value="openrouter_luna">OpenRouter · GPT-6 Luna</option>
            <option value="openai_luna">OpenAI Direct · GPT-6 Luna</option>
          </select>

          {selectedJudge === "clef" && (
            <div className="grid-2">
              <input
                type="text"
                placeholder="Clef URL (https://...trycloudflare.com)"
                value={draft.clefApiUrl}
                onChange={(e) => setDraft({ ...draft, clefApiUrl: e.target.value })}
              />
              <input
                type="password"
                placeholder="Clef Key"
                value={draft.clefApiKey}
                onChange={(e) => setDraft({ ...draft, clefApiKey: e.target.value })}
              />
            </div>
          )}

          {(selectedJudge === "openrouter_jev" ||
            selectedJudge === "openrouter_luna") && (
            <input
              type="password"
              placeholder="OpenRouter API Key (sk-or-...)"
              value={draft.openrouterApiKey}
              onChange={(e) =>
                setDraft({ ...draft, openrouterApiKey: e.target.value })
              }
            />
          )}

          {selectedJudge === "openai_luna" && (
            <input
              type="password"
              placeholder="OpenAI API Key (sk-...)"
              value={draft.openaiApiKey}
              onChange={(e) =>
                setDraft({ ...draft, openaiApiKey: e.target.value })
              }
            />
          )}
        </div>
      )}

      {needsGenerator && (
        <div className="fallback-group">
          <div className="row-between">
            <strong>Generator LLM</strong>
            <div className="row-between">
              {generatorReason && <span className="mono-xs">{generatorReason}</span>}
              <button
                type="button"
                className="btn-ghost btn-xs"
                onClick={onDismissGenerator}
              >
                Skip
              </button>
            </div>
          </div>

          <div className="grid-2">
            <input
              type="text"
              placeholder="Model (azure/gpt-4o or openai/gpt-4o-mini)"
              value={draft.generatorModel}
              onChange={(e) =>
                setDraft({ ...draft, generatorModel: e.target.value })
              }
            />
            <input
              type="password"
              placeholder="Generator / Azure API Key"
              value={draft.generatorApiKey}
              onChange={(e) =>
                setDraft({ ...draft, generatorApiKey: e.target.value })
              }
            />
          </div>
          <div className="grid-2">
            <input
              type="text"
              placeholder="Azure Endpoint (optional)"
              value={draft.generatorApiBase}
              onChange={(e) =>
                setDraft({ ...draft, generatorApiBase: e.target.value })
              }
            />
            <input
              type="text"
              placeholder="API Version (2024-10-21)"
              value={draft.generatorApiVersion}
              onChange={(e) =>
                setDraft({ ...draft, generatorApiVersion: e.target.value })
              }
            />
          </div>
        </div>
      )}

      <div className="row-between">
        <span className="muted-xs">Used automatically if .env is not set</span>
        <button type="submit" className="btn-primary btn-sm" disabled={verifying}>
          {verifying ? <PixelMatrixSpinner label="Verifying" /> : "Verify & Use"}
        </button>
      </div>
        </form>
      )}
    </div>
  );
}
