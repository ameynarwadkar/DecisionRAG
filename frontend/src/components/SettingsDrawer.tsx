import { useState } from "react";
import { api } from "../api";
import type {
  JudgeId,
  JudgeStatusInfo,
  JudgeTestResponse,
  LocalClientSettings,
} from "../types";

interface SettingsDrawerProps {
  open: boolean;
  onClose: () => void;
  settings: LocalClientSettings;
  onSave: (next: LocalClientSettings) => void;
  judges: JudgeStatusInfo[];
  serverGeneratorModel: string;
  serverGeneratorConfigured: boolean;
}

const ALL_JUDGES: { id: JudgeId; label: string }[] = [
  { id: "clef", label: "Clef-Flash 9B" },
  { id: "openrouter_jev", label: "OpenRouter Jev" },
  { id: "openrouter_luna", label: "OpenRouter Luna" },
  { id: "openai_luna", label: "OpenAI Luna" },
];

export function SettingsDrawer({
  open,
  onClose,
  settings,
  onSave,
  judges,
  serverGeneratorModel,
  serverGeneratorConfigured,
}: SettingsDrawerProps) {
  const [draft, setDraft] = useState<LocalClientSettings>(settings);
  const [testingJudge, setTestingJudge] = useState<JudgeId | null>(null);
  const [testResults, setTestResults] = useState<
    Partial<Record<JudgeId, { ok: boolean; msg: string; detail?: JudgeTestResponse }>>
  >({});
  const [testingGen, setTestingGen] = useState(false);
  const [genTestMsg, setGenTestMsg] = useState<string | null>(null);

  if (!open) return null;

  const handleTest = async (judgeId: JudgeId) => {
    onSave(draft);
    setTestingJudge(judgeId);
    try {
      const res = await api.testJudge(judgeId, draft);
      setTestResults((prev) => ({
        ...prev,
        [judgeId]: {
          ok: true,
          msg: `OK · ${res.latency_ms.toFixed(0)}ms · ${(res.p_relevant * 100).toFixed(1)}% / ${(res.p_irrelevant * 100).toFixed(2)}%`,
          detail: res,
        },
      }));
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [judgeId]: {
          ok: false,
          msg: err instanceof Error ? err.message : String(err),
        },
      }));
    } finally {
      setTestingJudge(null);
    }
  };

  const handleTestGenerator = async () => {
    onSave(draft);
    setTestingGen(true);
    setGenTestMsg(null);
    try {
      const res = await api.testGenerator(draft);
      setGenTestMsg(
        `OK · ${res.latency_ms.toFixed(0)}ms · ${res.model} -> "${res.reply}"`
      );
    } catch (err) {
      setGenTestMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setTestingGen(false);
    }
  };

  const serverMap = new Map(judges.map((j) => [j.judge_id, j]));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="drawer-panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Settings"
      >
        <div className="drawer-header">
          <h2>Settings</h2>
          <button className="btn-ghost" onClick={onClose} type="button">
            ×
          </button>
        </div>

        <div className="drawer-body">
          <section className="settings-section">
            <h3>Auth</h3>
            <label className="field-label">
              <span>X-API-Key</span>
              <input
                type="password"
                placeholder="Optional"
                value={draft.apiKey}
                onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })}
              />
            </label>
          </section>

          <section className="settings-section">
            <h3>Clef</h3>
            <label className="field-label">
              <span>URL</span>
              <input
                type="text"
                placeholder={
                  serverMap.get("clef")?.configured ? ".env configured" : "https://..."
                }
                value={draft.clefApiUrl}
                onChange={(e) => setDraft({ ...draft, clefApiUrl: e.target.value })}
              />
            </label>
            <label className="field-label">
              <span>Key</span>
              <input
                type="password"
                placeholder={serverMap.get("clef")?.configured ? ".env configured" : "Key"}
                value={draft.clefApiKey}
                onChange={(e) => setDraft({ ...draft, clefApiKey: e.target.value })}
              />
            </label>
          </section>

          <section className="settings-section">
            <h3>OpenRouter</h3>
            <label className="field-label">
              <span>Key</span>
              <input
                type="password"
                placeholder={
                  serverMap.get("openrouter_jev")?.configured ? ".env configured" : "sk-or-..."
                }
                value={draft.openrouterApiKey}
                onChange={(e) =>
                  setDraft({ ...draft, openrouterApiKey: e.target.value })
                }
              />
            </label>
          </section>

          <section className="settings-section">
            <h3>OpenAI</h3>
            <label className="field-label">
              <span>Key</span>
              <input
                type="password"
                placeholder={
                  serverMap.get("openai_luna")?.configured ? ".env configured" : "sk-..."
                }
                value={draft.openaiApiKey}
                onChange={(e) => setDraft({ ...draft, openaiApiKey: e.target.value })}
              />
            </label>
          </section>

          <section className="settings-section">
            <div className="row-between">
              <h3>Generator (OpenAI / Azure)</h3>
              <button
                type="button"
                className="btn-secondary btn-xs"
                disabled={testingGen}
                onClick={() => void handleTestGenerator()}
              >
                {testingGen ? "Testing…" : "Test Generator"}
              </button>
            </div>
            {genTestMsg && <div className="ping-status">{genTestMsg}</div>}
            <div className="grid-2">
              <label className="field-label">
                <span>Model / Deployment</span>
                <input
                  type="text"
                  placeholder={
                    serverGeneratorModel
                      ? `.env (${serverGeneratorModel})`
                      : "azure/gpt-4o"
                  }
                  value={draft.generatorModel}
                  onChange={(e) =>
                    setDraft({ ...draft, generatorModel: e.target.value })
                  }
                />
              </label>
              <label className="field-label">
                <span>API Key</span>
                <input
                  type="password"
                  placeholder={
                    serverGeneratorConfigured ? ".env configured" : "API key"
                  }
                  value={draft.generatorApiKey}
                  onChange={(e) =>
                    setDraft({ ...draft, generatorApiKey: e.target.value })
                  }
                />
              </label>
            </div>
            <div className="grid-2">
              <label className="field-label">
                <span>Azure Endpoint</span>
                <input
                  type="text"
                  placeholder={
                    serverGeneratorConfigured
                      ? ".env configured"
                      : "https://<res>.openai.azure.com"
                  }
                  value={draft.generatorApiBase}
                  onChange={(e) =>
                    setDraft({ ...draft, generatorApiBase: e.target.value })
                  }
                />
              </label>
              <label className="field-label">
                <span>API Version</span>
                <input
                  type="text"
                  placeholder="2024-10-21"
                  value={draft.generatorApiVersion}
                  onChange={(e) =>
                    setDraft({ ...draft, generatorApiVersion: e.target.value })
                  }
                />
              </label>
            </div>
          </section>

          <section className="settings-section">
            <h3>Test Judges</h3>
            <div className="judge-ping-list">
              {ALL_JUDGES.map((j) => {
                const res = testResults[j.id];
                return (
                  <div key={j.id} className="judge-ping-row">
                    <div className="judge-ping-meta">
                      <strong>{j.label}</strong>
                      {res && <div className="ping-status">{res.msg}</div>}
                    </div>
                    <button
                      type="button"
                      className="btn-secondary btn-xs"
                      disabled={testingJudge !== null}
                      onClick={() => handleTest(j.id)}
                    >
                      {testingJudge === j.id ? "…" : "Test"}
                    </button>
                  </div>
                );
              })}
            </div>
          </section>
        </div>

        <div className="drawer-footer">
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              const cleared: LocalClientSettings = {
                apiKey: "",
                clefApiUrl: "",
                clefApiKey: "",
                openrouterApiKey: "",
                openaiApiKey: "",
                generatorModel: "",
                generatorApiKey: "",
                generatorApiBase: "",
                generatorApiVersion: "",
              };
              setDraft(cleared);
              onSave(cleared);
            }}
          >
            Clear
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={() => {
              onSave(draft);
              onClose();
            }}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
