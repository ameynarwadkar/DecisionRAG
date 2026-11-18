import {
  type CSSProperties,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  api,
  isJudgeReady,
  loadLocalSettings,
  saveLocalSettings,
  subscribeToRunSteps,
} from "./api";
import { PassageModal } from "./components/PassageModal";
import { PixelMatrixSpinner, PixelScanBar, PixelTreeGlyph } from "./components/PixelArt";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { Sidebar } from "./components/Sidebar";
import { TelemetryPanel } from "./components/TelemetryPanel";
import { Workbench } from "./components/Workbench";
import { loadLayoutPrefs, updateLayoutPrefs } from "./layoutPrefs";
import type {
  ChatMessageEntry,
  DocumentPageRead,
  JudgeComparisonEntry,
  JudgeId,
  JudgeStatusInfo,
  LocalClientSettings,
  MCMode,
  NamespaceDetailRead,
  NamespaceRead,
  NavigationTelemetry,
  PassageRead,
  PipelineStepId,
  ProfileName,
  RankedPassageItem,
  WorkbenchMode,
} from "./types";

const JUDGE_LABELS: Record<JudgeId, string> = {
  clef: "Clef-Flash",
  openrouter_jev: "Jev 1.13",
  openrouter_luna: "OR Luna",
  openai_luna: "OpenAI Luna",
};

export function App() {
  const [localSettings, setLocalSettings] = useState<LocalClientSettings>(() =>
    loadLocalSettings()
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState<boolean>(
    () => loadLayoutPrefs().leftCollapsed
  );
  const [rightCollapsed, setRightCollapsed] = useState<boolean>(
    () => loadLayoutPrefs().rightCollapsed
  );

  const toggleLeftSidebar = () => {
    setLeftCollapsed((prev) => {
      const next = !prev;
      updateLayoutPrefs({ leftCollapsed: next });
      return next;
    });
  };

  const toggleRightSidebar = () => {
    setRightCollapsed((prev) => {
      const next = !prev;
      updateLayoutPrefs({ rightCollapsed: next });
      return next;
    });
  };

  // Health & Top-bar navigation config
  const [judges, setJudges] = useState<JudgeStatusInfo[]>([]);
  const [serverGeneratorModel, setServerGeneratorModel] = useState<string>("");
  const [serverGeneratorConfigured, setServerGeneratorConfigured] =
    useState<boolean>(false);
  const [selectedJudge, setSelectedJudge] = useState<JudgeId>("clef");

  // Live .env / runtime API probe state
  const [verifyingApis, setVerifyingApis] = useState<boolean>(false);
  const [needsJudgeInput, setNeedsJudgeInput] = useState<boolean>(false);
  const [judgeIssue, setJudgeIssue] = useState<string | null>(null);
  const [needsGeneratorInput, setNeedsGeneratorInput] = useState<boolean>(false);
  const [generatorIssue, setGeneratorIssue] = useState<string | null>(null);
  const [generatorDismissed, setGeneratorDismissed] = useState<boolean>(false);
  const [selectedProfile, setSelectedProfile] = useState<ProfileName>("auto");
  const [selectedMcMode, setSelectedMcMode] = useState<MCMode>("order");

  // Corpus state
  const [namespaces, setNamespaces] = useState<NamespaceRead[]>([]);
  const [activeNamespaceId, setActiveNamespaceId] = useState<string | null>(null);
  const [namespaceDetail, setNamespaceDetail] =
    useState<NamespaceDetailRead | null>(null);
  const [documents, setDocuments] = useState<DocumentPageRead[]>([]);

  // Workbench & Telemetry state
  const [mode, setMode] = useState<WorkbenchMode>("query");
  const [busyUpload, setBusyUpload] = useState(false);
  const [busyCluster, setBusyCluster] = useState(false);
  const [busyRun, setBusyRun] = useState(false);
  const [activePipelineStep, setActivePipelineStep] =
    useState<PipelineStepId | null>(null);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const [lastAnswer, setLastAnswer] = useState<string | null>(null);
  const [rankedPassages, setRankedPassages] = useState<RankedPassageItem[]>([]);
  const [telemetry, setTelemetry] = useState<NavigationTelemetry | null>(null);
  const [generatorLatencyMs, setGeneratorLatencyMs] = useState<number | null>(
    null
  );

  // Chat state
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessageEntry[]>([]);
  const [notepadPageIds, setNotepadPageIds] = useState<string[]>([]);

  // Compare state
  const [selectedCompareJudges, setSelectedCompareJudges] = useState<JudgeId[]>([
    "clef",
    "openrouter_jev",
  ]);
  const [compareEntries, setCompareEntries] = useState<JudgeComparisonEntry[]>(
    []
  );

  // Passage Inspector modal
  const [inspectPage, setInspectPage] = useState<DocumentPageRead | null>(null);
  const [inspectPassages, setInspectPassages] = useState<PassageRead[]>([]);
  const [loadingPassages, setLoadingPassages] = useState(false);
  const topUploadInputRef = useRef<HTMLInputElement | null>(null);

  const refreshHealthAndNamespaces = useCallback(
    async (cfg: LocalClientSettings, forceJudge?: JudgeId) => {
      setVerifyingApis(true);
      try {
        const health = await api.getHealth(cfg);
        setJudges(health.judges);
        setServerGeneratorModel(health.generator_model || "");
        const genConfigured = Boolean(health.generator_configured);
        setServerGeneratorConfigured(genConfigured);

        const candidateJudge =
          forceJudge ||
          health.judges.find((j) => isJudgeReady(j.judge_id, j.configured, cfg))
            ?.judge_id ||
          health.default_judge;
        setSelectedJudge(candidateJudge);

        const candidateInfo = health.judges.find(
          (j) => j.judge_id === candidateJudge
        );
        const judgeHasCreds = isJudgeReady(
          candidateJudge,
          Boolean(candidateInfo?.configured),
          cfg
        );

        // 1. Live-verify Decision Judge from .env / runtime overrides
        if (!judgeHasCreds) {
          setNeedsJudgeInput(true);
          setJudgeIssue("Not set in .env");
        } else {
          try {
            await api.testJudge(candidateJudge, cfg);
            setNeedsJudgeInput(false);
            setJudgeIssue(null);
          } catch {
            setNeedsJudgeInput(true);
            setJudgeIssue("Unreachable");
          }
        }

        // 2. Live-verify Generator LLM from .env / runtime overrides
        const genHasCreds = genConfigured || Boolean(cfg.generatorApiKey.trim());
        if (!genHasCreds) {
          setNeedsGeneratorInput(true);
          setGeneratorIssue("Not set in .env");
        } else {
          try {
            await api.testGenerator(cfg);
            setNeedsGeneratorInput(false);
            setGeneratorIssue(null);
          } catch {
            setNeedsGeneratorInput(true);
            setGeneratorIssue("Unreachable");
          }
        }
      } catch (err) {
        setErrorBanner(err instanceof Error ? err.message : String(err));
      } finally {
        setVerifyingApis(false);
      }

      try {
        const nsList = await api.listNamespaces(cfg);
        setNamespaces(nsList);
        setActiveNamespaceId((prev) => {
          if (prev && nsList.some((n) => n.id === prev)) return prev;
          const demoNs = nsList.find((n) => n.slug === "cloud-platform-docs");
          if (demoNs) return demoNs.id;
          return nsList.length > 0 ? nsList[0].id : null;
        });
      } catch (err) {
        setErrorBanner(err instanceof Error ? err.message : String(err));
      }
    },
    []
  );

  const refreshActiveNamespaceData = useCallback(
    async (nsId: string, cfg: LocalClientSettings) => {
      try {
        const [detail, docs] = await Promise.all([
          api.getNamespace(nsId, cfg),
          api.listDocuments(nsId, cfg),
        ]);
        setNamespaceDetail(detail);
        setDocuments(docs);
      } catch (err) {
        setErrorBanner(err instanceof Error ? err.message : String(err));
      }
    },
    []
  );

  useEffect(() => {
    void refreshHealthAndNamespaces(localSettings);
  }, [localSettings, refreshHealthAndNamespaces]);

  useEffect(() => {
    if (!activeNamespaceId) {
      setNamespaceDetail(null);
      setDocuments([]);
      return;
    }
    setLastAnswer(null);
    setRankedPassages([]);
    setTelemetry(null);
    setGeneratorLatencyMs(null);
    setChatSessionId(null);
    setChatMessages([]);
    setNotepadPageIds([]);
    setCompareEntries([]);
    void refreshActiveNamespaceData(activeNamespaceId, localSettings);
  }, [activeNamespaceId, localSettings, refreshActiveNamespaceData]);

  const handleSaveSettings = (next: LocalClientSettings) => {
    saveLocalSettings(next);
    setLocalSettings(next);
    setErrorBanner(null);
  };

  const handleApplyFallbackApis = async (
    next: LocalClientSettings,
    judgeToVerify: JudgeId
  ) => {
    saveLocalSettings(next);
    setLocalSettings(next);
    setErrorBanner(null);
    await refreshHealthAndNamespaces(next, judgeToVerify);
  };

  const handleCreateNamespace = async (payload: {
    name: string;
    description?: string;
    default_profile: ProfileName;
  }) => {
    setErrorBanner(null);
    try {
      const created = await api.createNamespace(payload, localSettings);
      const nsList = await api.listNamespaces(localSettings);
      setNamespaces(nsList);
      setActiveNamespaceId(created.id);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteNamespace = async (nsId: string) => {
    setErrorBanner(null);
    try {
      await api.deleteNamespace(nsId, localSettings);
      const nsList = await api.listNamespaces(localSettings);
      setNamespaces(nsList);
      setActiveNamespaceId(nsList.length > 0 ? nsList[0].id : null);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUploadFiles = async (
    files: FileList,
    category: string,
    mergePartSuffix: boolean
  ) => {
    setBusyUpload(true);
    setErrorBanner(null);
    try {
      let targetNsId = activeNamespaceId;
      if (!targetNsId) {
        const created = await api.createNamespace(
          { name: "Default", default_profile: selectedProfile },
          localSettings
        );
        targetNsId = created.id;
        setActiveNamespaceId(targetNsId);
      }

      for (const file of Array.from(files)) {
        await api.uploadDocument(
          targetNsId,
          file,
          { category, mergePartSuffix },
          localSettings
        );
      }
      const nsList = await api.listNamespaces(localSettings);
      setNamespaces(nsList);
      await refreshActiveNamespaceData(targetNsId, localSettings);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyUpload(false);
    }
  };

  const handleDeletePage = async (pageId: string) => {
    if (!activeNamespaceId) return;
    setErrorBanner(null);
    try {
      await api.deleteDocument(activeNamespaceId, pageId, localSettings);
      const nsList = await api.listNamespaces(localSettings);
      setNamespaces(nsList);
      await refreshActiveNamespaceData(activeNamespaceId, localSettings);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    }
  };

  const handleInspectPage = async (page: DocumentPageRead) => {
    if (!activeNamespaceId) return;
    setInspectPage(page);
    setInspectPassages([]);
    setLoadingPassages(true);
    try {
      const pList = await api.getPassages(
        activeNamespaceId,
        page.id,
        localSettings
      );
      setInspectPassages(pList);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingPassages(false);
    }
  };

  const handleBuildClusters = async (numClusters?: number) => {
    if (!activeNamespaceId) return;
    setBusyCluster(true);
    setErrorBanner(null);
    try {
      await api.buildClusters(
        activeNamespaceId,
        { num_clusters: numClusters, force: true },
        localSettings
      );
      await refreshActiveNamespaceData(activeNamespaceId, localSettings);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyCluster(false);
    }
  };

  const beginPipelineStream = (): [string, () => void] => {
    const runId = crypto.randomUUID();
    setActivePipelineStep("query");
    const unsub = subscribeToRunSteps(runId, (step) => {
      setActivePipelineStep(step);
    });
    return [runId, unsub];
  };

  const handleRunQueryOrRetrieve = async (query: string, scenario: string) => {
    if (!activeNamespaceId) return;
    setBusyRun(true);
    setErrorBanner(null);
    const [runId, unsub] = beginPipelineStream();
    const nav = {
      judge: selectedJudge,
      profile: selectedProfile,
      mc_mode: selectedMcMode,
    };
    try {
      if (mode === "retrieve") {
        const res = await api.retrieve(
          activeNamespaceId,
          query,
          scenario,
          nav,
          localSettings,
          runId
        );
        setLastAnswer(null);
        setRankedPassages(res.passages);
        setTelemetry(res.telemetry);
        setGeneratorLatencyMs(null);
      } else {
        const res = await api.query(
          activeNamespaceId,
          query,
          scenario,
          nav,
          localSettings,
          runId
        );
        setLastAnswer(res.answer);
        setRankedPassages(res.passages);
        setTelemetry(res.telemetry);
        setGeneratorLatencyMs(res.generator_latency_ms);
      }
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    } finally {
      unsub();
      setActivePipelineStep(null);
      setBusyRun(false);
    }
  };

  const handleSendChat = async (message: string) => {
    if (!activeNamespaceId) return;
    setBusyRun(true);
    setErrorBanner(null);
    const [runId, unsub] = beginPipelineStream();
    setChatMessages((prev) => [...prev, { role: "user", content: message }]);
    try {
      const res = await api.chat(
        activeNamespaceId,
        message,
        chatSessionId,
        {
          judge: selectedJudge,
          profile: selectedProfile,
          mc_mode: selectedMcMode,
        },
        localSettings,
        runId
      );
      setChatSessionId(res.session_id);
      setNotepadPageIds(res.notepad_page_ids);
      setRankedPassages(res.passages);
      setTelemetry(res.telemetry);
      setGeneratorLatencyMs(res.generator_latency_ms);
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          passages: res.passages,
          telemetry: res.telemetry,
          generator_latency_ms: res.generator_latency_ms,
        },
      ]);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    } finally {
      unsub();
      setActivePipelineStep(null);
      setBusyRun(false);
    }
  };

  const handleRunCompare = async (query: string, scenario: string) => {
    if (!activeNamespaceId || selectedCompareJudges.length === 0) return;
    setBusyRun(true);
    setErrorBanner(null);
    const [runId, unsub] = beginPipelineStream();
    try {
      const res = await api.compareJudges(
        activeNamespaceId,
        query,
        scenario,
        selectedCompareJudges,
        {
          profile: selectedProfile,
          mc_mode: selectedMcMode,
        },
        localSettings,
        runId
      );
      setCompareEntries(res.results);
      if (res.results.length > 0) {
        setTelemetry(res.results[0].telemetry);
      }
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : String(err));
    } finally {
      unsub();
      setActivePipelineStep(null);
      setBusyRun(false);
    }
  };

  const toggleCompareJudge = (j: JudgeId) => {
    setSelectedCompareJudges((prev) =>
      prev.includes(j) ? prev.filter((x) => x !== j) : [...prev, j]
    );
  };

  const activeJudgeStatus = judges.find((j) => j.judge_id === selectedJudge);
  const judgeReady = isJudgeReady(
    selectedJudge,
    Boolean(activeJudgeStatus?.configured),
    localSettings
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <PixelTreeGlyph active={busyRun || busyUpload || busyCluster} />
          <strong>DecisionRAG</strong>
          <span className="pixel-cursor" aria-hidden="true" />
        </div>

        <div className="topbar-controls">
          <label className="control-pill">
            <span className={`status-dot ${judgeReady ? "dot-ok" : "dot-warn"}`} />
            <span className="control-label">Judge</span>
            <select
              value={selectedJudge}
              onChange={(e) => setSelectedJudge(e.target.value as JudgeId)}
            >
              {(["clef", "openrouter_jev", "openrouter_luna", "openai_luna"] as JudgeId[]).map(
                (jid) => {
                  const st = judges.find((x) => x.judge_id === jid);
                  const rdy = isJudgeReady(
                    jid,
                    Boolean(st?.configured),
                    localSettings
                  );
                  return (
                    <option key={jid} value={jid}>
                      {rdy ? "● " : "○ "}
                      {JUDGE_LABELS[jid]}
                    </option>
                  );
                }
              )}
            </select>
          </label>

          <label className="control-pill">
            <span className="control-label">Profile</span>
            <select
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value as ProfileName)}
            >
              <option value="auto">auto</option>
              <option value="cqa">cqa</option>
              <option value="md2d">md2d</option>
              <option value="mtrag">mtrag</option>
              <option value="nq">nq</option>
            </select>
          </label>

          <label className="control-pill">
            <span className="control-label">MC</span>
            <select
              value={selectedMcMode}
              onChange={(e) => setSelectedMcMode(e.target.value as MCMode)}
            >
              <option value="order">order</option>
              <option value="top1">top1</option>
              <option value="off">off</option>
            </select>
          </label>

          <input
            ref={topUploadInputRef}
            type="file"
            multiple
            accept=".pdf,.md,.markdown,.html,.htm,.txt"
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                void handleUploadFiles(e.target.files, "", false);
                e.target.value = "";
              }
            }}
          />
          <button
            type="button"
            className="btn-primary btn-sm"
            disabled={busyUpload}
            onClick={() => topUploadInputRef.current?.click()}
          >
            {busyUpload ? <PixelMatrixSpinner label="Upload" /> : "+ Upload"}
          </button>

          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={() => setSettingsOpen(true)}
          >
            Settings
          </button>
        </div>
      </header>

      {(busyRun || busyUpload || busyCluster) && <PixelScanBar />}

      <div
        className="workspace-grid"
        style={
          {
            "--left-col": leftCollapsed ? "32px" : "270px",
            "--right-col": rightCollapsed ? "32px" : "290px",
          } as CSSProperties
        }
      >
        <Sidebar
          collapsed={leftCollapsed}
          onToggleCollapse={toggleLeftSidebar}
          namespaces={namespaces}
          activeNamespaceId={activeNamespaceId}
          namespaceDetail={namespaceDetail}
          documents={documents}
          busyUpload={busyUpload}
          busyCluster={busyCluster}
          onSelectNamespace={setActiveNamespaceId}
          onCreateNamespace={handleCreateNamespace}
          onDeleteNamespace={handleDeleteNamespace}
          onUploadFiles={handleUploadFiles}
          onInspectPage={(p) => void handleInspectPage(p)}
          onDeletePage={handleDeletePage}
          onBuildClusters={handleBuildClusters}
        />

        <Workbench
          mode={mode}
          onChangeMode={setMode}
          activePipelineStep={activePipelineStep}
          activeJudge={selectedJudge}
          hasNotepad={mode === "chat" && notepadPageIds.length > 0}
          hasNamespace={Boolean(activeNamespaceId)}
          hasDocuments={documents.length > 0}
          isDemoCorpus={namespaceDetail?.slug === "cloud-platform-docs"}
          busyUpload={busyUpload}
          busyRun={busyRun}
          errorBanner={errorBanner}
          onOpenSettings={() => setSettingsOpen(true)}
          onQuickUpload={(files) => handleUploadFiles(files, "", false)}
          localSettings={localSettings}
          verifyingApis={verifyingApis}
          needsJudgeInput={needsJudgeInput}
          judgeIssue={judgeIssue}
          needsGeneratorInput={
            needsGeneratorInput &&
            !generatorDismissed &&
            (mode === "query" || mode === "chat")
          }
          generatorIssue={generatorIssue}
          onSelectJudge={setSelectedJudge}
          onApplyFallbackApis={handleApplyFallbackApis}
          onRecheckEnv={() => refreshHealthAndNamespaces(localSettings, selectedJudge)}
          onDismissGenerator={() => setGeneratorDismissed(true)}
          lastAnswer={lastAnswer}
          rankedPassages={rankedPassages}
          chatMessages={chatMessages}
          chatSessionId={chatSessionId}
          onResetChat={() => {
            setChatSessionId(null);
            setChatMessages([]);
            setNotepadPageIds([]);
          }}
          compareEntries={compareEntries}
          selectedCompareJudges={selectedCompareJudges}
          onToggleCompareJudge={toggleCompareJudge}
          onRunQueryOrRetrieve={handleRunQueryOrRetrieve}
          onSendChat={handleSendChat}
          onRunCompare={handleRunCompare}
        />

        <TelemetryPanel
          collapsed={rightCollapsed}
          onToggleCollapse={toggleRightSidebar}
          telemetry={telemetry}
          generatorLatencyMs={generatorLatencyMs}
          notepadPageIds={notepadPageIds}
          documents={documents}
          onInspectPage={(p) => void handleInspectPage(p)}
        />
      </div>

      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={localSettings}
        onSave={handleSaveSettings}
        judges={judges}
        serverGeneratorModel={serverGeneratorModel}
        serverGeneratorConfigured={serverGeneratorConfigured}
      />

      <PassageModal
        page={inspectPage}
        passages={inspectPassages}
        loading={loadingPassages}
        onClose={() => setInspectPage(null)}
      />
    </div>
  );
}
