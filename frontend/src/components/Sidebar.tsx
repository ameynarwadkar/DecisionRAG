import { useRef, useState } from "react";
import type {
  DocumentPageRead,
  NamespaceDetailRead,
  NamespaceRead,
  ProfileName,
} from "../types";
import { PixelMatrixSpinner } from "./PixelArt";

interface SidebarProps {
  namespaces: NamespaceRead[];
  activeNamespaceId: string | null;
  namespaceDetail: NamespaceDetailRead | null;
  documents: DocumentPageRead[];
  busyUpload: boolean;
  busyCluster: boolean;
  onSelectNamespace: (id: string) => void;
  onCreateNamespace: (payload: {
    name: string;
    description?: string;
    default_profile: ProfileName;
  }) => Promise<void>;
  onDeleteNamespace: (id: string) => Promise<void>;
  onUploadFiles: (
    files: FileList,
    category: string,
    mergePartSuffix: boolean
  ) => Promise<void>;
  onInspectPage: (page: DocumentPageRead) => void;
  onDeletePage: (pageId: string) => Promise<void>;
  onBuildClusters: (numClusters?: number) => Promise<void>;
}

export function Sidebar({
  namespaces,
  activeNamespaceId,
  namespaceDetail,
  documents,
  busyUpload,
  busyCluster,
  onSelectNamespace,
  onCreateNamespace,
  onDeleteNamespace,
  onUploadFiles,
  onInspectPage,
  onDeletePage,
  onBuildClusters,
}: SidebarProps) {
  const [showCreate, setShowCreate] = useState(false);
  const [nsName, setNsName] = useState("");
  const [nsProfile, setNsProfile] = useState<ProfileName>("auto");
  const [categoryInput, setCategoryInput] = useState("");
  const [mergeParts, setMergeParts] = useState(false);
  const [clusterCountInput, setClusterCountInput] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!nsName.trim()) return;
    await onCreateNamespace({
      name: nsName.trim(),
      default_profile: nsProfile,
    });
    setNsName("");
    setShowCreate(false);
  };

  const categoryMap = new Map(
    (namespaceDetail?.categories || []).map((c) => [c.id, c.name])
  );

  return (
    <aside className="sidebar">
      <div className="panel-block">
        <div className="section-title-row">
          <span className="eyebrow">Namespace</span>
          <button
            type="button"
            className="btn-secondary btn-xs"
            onClick={() => setShowCreate((v) => !v)}
          >
            {showCreate ? "×" : "+ New"}
          </button>
        </div>

        {showCreate && (
          <form className="inline-card-form" onSubmit={handleCreateSubmit}>
            <input
              type="text"
              placeholder="Name"
              value={nsName}
              onChange={(e) => setNsName(e.target.value)}
              required
            />
            <div className="row-between">
              <select
                value={nsProfile}
                onChange={(e) => setNsProfile(e.target.value as ProfileName)}
              >
                <option value="auto">auto</option>
                <option value="cqa">cqa</option>
                <option value="md2d">md2d</option>
                <option value="mtrag">mtrag</option>
                <option value="nq">nq</option>
              </select>
              <button type="submit" className="btn-primary btn-xs">
                Add
              </button>
            </div>
          </form>
        )}

        {namespaces.length === 0 ? (
          <p className="empty-hint">Auto-created on upload</p>
        ) : (
          <div className="ns-selector-row">
            <select
              className="ns-select"
              value={activeNamespaceId || ""}
              onChange={(e) => onSelectNamespace(e.target.value)}
            >
              {namespaces.map((ns) => (
                <option key={ns.id} value={ns.id}>
                  {ns.name} ({ns.page_count})
                </option>
              ))}
            </select>
            {activeNamespaceId && (
              <button
                type="button"
                className="btn-ghost btn-xs"
                title="Delete"
                onClick={() => {
                  if (window.confirm("Delete namespace?")) {
                    void onDeleteNamespace(activeNamespaceId);
                  }
                }}
              >
                ×
              </button>
            )}
          </div>
        )}

        {namespaceDetail && (
          <div className="ns-stats-strip">
            <span>{namespaceDetail.page_count}p</span>
            <span>·</span>
            <span>{namespaceDetail.category_count}c</span>
            <span>·</span>
            <span>{namespaceDetail.cluster_count}k</span>
          </div>
        )}
      </div>

      <div className="panel-block">
        <span className="eyebrow">Upload Documents</span>
        <input
          type="text"
          placeholder="Category (optional)"
          value={categoryInput}
          onChange={(e) => setCategoryInput(e.target.value)}
        />
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={mergeParts}
            onChange={(e) => setMergeParts(e.target.checked)}
          />
          <span>Merge #1 / #2 parts</span>
        </label>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.md,.markdown,.html,.htm,.txt"
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              void onUploadFiles(e.target.files, categoryInput, mergeParts);
              e.target.value = "";
            }
          }}
        />
        <button
          type="button"
          className="btn-primary pixel-dither-hover"
          disabled={busyUpload}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
              void onUploadFiles(
                e.dataTransfer.files,
                categoryInput,
                mergeParts
              );
            }
          }}
        >
          {busyUpload ? <PixelMatrixSpinner label="Ingesting" /> : "+ Upload Documents"}
        </button>
        <span className="muted-xs">.pdf · .md · .html · .txt (or drag &amp; drop)</span>
      </div>

      <div className="panel-block docs-scroll-block">
        <span className="eyebrow">Documents ({documents.length})</span>

        {documents.length === 0 ? (
          <p className="empty-hint">Empty.</p>
        ) : (
          <div className="doc-list">
            {documents.map((doc) => {
              const catName = doc.category_id
                ? categoryMap.get(doc.category_id)
                : null;
              return (
                <div key={doc.id} className="doc-item">
                  <div
                    className="doc-item-main"
                    onClick={() => onInspectPage(doc)}
                  >
                    <div className="doc-item-title">{doc.title}</div>
                    <div className="doc-item-sub">
                      {catName && <span className="pill-cat">{catName}</span>}
                      <span>{doc.passage_count} ¶</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn-ghost btn-xs"
                    onClick={() => void onDeletePage(doc.id)}
                  >
                    ×
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {activeNamespaceId && documents.length >= 2 && (
        <div className="panel-block">
          <span className="eyebrow">Clusters</span>
          <div className="row-between">
            <input
              type="number"
              min={1}
              max={500}
              placeholder="k (auto)"
              value={clusterCountInput}
              onChange={(e) => setClusterCountInput(e.target.value)}
              style={{ width: "100px" }}
            />
            <button
              type="button"
              className="btn-secondary btn-xs"
              disabled={busyCluster}
              onClick={() => {
                const parsed = parseInt(clusterCountInput, 10);
                void onBuildClusters(
                  Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
                );
              }}
            >
              {busyCluster ? "…" : "Build"}
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
