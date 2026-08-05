import { useEffect, useState } from "react";
import * as api from "../lib/api";
import type { Project } from "../lib/types";
import { TIER_LABELS } from "../lib/types";
import { CassetteIcon } from "./icons";

interface Props {
  onOpen: (project: Project) => void;
  onNew: () => void;
  refreshKey: number;
}

const STEM_DOTS: Record<string, string> = {
  vocals: "#E0564B",
  instrumental: "#4A8FC0",
  drums: "#E8A13A",
  bass: "#7C5CD0",
  guitar: "#3FA66A",
  piano: "#2E9FB0",
  other: "#8A93A6",
};

export default function LibraryView({ onOpen, onNew, refreshKey }: Props) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState<string>("");

  useEffect(() => {
    setLoading(true);
    Promise.all([api.listProjects(), api.health().catch(() => null)])
      .then(([ps, h]) => {
        setProjects(ps);
        if (h) setHealth(h.mock_mode ? "mock mode" : "live");
      })
      .catch(() => setProjects([]))
      .finally(() => setLoading(false));
  }, [refreshKey]);

  return (
    <div className="library">
      <header className="lib-header">
        <div className="brand">
          <span className="brand-mark">◉</span>
          <span className="brand-name">ReTone</span>
          {health && <span className={`env-chip ${health === "live" ? "live" : ""}`}>{health}</span>}
        </div>
        <button className="btn btn-primary" onClick={onNew}>
          + New session
        </button>
      </header>

      <div className="lib-body">
        <h1 className="lib-title">Library</h1>
        <p className="lib-sub">Upload a song, split it into stems, and open it in the editor.</p>

        {loading ? (
          <div className="muted">Loading…</div>
        ) : projects.length === 0 ? (
          <div className="empty-state">
            <CassetteIcon size={40} />
            <div className="empty-title">No sessions yet</div>
            <div className="muted">Create your first reprocessing session to get started.</div>
            <button className="btn btn-primary" onClick={onNew} style={{ marginTop: 16 }}>
              + New session
            </button>
          </div>
        ) : (
          <div className="proj-list">
            {projects.map((p) => (
              <button key={p.id} className="proj-row" onClick={() => onOpen(p)}>
                <span className="proj-thumb">
                  <CassetteIcon size={22} />
                </span>
                <span className="proj-info">
                  <span className="proj-title">{p.name}</span>
                  <span className="proj-file mono">{p.original_filename}</span>
                </span>
                <span className="proj-stems">
                  {(p.stems.length ? p.stems.map((s) => s.name) : tierStems(p.tier)).map((n) => (
                    <span
                      key={n}
                      className="stem-dot"
                      title={n}
                      style={{ background: STEM_DOTS[n] ?? "#8A93A6" }}
                    />
                  ))}
                </span>
                <span className="proj-tier-chip">{p.tier.replace("stem", " stem")}</span>
                <span className={`status-chip s-${p.status}`}>{p.status}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function tierStems(tier: string): string[] {
  const label = TIER_LABELS[tier as keyof typeof TIER_LABELS] ?? "";
  return label
    .split("·")
    .slice(1)
    .map((s) => s.trim().toLowerCase().replace("+ ", ""))
    .filter(Boolean);
}
