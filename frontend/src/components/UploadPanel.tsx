import { useRef, useState } from "react";
import * as api from "../lib/api";
import type { Project, Tier } from "../lib/types";
import { TIER_LABELS } from "../lib/types";
import { CassetteIcon, UploadIcon } from "./icons";

type Phase = "idle" | "uploading" | "separating" | "error";

const TIERS: Tier[] = ["2stem", "4stem", "6stem"];

interface Props {
  onComplete: (project: Project) => void;
  onCancel: () => void;
}

export default function UploadPanel({ onComplete, onCancel }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [tier, setTier] = useState<Tier>("4stem");
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pickFile = (f: File | null) => {
    if (!f) return;
    setFile(f);
    if (!name) setName(f.name.replace(/\.[^.]+$/, ""));
  };

  const run = async () => {
    if (!file) return;
    try {
      setPhase("uploading");
      setStatus("Creating project…");
      const created = await api.createProject({
        name: name || file.name,
        filename: file.name,
        tier,
        contentType: file.type || "application/octet-stream",
      });

      setStatus("Uploading audio…");
      await api.uploadToPresigned(created.upload_url, file, setProgress);

      setPhase("separating");
      setStatus("Separating stems…");
      let project = await api.separate(created.project.id);

      // Poll until ready (mock returns ready immediately; real RunPod takes ~30–90s).
      const deadline = Date.now() + 5 * 60 * 1000;
      while (project.status === "separating" && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 2500));
        project = await api.getProject(project.id);
      }

      if (project.status === "ready") {
        onComplete(project);
      } else {
        throw new Error(project.error || `separation ${project.status}`);
      }
    } catch (e) {
      setStatus(String(e));
      setPhase("error");
    }
  };

  const busy = phase === "uploading" || phase === "separating";

  return (
    <div className="upload-wrap">
      <div className="panel upload-card">
        <div className="upload-head">
          <h2>New reprocessing session</h2>
          <button className="btn btn-ghost" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
        </div>

        <div
          className={`dropzone ${dragOver ? "drag" : ""} ${file ? "has-file" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            pickFile(e.dataTransfer.files?.[0] ?? null);
          }}
          onClick={() => !busy && inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="audio/*"
            hidden
            onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <div className="drop-file">
              <CassetteIcon size={30} />
              <div>
                <div className="drop-file-name">{file.name}</div>
                <div className="muted">{(file.size / 1e6).toFixed(1)} MB</div>
              </div>
            </div>
          ) : (
            <div className="drop-empty">
              <UploadIcon size={30} />
              <div>
                Drop an audio file here, or <span className="link">browse</span>
              </div>
              <div className="muted">MP3, WAV, FLAC…</div>
            </div>
          )}
        </div>

        <label className="field">
          <span className="field-label">Session name</span>
          <input
            className="text-input"
            value={name}
            placeholder="Untitled session"
            onChange={(e) => setName(e.target.value)}
            disabled={busy}
          />
        </label>

        <div className="field">
          <span className="field-label">Separation tier</span>
          <div className="tier-row">
            {TIERS.map((t) => (
              <button
                key={t}
                className={`tier-opt ${tier === t ? "sel" : ""}`}
                onClick={() => setTier(t)}
                disabled={busy}
              >
                <span className="tier-title">{t.replace("stem", " stem")}</span>
                <span className="tier-sub">{TIER_LABELS[t].split("·").slice(1).join("·").trim()}</span>
              </button>
            ))}
          </div>
        </div>

        {busy && (
          <div className="progress-block">
            {phase === "uploading" ? (
              <>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
                </div>
                <div className="muted mono">{Math.round(progress * 100)}% · {status}</div>
              </>
            ) : (
              <div className="separating">
                <div className="reel-loader" />
                <div className="muted">{status}</div>
              </div>
            )}
          </div>
        )}

        {phase === "error" && <div className="error-block">{status}</div>}

        <div className="upload-actions">
          <button className="btn btn-primary" onClick={run} disabled={!file || busy}>
            {busy ? "Working…" : "Separate stems"}
          </button>
        </div>
      </div>
    </div>
  );
}
