import { useState } from "react";
import "./styles/app.css";
import * as api from "./lib/api";
import DawView from "./components/DawView";
import LibraryView from "./components/LibraryView";
import UploadPanel from "./components/UploadPanel";
import type { Project } from "./lib/types";

type View = "library" | "upload" | "daw";

export default function App() {
  const [view, setView] = useState<View>("library");
  const [active, setActive] = useState<Project | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);

  const openProject = async (p: Project) => {
    try {
      const fresh = await api.getProject(p.id); // fresh presigned urls
      if (fresh.status === "ready" && fresh.stems.length) {
        setActive(fresh);
        setView("daw");
      } else {
        setNotice(
          fresh.status === "separating"
            ? "This session is still separating — try again in a moment."
            : `This session is ${fresh.status}; re-run separation to open it.`
        );
        setTimeout(() => setNotice(null), 4000);
      }
    } catch (e) {
      setNotice(String(e));
      setTimeout(() => setNotice(null), 4000);
    }
  };

  return (
    <>
      {notice && <div className="toast">{notice}</div>}

      {view === "library" && (
        <LibraryView
          refreshKey={refreshKey}
          onNew={() => setView("upload")}
          onOpen={openProject}
        />
      )}

      {view === "upload" && (
        <UploadPanel
          onCancel={() => setView("library")}
          onComplete={(p) => {
            setActive(p);
            setRefreshKey((k) => k + 1);
            setView("daw");
          }}
        />
      )}

      {view === "daw" && active && (
        <DawView
          project={active}
          onBack={() => {
            setRefreshKey((k) => k + 1);
            setView("library");
          }}
        />
      )}
    </>
  );
}
