import type { CreateProjectResponse, Note, Project, Tier } from "./types";

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function health(): Promise<{
  status: string;
  mock_mode: boolean;
  r2_configured: boolean;
  runpod_configured: boolean;
}> {
  return json(await fetch(`${API_BASE}/health`));
}

export async function listProjects(): Promise<Project[]> {
  return json(await fetch(`${API_BASE}/api/projects`));
}

export async function getProject(id: string): Promise<Project> {
  const r = await json<{ project: Project }>(
    await fetch(`${API_BASE}/api/projects/${id}`)
  );
  return r.project;
}

export async function createProject(params: {
  name: string;
  filename: string;
  tier: Tier;
  contentType: string;
}): Promise<CreateProjectResponse> {
  return json(
    await fetch(`${API_BASE}/api/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: params.name,
        filename: params.filename,
        tier: params.tier,
        content_type: params.contentType,
      }),
    })
  );
}

export async function uploadToPresigned(
  url: string,
  file: File,
  onProgress?: (fraction: number) => void
): Promise<void> {
  // Use XHR for upload progress (fetch has no upload progress events).
  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url, true);
    xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new Error(`upload failed: ${xhr.status} ${xhr.responseText}`));
    xhr.onerror = () => reject(new Error("upload network error"));
    xhr.send(file);
  });
}

export async function separate(id: string): Promise<Project> {
  const r = await json<{ project: Project }>(
    await fetch(`${API_BASE}/api/projects/${id}/separate`, { method: "POST" })
  );
  return r.project;
}

export async function analyzeStem(projectId: string, stemName: string): Promise<Project> {
  const r = await json<{ project: Project }>(
    await fetch(`${API_BASE}/api/projects/${projectId}/stems/${stemName}/analyze`, {
      method: "POST",
    })
  );
  return r.project;
}

export async function saveNotes(
  projectId: string,
  stemName: string,
  notes: Note[]
): Promise<Project> {
  const r = await json<{ project: Project }>(
    await fetch(`${API_BASE}/api/projects/${projectId}/stems/${stemName}/notes`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(notes),
    })
  );
  return r.project;
}

export async function ddspInstruments(): Promise<string[]> {
  const r = await json<{ instruments: string[] }>(
    await fetch(`${API_BASE}/api/projects/meta/ddsp-instruments`)
  );
  return r.instruments;
}

export async function toneTransfer(
  projectId: string,
  stemName: string,
  instrument: string
): Promise<{ status: string; instrument: string; url: string }> {
  return json(
    await fetch(
      `${API_BASE}/api/projects/${projectId}/stems/${stemName}/tone-transfer?instrument=${encodeURIComponent(
        instrument
      )}`,
      { method: "POST" }
    )
  );
}

export { API_BASE };
