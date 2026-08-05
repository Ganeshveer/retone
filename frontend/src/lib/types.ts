export type Tier = "2stem" | "4stem" | "6stem";

export type ProjectStatus =
  | "created"
  | "uploaded"
  | "separating"
  | "ready"
  | "failed";

export interface Note {
  id: string;
  start: number; // seconds
  dur: number; // seconds
  midi: number; // current (possibly edited) pitch
  original_midi: number; // detected pitch (for reset / cents)
  confidence: number;
}

export interface Stem {
  name: string;
  key: string;
  color: string;
  url?: string | null;
  analyzed?: boolean;
  notes?: Note[];
}

export type SnapMode = "none" | "chromatic" | "scale";

export interface Project {
  id: string;
  name: string;
  original_filename: string;
  tier: Tier;
  status: ProjectStatus;
  upload_key: string;
  created_at: string;
  job_id?: string | null;
  error?: string | null;
  stems: Stem[];
  bpm?: number | null;
  musical_key?: string | null;
  duration_seconds?: number | null;
}

export interface CreateProjectResponse {
  project: Project;
  upload_url: string;
  upload_method: string;
}

export const TIER_LABELS: Record<Tier, string> = {
  "2stem": "2 stems · Vocals + Instrumental",
  "4stem": "4 stems · Vocals · Drums · Bass · Other",
  "6stem": "6 stems · + Guitar · Piano",
};
