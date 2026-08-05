import type { SnapMode } from "./types";

export const NOTE_NAMES = [
  "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/** MIDI 60 = C4 (C4=middle-C convention). */
export function noteName(midi: number): string {
  const m = Math.round(midi);
  return NOTE_NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
}

export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function isBlackKey(midi: number): boolean {
  return [1, 3, 6, 8, 10].includes(((Math.round(midi) % 12) + 12) % 12);
}

/** Cents deviation of a fractional MIDI value from its nearest semitone. */
export function centsOff(midiFractional: number): number {
  return Math.round((midiFractional - Math.round(midiFractional)) * 100);
}

export interface Scale {
  root: number; // pitch class 0..11
  intervals: number[]; // semitone offsets from root
  label: string;
}

export const SCALE_INTERVALS: Record<string, number[]> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

/** Parse "G major" / "A minor" -> Scale. Falls back to C major. */
export function parseKey(key: string | null | undefined): Scale {
  if (!key) return { root: 0, intervals: SCALE_INTERVALS.major, label: "C major" };
  const parts = key.trim().split(/\s+/);
  const pc = NOTE_NAMES.indexOf(parts[0]);
  const quality = (parts[1] || "major").toLowerCase();
  const intervals = SCALE_INTERVALS[quality] ?? SCALE_INTERVALS.major;
  return { root: pc >= 0 ? pc : 0, intervals, label: key };
}

export function isInScale(scale: Scale, midi: number): boolean {
  const pc = ((Math.round(midi) - scale.root) % 12 + 12) % 12;
  return scale.intervals.includes(pc);
}

/** Nearest in-scale semitone (walks outward from the chromatic rounding). */
export function snapToScale(midi: number, scale: Scale): number {
  const base = Math.round(midi);
  for (let d = 0; d <= 6; d++) {
    if (isInScale(scale, base + d)) return base + d;
    if (isInScale(scale, base - d)) return base - d;
  }
  return base;
}

export function applySnap(midiFractional: number, mode: SnapMode, scale: Scale): number {
  switch (mode) {
    case "none":
      return midiFractional;
    case "chromatic":
      return Math.round(midiFractional);
    case "scale":
      return snapToScale(midiFractional, scale);
  }
}
