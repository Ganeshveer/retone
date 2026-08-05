import { offlinePitchShift } from "./signalStretch";
import type { Note } from "./types";

/**
 * Render a stem's audio with per-note pitch edits applied (Melodyne-style).
 * The original buffer is never mutated: for each edited note we slice its samples (with a
 * guard region), pitch-shift that slice formant-preserved, and equal-power crossfade it
 * back into a fresh copy. Returns null when there are no edits (caller keeps the original).
 */
export async function renderStemWithEdits(
  original: AudioBuffer,
  notes: Note[]
): Promise<AudioBuffer | null> {
  const edited = notes.filter((n) => Math.abs(n.midi - n.original_midi) > 0.01);
  if (edited.length === 0) return null;

  const numCh = original.numberOfChannels;
  const len = original.length;
  const sr = original.sampleRate;

  const out: Float32Array[] = [];
  for (let c = 0; c < numCh; c++) out.push(Float32Array.from(original.getChannelData(c)));

  const guard = Math.round(0.03 * sr);
  const xfade = Math.max(1, Math.round(0.008 * sr));

  for (const n of edited) {
    const semis = n.midi - n.original_midi;
    const s0 = Math.max(0, Math.floor(n.start * sr));
    const s1 = Math.min(len, Math.ceil((n.start + n.dur) * sr));
    const a = Math.max(0, s0 - guard - xfade);
    const b = Math.min(len, s1 + guard + xfade);
    if (b - a < 4) continue;

    const slice: Float32Array[] = [];
    for (let c = 0; c < numCh; c++) slice.push(original.getChannelData(c).slice(a, b));

    const shifted = await offlinePitchShift(slice, sr, semis);

    const keepStart = Math.max(a, s0 - xfade);
    const keepEnd = Math.min(b, s1 + xfade);
    for (let c = 0; c < numCh; c++) {
      const dst = out[c];
      const sh = shifted[c];
      for (let i = keepStart; i < keepEnd; i++) {
        const sv = sh[i - a];
        if (sv === undefined) continue;
        if (i < keepStart + xfade) {
          const t = (i - keepStart) / xfade;
          dst[i] = Math.cos((t * Math.PI) / 2) * dst[i] + Math.sin((t * Math.PI) / 2) * sv;
        } else if (i >= keepEnd - xfade) {
          const t = (keepEnd - i) / xfade;
          dst[i] = Math.cos((t * Math.PI) / 2) * dst[i] + Math.sin((t * Math.PI) / 2) * sv;
        } else {
          dst[i] = sv;
        }
      }
    }
  }

  let maxDiff = 0;
  for (let c = 0; c < numCh; c++) {
    const orig = original.getChannelData(c);
    for (let i = 0; i < len; i++) {
      const d = Math.abs(out[c][i] - orig[i]);
      if (d > maxDiff) maxDiff = d;
    }
  }
  console.log(
    `[retone] pitch-render: ${edited.length} edited note(s), maxΔ=${maxDiff.toFixed(4)}`
  );

  const buf = new AudioBuffer({ length: len, numberOfChannels: numCh, sampleRate: sr });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for (let c = 0; c < numCh; c++) buf.copyToChannel(out[c] as any, c);
  return buf;
}
