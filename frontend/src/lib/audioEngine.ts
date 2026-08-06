import type { Note } from "./types";

/**
 * Multitrack Web Audio engine for the ReTone DAW.
 *
 * - Decodes each stem to an AudioBuffer and precomputes waveform peaks.
 * - Plays all stems sample-synced from a shared AudioContext clock.
 * - Per-stem gain / mute / solo; master gain -> analyser (real VU metering) -> output.
 */

export interface StemSource {
  name: string;
  color: string;
  url: string;
}

export interface LoadedStem {
  name: string;
  color: string;
  peaks: Float32Array;
  duration: number;
}

interface StemNode {
  name: string;
  color: string;
  buffer: AudioBuffer; // currently-playing buffer (may be an edited render)
  original: AudioBuffer; // immutable source for re-rendering edits
  gain: GainNode;
  peaks: Float32Array;
  volume: number;
  muted: boolean;
  soloed: boolean;
}

const PEAK_BUCKETS = 1600;

function computePeaks(buffer: AudioBuffer): Float32Array {
  const data = buffer.getChannelData(0);
  const block = Math.max(1, Math.floor(data.length / PEAK_BUCKETS));
  const peaks = new Float32Array(PEAK_BUCKETS);
  for (let i = 0; i < PEAK_BUCKETS; i++) {
    const start = i * block;
    const end = Math.min(start + block, data.length);
    let max = 0;
    for (let j = start; j < end; j++) {
      const v = Math.abs(data[j]);
      if (v > max) max = v;
    }
    peaks[i] = max;
  }
  return peaks;
}

export class AudioEngine {
  private ctx: AudioContext;
  private master: GainNode;
  private analyser: AnalyserNode;
  private levelBuf = new Uint8Array(1024);
  private stems = new Map<string, StemNode>();
  order: string[] = [];

  private sources = new Map<string, AudioBufferSourceNode>();

  // Instrument-change (resynthesis): play a stem's notes through a sampler instead of its
  // original audio. Sampler output routes to the stem's gain (mute/solo/volume still apply).
  private stemInstrument = new Map<string, string>();
  private instrumentSamplers = new Map<string, any>();
  private stemNotesMap = new Map<string, Note[]>();

  private playing = false;
  private startedAt = 0; // ctx time when the current play began
  private offset = 0; // seconds into the song at startedAt
  duration = 0;

  onStateChange?: () => void;

  constructor() {
    this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.master = this.ctx.createGain();
    this.master.gain.value = 1;
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.master.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
  }

  async load(stems: StemSource[]): Promise<LoadedStem[]> {
    const loaded: LoadedStem[] = [];
    this.order = [];
    for (const s of stems) {
      const res = await fetch(s.url);
      if (!res.ok) throw new Error(`failed to fetch stem ${s.name}: ${res.status}`);
      const arr = await res.arrayBuffer();
      const buffer = await this.ctx.decodeAudioData(arr);
      const peaks = computePeaks(buffer);
      const gain = this.ctx.createGain();
      gain.connect(this.master);
      this.stems.set(s.name, {
        name: s.name,
        color: s.color,
        buffer,
        original: buffer,
        gain,
        peaks,
        volume: 0.85,
        muted: false,
        soloed: false,
      });
      this.order.push(s.name);
      this.duration = Math.max(this.duration, buffer.duration);
      loaded.push({ name: s.name, color: s.color, peaks, duration: buffer.duration });
    }
    this.applyGains();
    return loaded;
  }

  private applyGains() {
    const anySolo = [...this.stems.values()].some((s) => s.soloed);
    const t = this.ctx.currentTime;
    for (const s of this.stems.values()) {
      const target = anySolo ? (s.soloed ? s.volume : 0) : s.muted ? 0 : s.volume;
      s.gain.gain.setTargetAtTime(target, t, 0.01);
    }
  }

  private stopSources() {
    for (const src of this.sources.values()) {
      try {
        src.onended = null;
        src.stop();
      } catch {
        /* already stopped */
      }
      src.disconnect();
    }
    this.sources.clear();
    for (const inst of this.instrumentSamplers.values()) {
      try {
        inst.stop();
      } catch {
        /* ignore */
      }
    }
  }

  private startSources(from: number) {
    this.sources.clear();
    this.startedAt = this.ctx.currentTime;
    for (const name of this.order) {
      const s = this.stems.get(name)!;
      if (this.instrumentSamplers.get(name)) {
        this.scheduleInstrument(name); // play notes via sampler instead of the buffer
        continue;
      }
      const src = this.ctx.createBufferSource();
      src.buffer = s.buffer;
      src.connect(s.gain);
      src.start(0, Math.min(from, s.buffer.duration));
      this.sources.set(name, src);
    }
  }

  /** Store the notes used to resynthesize a stem when an instrument is selected. */
  setStemNotes(name: string, notes: Note[]) {
    this.stemNotesMap.set(name, notes);
    if (this.playing && this.instrumentSamplers.get(name)) {
      try {
        this.instrumentSamplers.get(name).stop();
      } catch {
        /* ignore */
      }
      this.scheduleInstrument(name);
    }
  }

  hasInstrument(name: string): boolean {
    return this.instrumentSamplers.has(name);
  }

  /** Switch a stem to an instrument (GM name) or back to its original audio (null). */
  async setStemInstrument(name: string, instrumentId: string | null): Promise<void> {
    const stem = this.stems.get(name);
    if (!stem) return;
    const existing = this.instrumentSamplers.get(name);
    if (existing) {
      try {
        existing.stop();
        existing.output?.output?.disconnect?.();
        existing.disconnect?.();
      } catch {
        /* ignore */
      }
      this.instrumentSamplers.delete(name);
    }
    if (!instrumentId) {
      this.stemInstrument.delete(name);
      if (this.playing) this.restartStem(name, true);
      return;
    }
    this.stemInstrument.set(name, instrumentId);
    const { Soundfont } = await import("smplr");
    const inst: any = Soundfont(this.ctx, {
      instrument: instrumentId,
      kit: "FluidR3_GM",
      destination: stem.gain,
    });
    await inst.load;
    if (this.stemInstrument.get(name) !== instrumentId) {
      try {
        inst.stop();
      } catch {
        /* superseded by a newer selection */
      }
      return;
    }
    this.instrumentSamplers.set(name, inst);
    if (this.playing) this.restartStem(name, false);
  }

  private restartStem(name: string, toBuffer: boolean) {
    const src = this.sources.get(name);
    if (src) {
      try {
        src.onended = null;
        src.stop();
      } catch {
        /* ignore */
      }
      src.disconnect();
      this.sources.delete(name);
    }
    const inst = this.instrumentSamplers.get(name);
    if (inst) {
      try {
        inst.stop();
      } catch {
        /* ignore */
      }
    }
    const stem = this.stems.get(name);
    if (!stem) return;
    if (!toBuffer && inst) {
      this.scheduleInstrument(name);
    } else {
      const s = this.ctx.createBufferSource();
      s.buffer = stem.buffer;
      s.connect(stem.gain);
      s.start(0, Math.min(this.rawTime(), stem.buffer.duration));
      this.sources.set(name, s);
    }
  }

  private scheduleInstrument(name: string) {
    const inst = this.instrumentSamplers.get(name);
    if (!inst) return;
    const notes = this.stemNotesMap.get(name) || [];
    const now = this.rawTime();
    for (const n of notes) {
      const end = n.start + n.dur;
      if (end <= now + 0.02) continue;
      const when = this.startedAt + (n.start - this.offset);
      const startAt = Math.max(when, this.ctx.currentTime + 0.01);
      const remaining = end - Math.max(n.start, now);
      inst.start({
        note: Math.round(n.midi),
        detune: Math.round((n.midi - Math.round(n.midi)) * 100), // cents, for off-tune notes
        time: startAt,
        duration: Math.max(0.08, remaining),
        velocity: Math.round(50 + 70 * Math.min(1, Math.max(0, n.confidence ?? 1))),
      });
    }
  }

  /** Immutable original buffer for a stem (for re-rendering pitch edits). */
  getOriginalBuffer(name: string): AudioBuffer | null {
    return this.stems.get(name)?.original ?? null;
  }

  /** Fetch + decode an audio URL into an AudioBuffer on this engine's context. */
  async decodeUrl(url: string): Promise<AudioBuffer> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
    return this.ctx.decodeAudioData(await res.arrayBuffer());
  }

  /** Swap a stem's playing buffer (e.g. an edited render). Glitch-free while playing:
   *  the affected source is rebuilt at the current playhead. Buffer length must match. */
  swapBuffer(name: string, buffer: AudioBuffer) {
    const s = this.stems.get(name);
    if (!s) return;
    s.buffer = buffer;
    if (this.playing) {
      const old = this.sources.get(name);
      if (old) {
        try {
          old.onended = null;
          old.stop();
        } catch {
          /* already stopped */
        }
        old.disconnect();
      }
      const pos = this.rawTime();
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(s.gain);
      src.start(0, Math.min(pos, buffer.duration));
      this.sources.set(name, src);
    }
  }

  async play() {
    if (this.playing || this.stems.size === 0) return;
    if (this.ctx.state === "suspended") await this.ctx.resume();
    if (this.offset >= this.duration) this.offset = 0;
    this.startSources(this.offset);
    this.playing = true;
    this.onStateChange?.();
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.rawTime();
    this.stopSources();
    this.playing = false;
    this.onStateChange?.();
  }

  toggle() {
    this.playing ? this.pause() : this.play();
  }

  stop() {
    this.stopSources();
    this.offset = 0;
    if (this.playing) {
      this.playing = false;
      this.onStateChange?.();
    }
  }

  seek(t: number) {
    const clamped = Math.max(0, Math.min(t, this.duration));
    if (this.playing) {
      this.stopSources();
      this.offset = clamped;
      this.startSources(clamped);
    } else {
      this.offset = clamped;
    }
    this.onStateChange?.();
  }

  private rawTime(): number {
    if (!this.playing) return this.offset;
    return this.offset + (this.ctx.currentTime - this.startedAt);
  }

  /** Current playhead time; auto-stops at end. */
  getCurrentTime(): number {
    const t = this.rawTime();
    if (this.playing && t >= this.duration) {
      this.stopSources();
      this.offset = this.duration;
      this.playing = false;
      this.onStateChange?.();
      return this.duration;
    }
    return Math.min(t, this.duration);
  }

  isPlaying() {
    return this.playing;
  }

  /** Master output level 0..1 (RMS), for the VU meter. */
  getLevel(): number {
    this.analyser.getByteTimeDomainData(this.levelBuf);
    let sum = 0;
    for (let i = 0; i < this.levelBuf.length; i++) {
      const v = (this.levelBuf[i] - 128) / 128;
      sum += v * v;
    }
    return Math.min(1, Math.sqrt(sum / this.levelBuf.length) * 1.8);
  }

  setVolume(name: string, v: number) {
    const s = this.stems.get(name);
    if (!s) return;
    s.volume = Math.max(0, Math.min(1, v));
    this.applyGains();
  }

  toggleMute(name: string) {
    const s = this.stems.get(name);
    if (!s) return;
    s.muted = !s.muted;
    this.applyGains();
  }

  toggleSolo(name: string) {
    const s = this.stems.get(name);
    if (!s) return;
    s.soloed = !s.soloed;
    this.applyGains();
  }

  getState(name: string) {
    const s = this.stems.get(name);
    return s ? { volume: s.volume, muted: s.muted, soloed: s.soloed } : null;
  }

  async dispose() {
    this.stopSources();
    try {
      await this.ctx.close();
    } catch {
      /* ignore */
    }
  }
}

export function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const cs = Math.floor((seconds * 100) % 100);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}
