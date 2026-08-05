import { useEffect, useRef, useState } from "react";
import * as api from "../lib/api";
import type { AudioEngine } from "../lib/audioEngine";
import { CANVAS_BASE, darken, lighten, rgba } from "../lib/colors";
import {
  applySnap,
  centsOff,
  isBlackKey,
  isInScale,
  noteName,
  parseKey,
  Scale,
  snapToScale,
} from "../lib/musicTheory";
import type { Note, SnapMode } from "../lib/types";
import { BackIcon } from "./icons";

interface Props {
  projectId: string;
  stemName: string;
  color: string;
  initialNotes: Note[];
  analyzed: boolean;
  musicalKey: string | null;
  engine: AudioEngine;
  rendering?: boolean;
  onBack: () => void;
  onNotesChanged: (notes: Note[]) => void;
}

interface Layout {
  gutterW: number;
  rulerH: number;
  pxPerSec: number;
  pxPerSemi: number;
  minMidi: number;
  maxMidi: number;
  duration: number;
  w: number;
  h: number;
}

const GUTTER = 54;
const RULER = 24;

export default function NoteEditor({
  projectId,
  stemName,
  color,
  initialNotes,
  analyzed,
  musicalKey,
  engine,
  rendering,
  onBack,
  onNotesChanged,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const readoutRef = useRef<HTMLSpanElement>(null);
  const notesRef = useRef<Note[]>(initialNotes.map((n) => ({ ...n })));
  const layoutRef = useRef<Layout | null>(null);
  const dragRef = useRef<{ id: string; startY: number; startMidi: number } | null>(null);
  const snapRef = useRef<SnapMode>("chromatic");
  const rafRef = useRef<number | null>(null);
  const saveTimer = useRef<number | null>(null);

  const [keyStr, setKeyStr] = useState<string | null>(musicalKey);
  const scale: Scale = parseKey(keyStr);
  const [loading, setLoading] = useState(!analyzed || initialNotes.length === 0);
  const [error, setError] = useState<string | null>(null);
  const [snapMode, setSnapMode] = useState<SnapMode>("chromatic");
  const [editedCount, setEditedCount] = useState(0);

  useEffect(() => {
    snapRef.current = snapMode;
  }, [snapMode]);

  const recountEdited = () => {
    const c = notesRef.current.filter(
      (n) => Math.abs(n.midi - n.original_midi) > 0.01
    ).length;
    setEditedCount(c);
  };

  // ---- Analyze on open if needed ----
  useEffect(() => {
    let cancelled = false;
    if (analyzed && initialNotes.length > 0) {
      setLoading(false);
      recountEdited();
      draw();
      return;
    }
    setLoading(true);
    api
      .analyzeStem(projectId, stemName)
      .then((proj) => {
        if (cancelled) return;
        const stem = proj.stems.find((s) => s.name === stemName);
        notesRef.current = (stem?.notes ?? []).map((n) => ({ ...n }));
        setKeyStr(proj.musical_key ?? null);
        setLoading(false);
        recountEdited();
        draw();
      })
      .catch((e) => !cancelled && (setError(String(e)), setLoading(false)));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, stemName]);

  // ---- Layout ----
  const computeLayout = (): Layout | null => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return null;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w < 2 || h < 2) return null;

    const notes = notesRef.current;
    const maxNoteEnd = notes.reduce((m, n) => Math.max(m, n.start + n.dur), 0);
    const duration = Math.max(engine.duration || 0, maxNoteEnd, 1);

    let minP = 60;
    let maxP = 72;
    if (notes.length) {
      const ps = notes.flatMap((n) => [n.midi, n.original_midi]);
      minP = Math.floor(Math.min(...ps)) - 3;
      maxP = Math.ceil(Math.max(...ps)) + 3;
    }
    if (maxP - minP < 12) {
      const mid = Math.round((minP + maxP) / 2);
      minP = mid - 6;
      maxP = mid + 6;
    }
    const laneCount = maxP - minP + 1;
    return {
      gutterW: GUTTER,
      rulerH: RULER,
      pxPerSec: (w - GUTTER) / duration,
      pxPerSemi: (h - RULER) / laneCount,
      minMidi: minP,
      maxMidi: maxP,
      duration,
      w,
      h,
    };
  };

  // ---- Draw ----
  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const L = computeLayout();
    if (!L) return;
    layoutRef.current = L;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(L.w * dpr);
    canvas.height = Math.floor(L.h * dpr);
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, L.w, L.h);

    const xOf = (t: number) => L.gutterW + t * L.pxPerSec;
    const yOf = (m: number) => L.rulerH + (L.maxMidi - m) * L.pxPerSemi;

    // ---- pitch lanes ----
    for (let m = L.minMidi; m <= L.maxMidi; m++) {
      const y = yOf(m) - L.pxPerSemi; // top of lane row for pitch m (row spans m..m centered)
      const top = yOf(m);
      const inScale = isInScale(scale, m);
      ctx.fillStyle = inScale
        ? "rgba(255,255,255,0.045)"
        : "rgba(0,0,0,0.22)";
      ctx.fillRect(L.gutterW, top, L.w - L.gutterW, L.pxPerSemi);
      // divider
      const isC = ((m % 12) + 12) % 12 === 0;
      ctx.strokeStyle = isC ? "rgba(255,255,255,0.14)" : "rgba(255,255,255,0.05)";
      ctx.beginPath();
      ctx.moveTo(L.gutterW, Math.floor(top) + 0.5);
      ctx.lineTo(L.w, Math.floor(top) + 0.5);
      ctx.stroke();
      // gutter label on C and naturals
      if (isC || !isBlackKey(m)) {
        ctx.fillStyle = isC ? "rgba(233,235,239,0.9)" : "rgba(167,173,186,0.6)";
        ctx.font = `${isC ? 600 : 400} 10px ui-monospace, monospace`;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(noteName(m), L.gutterW - 8, top + L.pxPerSemi / 2);
      }
      void y;
    }

    // ---- ruler ----
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fillRect(0, 0, L.w, L.rulerH);
    ctx.fillStyle = "rgba(167,173,186,0.8)";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const secStep = L.duration > 30 ? 5 : 1;
    for (let s = 0; s <= L.duration; s += secStep) {
      const x = xOf(s);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.beginPath();
      ctx.moveTo(Math.floor(x) + 0.5, L.rulerH);
      ctx.lineTo(Math.floor(x) + 0.5, L.h);
      ctx.stroke();
      ctx.fillText(`${s}s`, x + 3, L.rulerH / 2);
    }

    // ---- gutter separator ----
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.beginPath();
    ctx.moveTo(L.gutterW + 0.5, 0);
    ctx.lineTo(L.gutterW + 0.5, L.h);
    ctx.stroke();

    // ---- blobs ----
    const fillBase = darken(color, CANVAS_BASE, 0.15);
    for (const n of notesRef.current) {
      const edited = Math.abs(n.midi - n.original_midi) > 0.01;
      const x = xOf(n.start);
      const wRect = Math.max(3, n.dur * L.pxPerSec);
      const h = Math.max(4, L.pxPerSemi - 2);

      // ghost at original pitch when edited
      if (edited) {
        const gy = yOf(n.original_midi) + 1;
        ctx.strokeStyle = "rgba(255,255,255,0.28)";
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(x, gy, wRect, h);
        ctx.setLineDash([]);
        // connector line
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.beginPath();
        ctx.moveTo(x + wRect / 2, gy + h / 2);
        ctx.lineTo(x + wRect / 2, yOf(n.midi) + 1 + h / 2);
        ctx.stroke();
      }

      const y = yOf(n.midi) + 1;
      const alpha = 0.55 + 0.45 * Math.min(1, Math.max(0, n.confidence));
      ctx.fillStyle = rgba(fillBase, alpha);
      ctx.beginPath();
      ctx.roundRect(x, y, wRect, h, 3);
      ctx.fill();
      // top highlight
      ctx.fillStyle = rgba(lighten(color, 0.4), 0.9);
      ctx.fillRect(x + 1, y + 1, wRect - 2, Math.max(1, h * 0.18));
      // edited outline
      if (edited) {
        ctx.strokeStyle = "#f2a63d";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.5, y + 0.5, wRect - 1, h - 1);
        ctx.lineWidth = 1;
      }
    }
  };

  // ---- Redraw on snap change / resize ----
  useEffect(() => {
    draw();
    const ro = new ResizeObserver(() => draw());
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapMode, loading, keyStr]);

  // ---- Playhead RAF ----
  useEffect(() => {
    const loop = () => {
      const L = layoutRef.current;
      if (L && playheadRef.current) {
        const t = engine.getCurrentTime();
        playheadRef.current.style.transform = `translateX(${L.gutterW + t * L.pxPerSec}px)`;
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [engine]);

  // ---- Persist (debounced) ----
  const persist = () => {
    onNotesChanged(notesRef.current.map((n) => ({ ...n })));
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => {
      api.saveNotes(projectId, stemName, notesRef.current).catch(() => {});
    }, 400);
  };

  // ---- Hit testing ----
  const hitTest = (ox: number, oy: number) => {
    const L = layoutRef.current;
    if (!L) return null;
    if (oy < L.rulerH) return { type: "ruler" as const };
    const t = (ox - L.gutterW) / L.pxPerSec;
    const mfrac = L.maxMidi - (oy - L.rulerH) / L.pxPerSemi;
    const notes = notesRef.current;
    for (let i = notes.length - 1; i >= 0; i--) {
      const n = notes[i];
      if (t >= n.start && t <= n.start + n.dur && Math.abs(mfrac - n.midi) <= 0.6) {
        return { type: "note" as const, note: n };
      }
    }
    return { type: "grid" as const, time: Math.max(0, t) };
  };

  const setReadout = (txt: string) => {
    if (readoutRef.current) readoutRef.current.textContent = txt;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ox = e.clientX - rect.left;
    const oy = e.clientY - rect.top;
    const hit = hitTest(ox, oy);
    if (!hit) return;
    if (hit.type === "note") {
      dragRef.current = { id: hit.note.id, startY: oy, startMidi: hit.note.midi };
      e.currentTarget.setPointerCapture(e.pointerId);
    } else if (hit.type === "grid") {
      engine.seek(hit.time);
    } else if (hit.type === "ruler") {
      const L = layoutRef.current!;
      engine.seek(Math.max(0, (ox - L.gutterW) / L.pxPerSec));
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const L = layoutRef.current;
    if (!L) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const oy = e.clientY - rect.top;
    const d = dragRef.current;
    if (d) {
      const deltaSemi = -(oy - d.startY) / L.pxPerSemi;
      let target = applySnap(d.startMidi + deltaSemi, snapRef.current, scale);
      target = Math.max(L.minMidi, Math.min(L.maxMidi, target));
      const n = notesRef.current.find((x) => x.id === d.id);
      if (n) {
        n.midi = target;
        const cents = centsOff(target);
        setReadout(`${noteName(target)} ${cents >= 0 ? "+" : ""}${cents}¢`);
        draw();
      }
    } else {
      const mfrac = L.maxMidi - (oy - L.rulerH) / L.pxPerSemi;
      const cents = centsOff(mfrac);
      setReadout(`${noteName(mfrac)} ${cents >= 0 ? "+" : ""}${cents}¢`);
    }
  };

  const onPointerUp = () => {
    if (dragRef.current) {
      dragRef.current = null;
      recountEdited();
      persist();
    }
  };

  const correctPitch = () => {
    for (const n of notesRef.current) n.midi = snapToScale(n.midi, scale);
    recountEdited();
    draw();
    persist();
  };

  const resetAll = () => {
    for (const n of notesRef.current) n.midi = n.original_midi;
    recountEdited();
    draw();
    persist();
  };

  return (
    <div className="note-editor">
      <div className="ne-toolbar">
        <button className="iconbtn" onClick={onBack} title="Back to tracks">
          <BackIcon />
        </button>
        <span className="ne-color" style={{ background: color }} />
        <span className="ne-title">{stemName}</span>
        <span className="ne-key mono">{scale.label}</span>

        <div className="ne-spacer" />

        <div className="ne-snap">
          {(["none", "chromatic", "scale"] as SnapMode[]).map((m) => (
            <button
              key={m}
              className={snapMode === m ? "sel" : ""}
              onClick={() => setSnapMode(m)}
              title={`Snap: ${m}`}
            >
              {m === "none" ? "Free" : m === "chromatic" ? "½" : "Scale"}
            </button>
          ))}
        </div>

        <button className="btn btn-ghost ne-macro" onClick={correctPitch} title="Snap all notes to the key">
          Correct Pitch
        </button>
        <button className="btn btn-ghost" onClick={resetAll} disabled={editedCount === 0}>
          Reset
        </button>
        <span className="ne-edited mono">{editedCount} edited</span>
        {rendering && <span className="ne-rendering mono">rendering…</span>}
        <span className="ne-readout mono" ref={readoutRef} />
      </div>

      <div className="ne-canvas-wrap" ref={wrapRef}>
        {loading && (
          <div className="ne-overlay">
            <div className="reel-loader" />
            <div className="muted">Detecting notes…</div>
          </div>
        )}
        {error && <div className="ne-overlay error">Analysis failed: {error}</div>}
        {!loading && !error && (
          <canvas
            ref={canvasRef}
            className="ne-canvas"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => setReadout("")}
          />
        )}
        <div className="ne-playhead" ref={playheadRef} />
      </div>
    </div>
  );
}
