import { useEffect, useRef, useState } from "react";
import { AudioEngine, formatTime, LoadedStem } from "../lib/audioEngine";
import type { Note, Project } from "../lib/types";
import { TIER_LABELS } from "../lib/types";
import { renderStemWithEdits } from "../lib/noteShift";
import NoteEditor from "./NoteEditor";
import StemLane from "./StemLane";
import TransportBar from "./TransportBar";
import { WaveformHandle } from "./Waveform";
import { VuHandle } from "./VuMeter";

const CONTROLS_W = 240;

interface Props {
  project: Project;
  onBack: () => void;
}

export default function DawView({ project, onBack }: Props) {
  const engineRef = useRef<AudioEngine | null>(null);
  const rafRef = useRef<number | null>(null);
  const waveRefs = useRef<Array<WaveformHandle | null>>([]);
  const timecodeRef = useRef<HTMLSpanElement>(null);
  const vuRef = useRef<VuHandle>(null);
  const playheadRef = useRef<HTMLDivElement>(null);

  const [loadedStems, setLoadedStems] = useState<LoadedStem[]>([]);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<"tracks" | "notes">("tracks");
  const [activeStemName, setActiveStemName] = useState<string | null>(null);
  const [stemNotes, setStemNotes] = useState<Record<string, Note[]>>({});
  const [rendering, setRendering] = useState(false);
  const renderTimer = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const eng = new AudioEngine();
    engineRef.current = eng;
    eng.onStateChange = () => setPlaying(eng.isPlaying());

    const sources = project.stems
      .filter((s) => s.url)
      .map((s) => ({ name: s.name, color: s.color, url: s.url as string }));

    const tick = () => {
      const t = eng.getCurrentTime();
      const frac = eng.duration > 0 ? t / eng.duration : 0;
      if (playheadRef.current) playheadRef.current.style.left = `${frac * 100}%`;
      if (timecodeRef.current) timecodeRef.current.textContent = formatTime(t);
      vuRef.current?.setLevel(eng.isPlaying() ? eng.getLevel() : 0);
      for (const w of waveRefs.current) w?.setProgress(frac);
      rafRef.current = requestAnimationFrame(tick);
    };

    eng
      .load(sources)
      .then((loaded) => {
        if (cancelled) return;
        waveRefs.current = new Array(loaded.length).fill(null);
        setLoadedStems(loaded);
        setDuration(eng.duration);
        setLoading(false);
        rafRef.current = requestAnimationFrame(tick);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(String(e));
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      eng.dispose();
    };
  }, [project.id]);

  // Spacebar toggles playback.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && !(e.target as HTMLElement)?.closest("input")) {
        e.preventDefault();
        engineRef.current?.toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const seekFraction = (f: number) => {
    const eng = engineRef.current;
    if (eng) eng.seek(f * eng.duration);
  };

  const activeStem = activeStemName
    ? project.stems.find((s) => s.name === activeStemName) ?? null
    : null;

  // Debounced: re-render the stem's audio from the immutable original + edits, then swap.
  const handleNotesChanged = (stemName: string, notes: Note[]) => {
    setStemNotes((p) => ({ ...p, [stemName]: notes }));
    const eng = engineRef.current;
    if (!eng) return;
    if (renderTimer.current) window.clearTimeout(renderTimer.current);
    renderTimer.current = window.setTimeout(async () => {
      const orig = eng.getOriginalBuffer(stemName);
      if (!orig) return;
      setRendering(true);
      try {
        const buf = await renderStemWithEdits(orig, notes);
        eng.swapBuffer(stemName, buf ?? orig);
      } catch (e) {
        console.error("pitch render failed", e);
      } finally {
        setRendering(false);
      }
    }, 350);
  };

  return (
    <div className="daw">
      <TransportBar
        projectName={project.name}
        tierLabel={TIER_LABELS[project.tier]}
        playing={playing}
        durationText={formatTime(duration)}
        bpm={project.bpm}
        musicalKey={project.musical_key}
        timecodeRef={timecodeRef}
        vuRef={vuRef}
        onToggle={() => engineRef.current?.toggle()}
        onStop={() => engineRef.current?.stop()}
        onSeekStart={() => engineRef.current?.seek(0)}
        onBack={onBack}
      />

      {loading && (
        <div className="daw-center">
          <div className="reel-loader" />
          <div className="muted">Loading stems…</div>
        </div>
      )}
      {error && <div className="daw-center error">Failed to load stems: {error}</div>}

      {!loading && !error && mode === "tracks" && (
        <div className="tracks-wrap" style={{ ["--controls-w" as any]: `${CONTROLS_W}px` }}>
          {loadedStems.map((s, i) => (
            <StemLane
              key={s.name}
              name={s.name}
              color={s.color}
              peaks={s.peaks}
              engine={engineRef.current!}
              onSeekFraction={seekFraction}
              onOpenNotes={() => {
                setActiveStemName(s.name);
                setMode("notes");
              }}
              ref={(el) => (waveRefs.current[i] = el)}
            />
          ))}
          <div className="playhead-layer">
            <div ref={playheadRef} className="playhead" />
          </div>
        </div>
      )}

      {!loading && !error && mode === "notes" && activeStem && engineRef.current && (
        <NoteEditor
          key={activeStem.name}
          projectId={project.id}
          stemName={activeStem.name}
          color={activeStem.color}
          initialNotes={stemNotes[activeStem.name] ?? activeStem.notes ?? []}
          analyzed={
            (stemNotes[activeStem.name]?.length ?? 0) > 0 || !!activeStem.analyzed
          }
          musicalKey={project.musical_key ?? null}
          engine={engineRef.current}
          rendering={rendering}
          onBack={() => setMode("tracks")}
          onNotesChanged={(n) => handleNotesChanged(activeStem.name, n)}
        />
      )}
    </div>
  );
}
