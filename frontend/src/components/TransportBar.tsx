import { RefObject } from "react";
import { BackIcon, PauseIcon, PlayIcon, SkipStartIcon, StopIcon } from "./icons";
import VuMeter, { VuHandle } from "./VuMeter";

interface Props {
  projectName: string;
  tierLabel: string;
  playing: boolean;
  durationText: string;
  bpm?: number | null;
  musicalKey?: string | null;
  timecodeRef: RefObject<HTMLSpanElement>;
  vuRef: RefObject<VuHandle>;
  onToggle: () => void;
  onStop: () => void;
  onSeekStart: () => void;
  onBack: () => void;
}

export default function TransportBar({
  projectName,
  tierLabel,
  playing,
  durationText,
  bpm,
  musicalKey,
  timecodeRef,
  vuRef,
  onToggle,
  onStop,
  onSeekStart,
  onBack,
}: Props) {
  return (
    <div className="transport">
      <div className="transport-left">
        <button className="iconbtn" onClick={onBack} title="Back to library">
          <BackIcon />
        </button>
        <div className="proj-meta">
          <div className="proj-name">{projectName}</div>
          <div className="proj-tier">{tierLabel}</div>
        </div>
      </div>

      <div className="transport-center">
        <button className="iconbtn" onClick={onSeekStart} title="Return to start">
          <SkipStartIcon />
        </button>
        <button className="play-btn" onClick={onToggle} title={playing ? "Pause" : "Play"}>
          {playing ? <PauseIcon size={22} /> : <PlayIcon size={22} />}
        </button>
        <button className="iconbtn" onClick={onStop} title="Stop">
          <StopIcon />
        </button>
        <div className="timecode mono">
          <span ref={timecodeRef}>00:00.00</span>
          <span className="tc-total"> / {durationText}</span>
        </div>
      </div>

      <div className="transport-right">
        <div className="readout mono">
          <span className="readout-label">BPM</span>
          <span className="readout-val">{bpm ? Math.round(bpm) : "—"}</span>
        </div>
        <div className="readout mono">
          <span className="readout-label">KEY</span>
          <span className="readout-val">{musicalKey ?? "—"}</span>
        </div>
        <VuMeter ref={vuRef} />
      </div>
    </div>
  );
}
