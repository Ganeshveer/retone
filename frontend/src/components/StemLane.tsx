import { forwardRef, useState } from "react";
import type { AudioEngine } from "../lib/audioEngine";
import { CANVAS_BASE, darken, rgba } from "../lib/colors";
import Waveform, { WaveformHandle } from "./Waveform";

interface Props {
  name: string;
  color: string;
  peaks: Float32Array;
  engine: AudioEngine;
  onSeekFraction: (f: number) => void;
  onOpenNotes: () => void;
}

/** One multitrack lane: fixed control column + a colored clip containing the waveform. */
const StemLane = forwardRef<WaveformHandle, Props>(
  ({ name, color, peaks, engine, onSeekFraction, onOpenNotes }, ref) => {
    const init = engine.getState(name) ?? { volume: 0.85, muted: false, soloed: false };
    const [muted, setMuted] = useState(init.muted);
    const [soloed, setSoloed] = useState(init.soloed);
    const [volume, setVolume] = useState(init.volume);

    const clipBg = rgba(darken(color, CANVAS_BASE, 0.6), 1);
    const clipBorder = rgba(darken(color, CANVAS_BASE, 0.3), 1);

    return (
      <div className="lane">
        <div className="lane-controls">
          <span className="lane-color" style={{ background: color }} />
          <span className="lane-name" title={name}>
            {name}
          </span>
          <div className="lane-btns">
            <button
              className={`tag-btn ${muted ? "on-mute" : ""}`}
              onClick={() => {
                engine.toggleMute(name);
                setMuted(engine.getState(name)!.muted);
              }}
              title="Mute"
            >
              M
            </button>
            <button
              className={`tag-btn ${soloed ? "on-solo" : ""}`}
              onClick={() => {
                engine.toggleSolo(name);
                setSoloed(engine.getState(name)!.soloed);
              }}
              title="Solo"
            >
              S
            </button>
            <button className="tag-btn notes-btn" onClick={onOpenNotes} title="Edit notes / pitch">
              ✎
            </button>
          </div>
          <input
            className="vol"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={volume}
            style={{ ["--track-color" as any]: color }}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setVolume(v);
              engine.setVolume(name, v);
            }}
            title="Volume"
          />
        </div>
        <div
          className="lane-clip"
          style={{ background: clipBg, borderColor: clipBorder }}
        >
          <Waveform ref={ref} peaks={peaks} color={color} onSeek={onSeekFraction} />
        </div>
      </div>
    );
  }
);

StemLane.displayName = "StemLane";
export default StemLane;
