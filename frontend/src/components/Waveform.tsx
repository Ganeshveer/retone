import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";
import { lighten, rgba } from "../lib/colors";

export interface WaveformHandle {
  setProgress: (fraction: number) => void;
}

interface Props {
  peaks: Float32Array;
  color: string;
  onSeek?: (fraction: number) => void;
}

/**
 * Canvas waveform drawn inside the track's colored clip (SoundCloud-style).
 * Static bars are rendered once to an offscreen canvas; each frame we blit and tint the
 * played region — so 60fps playhead updates cost almost nothing and never re-render React.
 */
const Waveform = forwardRef<WaveformHandle, Props>(({ peaks, color, onSeek }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const staticRef = useRef<HTMLCanvasElement | null>(null);
  const progressRef = useRef(0);
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });

  const renderStatic = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    sizeRef.current = { w, h, dpr };
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);

    const off = staticRef.current ?? document.createElement("canvas");
    staticRef.current = off;
    off.width = canvas.width;
    off.height = canvas.height;
    const g = off.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);

    const bar = rgba(lighten(color, 0.42), 0.9);
    g.fillStyle = bar;
    const mid = h / 2;
    const n = peaks.length;
    for (let x = 0; x < w; x++) {
      const idx = Math.min(n - 1, Math.floor((x / w) * n));
      const amp = peaks[idx];
      const barH = Math.max(1, amp * (h * 0.46));
      g.fillRect(x, mid - barH, 1, barH * 2);
    }
    draw();
  };

  const draw = () => {
    const canvas = canvasRef.current;
    const off = staticRef.current;
    if (!canvas || !off) return;
    const { w, h, dpr } = sizeRef.current;
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0);
    // Brighten only the bar pixels within the played region.
    const playedX = progressRef.current * w * dpr;
    ctx.globalCompositeOperation = "source-atop";
    ctx.fillStyle = "rgba(255,255,255,0.28)";
    ctx.fillRect(0, 0, playedX, h * dpr);
    ctx.globalCompositeOperation = "source-over";
  };

  useImperativeHandle(ref, () => ({
    setProgress: (fraction: number) => {
      progressRef.current = Math.max(0, Math.min(1, fraction));
      draw();
    },
  }));

  useEffect(() => {
    renderStatic();
    const ro = new ResizeObserver(() => renderStatic());
    if (canvasRef.current) ro.observe(canvasRef.current);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [peaks, color]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek((e.clientX - rect.left) / rect.width);
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{ width: "100%", height: "100%", display: "block", cursor: "text" }}
    />
  );
});

Waveform.displayName = "Waveform";
export default Waveform;
