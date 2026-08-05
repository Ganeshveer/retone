import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

export interface VuHandle {
  setLevel: (level: number) => void;
}

const SEGMENTS = 22;
const WIDTH = 132;
const HEIGHT = 12;

/** Segmented LED VU meter (green → amber → red) with peak-hold. Driven imperatively. */
const VuMeter = forwardRef<VuHandle, {}>((_props, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peakRef = useRef(0);

  const render = (level: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== WIDTH * dpr) {
      canvas.width = WIDTH * dpr;
      canvas.height = HEIGHT * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    peakRef.current = Math.max(level, peakRef.current - 0.02);
    const gap = 2;
    const segW = (WIDTH - (SEGMENTS - 1) * gap) / SEGMENTS;
    const lit = Math.round(level * SEGMENTS);
    const peakSeg = Math.round(peakRef.current * SEGMENTS);

    for (let i = 0; i < SEGMENTS; i++) {
      const frac = i / SEGMENTS;
      let color: string;
      if (frac > 0.82) color = "#e0564b";
      else if (frac > 0.62) color = "#e8c33a";
      else color = "#3fa66a";
      const on = i < lit || i === peakSeg - 1;
      ctx.globalAlpha = on ? 1 : 0.16;
      ctx.fillStyle = color;
      ctx.fillRect(i * (segW + gap), 0, segW, HEIGHT);
    }
    ctx.globalAlpha = 1;
  };

  useImperativeHandle(ref, () => ({ setLevel: render }));

  useEffect(() => {
    render(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: WIDTH, height: HEIGHT, display: "block", borderRadius: 2 }}
      title="Master output"
    />
  );
});

VuMeter.displayName = "VuMeter";
export default VuMeter;
