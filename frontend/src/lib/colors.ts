export interface RGB {
  r: number;
  g: number;
  b: number;
}

export function hexToRgb(hex: string): RGB {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(full, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export function rgba({ r, g, b }: RGB, a: number): string {
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

/** Mix a color toward white by amt (0..1). */
export function lighten(hex: string, amt: number): RGB {
  const { r, g, b } = hexToRgb(hex);
  return {
    r: Math.round(r + (255 - r) * amt),
    g: Math.round(g + (255 - g) * amt),
    b: Math.round(b + (255 - b) * amt),
  };
}

/** Mix a color toward a dark base by amt (0..1) — for muted clip backgrounds. */
export function darken(hex: string, base: RGB, amt: number): RGB {
  const { r, g, b } = hexToRgb(hex);
  return {
    r: Math.round(r * (1 - amt) + base.r * amt),
    g: Math.round(g * (1 - amt) + base.g * amt),
    b: Math.round(b * (1 - amt) + base.b * amt),
  };
}

export const CANVAS_BASE: RGB = { r: 20, g: 22, b: 26 }; // matches --bg-0
