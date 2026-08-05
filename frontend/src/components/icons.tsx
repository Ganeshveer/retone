/* Flat 2D iconography — no gloss, no bevel. Single-color strokes/fills. */
import React from "react";

type P = { size?: number; className?: string };

const svg = (size: number, className: string | undefined, children: React.ReactNode) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    {children}
  </svg>
);

export const PlayIcon = ({ size = 20, className }: P) =>
  svg(size, className, <path d="M7 5l12 7-12 7V5z" fill="currentColor" stroke="none" />);

export const PauseIcon = ({ size = 20, className }: P) =>
  svg(
    size,
    className,
    <>
      <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none" />
      <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none" />
    </>
  );

export const StopIcon = ({ size = 20, className }: P) =>
  svg(size, className, <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />);

export const UploadIcon = ({ size = 20, className }: P) =>
  svg(
    size,
    className,
    <>
      <path d="M12 16V4" />
      <path d="M7 9l5-5 5 5" />
      <path d="M5 20h14" />
    </>
  );

export const CassetteIcon = ({ size = 20, className }: P) =>
  svg(
    size,
    className,
    <>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <circle cx="8.5" cy="12" r="2.2" />
      <circle cx="15.5" cy="12" r="2.2" />
      <path d="M7 17h10" />
    </>
  );

export const BackIcon = ({ size = 20, className }: P) =>
  svg(size, className, <path d="M15 18l-6-6 6-6" />);

export const SkipStartIcon = ({ size = 18, className }: P) =>
  svg(
    size,
    className,
    <>
      <path d="M18 5v14l-9-7 9-7z" fill="currentColor" stroke="none" />
      <rect x="5" y="5" width="2.4" height="14" rx="1" fill="currentColor" stroke="none" />
    </>
  );
