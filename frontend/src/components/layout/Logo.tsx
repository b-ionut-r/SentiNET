/** SentiNET mark: a signal trace resolving into a single point of truth. */
export function LogoMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden className="shrink-0">
      <rect width="32" height="32" rx="8" fill="rgb(var(--ink))" />
      <path d="M6 18h5l3-8 4 13 3-9h5" fill="none" stroke="rgb(var(--page))" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="26" cy="14" r="2.6" fill="rgb(var(--accent))" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="flex items-center gap-2">
      <LogoMark />
      <span className="text-[15px] font-semibold tracking-[-0.01em] text-ink">
        Senti<span className="text-ink-2">NET</span>
      </span>
    </span>
  );
}
