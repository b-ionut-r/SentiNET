import { useEffect, useRef, useState, type FocusEvent, type KeyboardEvent, type ReactNode } from "react";

import { useTooltip } from "../components/ui/Tooltip";

/**
 * Keyboard access for a row of chart marks with ONE tab stop: the container
 * takes focus, ←/→ (Home/End) walk the marks, and the active mark shows its
 * tooltip plus a visible focus ring (`[data-kb-active]`, styled in CSS).
 * Pointer hover stays per mark.
 */
export function useRoving(tips: ReadonlyArray<ReactNode | null>, ariaLabel: string) {
  const { showAt, hide } = useTooltip();
  const marks = useRef<Array<HTMLElement | null>>([]);
  const [active, setActive] = useState<number | null>(null);
  const n = tips.length;

  useEffect(() => {
    if (active == null) return;
    const el = marks.current[active];
    const tip = tips[active];
    if (el && tip) showAt(tip, el);
  }, [active, tips, showAt]);

  const container = {
    tabIndex: n > 0 ? 0 : -1,
    role: "group" as const,
    "aria-label": `${ariaLabel} — use arrow keys to read each value`,
    className: "roving",
    onFocus: (e: FocusEvent<HTMLElement>) => {
      if (e.target === e.currentTarget && n > 0) setActive((a) => a ?? n - 1);
    },
    onBlur: () => {
      setActive(null);
      hide();
    },
    onKeyDown: (e: KeyboardEvent<HTMLElement>) => {
      if (!n) return;
      const step: Record<string, (a: number) => number> = {
        ArrowRight: (a) => Math.min(n - 1, a + 1),
        ArrowLeft: (a) => Math.max(0, a - 1),
        Home: () => 0,
        End: () => n - 1,
      };
      const f = step[e.key];
      if (!f) return;
      e.preventDefault();
      setActive((a) => f(a ?? n - 1));
    },
  };

  /** Props for mark `i`: ref for tooltip anchoring, active flag, pointer hover. */
  const mark = (i: number) => ({
    ref: (el: HTMLElement | null) => {
      marks.current[i] = el;
    },
    "data-kb-active": active === i ? "true" : undefined,
    onMouseEnter: (e: React.MouseEvent<HTMLElement>) => {
      const tip = tips[i];
      if (tip) showAt(tip, e.currentTarget);
    },
    onMouseLeave: () => {
      if (active == null) hide();
    },
  });

  return { container, mark, active };
}
