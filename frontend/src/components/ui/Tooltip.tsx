/**
 * One floating tooltip for the whole app. Charts call `show()` with rich
 * content on pointer move / focus and `hide()` on leave; text elements can
 * wrap themselves in <Tip>. Rendered in a portal, clamped to the viewport.
 */
import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

interface TipState {
  content: ReactNode;
  x: number;
  y: number;
}

interface TipApi {
  show: (content: ReactNode, x: number, y: number) => void;
  showAt: (content: ReactNode, el: Element) => void;
  hide: () => void;
}

const Ctx = createContext<TipApi>({ show: () => {}, showAt: () => {}, hide: () => {} });

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const show = useCallback((content: ReactNode, x: number, y: number) => setTip({ content, x, y }), []);
  const showAt = useCallback((content: ReactNode, el: Element) => {
    const r = el.getBoundingClientRect();
    setTip({ content, x: r.left + r.width / 2, y: r.top });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const api = useMemo(() => ({ show, showAt, hide }), [show, showAt, hide]);
  return (
    <Ctx.Provider value={api}>
      {children}
      {tip && createPortal(<Floating {...tip} />, document.body)}
    </Ctx.Provider>
  );
}

function Floating({ content, x, y }: TipState) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: x, top: y, ready: false });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const pad = 8;
    let left = x - w / 2;
    left = Math.max(pad, Math.min(window.innerWidth - w - pad, left));
    let top = y - h - 10;
    if (top < pad) top = y + 18;
    setPos({ left, top, ready: true });
  }, [x, y, content]);

  return (
    <div
      ref={ref}
      role="tooltip"
      className="pointer-events-none fixed z-[100] max-w-[300px] rounded-lg bg-raised px-2.5 py-2 text-xs text-ink-2 shadow-pop"
      style={{ left: pos.left, top: pos.top, opacity: pos.ready ? 1 : 0 }}
    >
      {content}
    </div>
  );
}

export function useTooltip(): TipApi {
  return useContext(Ctx);
}

/** Hover/focus tooltip for any inline element. */
export function Tip({ content, children, className }: { content: ReactNode; children: ReactNode; className?: string }) {
  const { showAt, hide } = useTooltip();
  return (
    <span
      className={className}
      onMouseEnter={(e) => showAt(content, e.currentTarget)}
      onMouseLeave={hide}
      onFocus={(e) => showAt(content, e.currentTarget)}
      onBlur={hide}
    >
      {children}
    </span>
  );
}

/** Tooltip row: value first (strong), then the series name with a line key. */
export function TipRow({ color, label, value }: { color?: string; label: ReactNode; value: ReactNode }) {
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      {color && <span className="h-0.5 w-3 shrink-0 rounded-full" style={{ background: color }} />}
      <span className="font-semibold text-ink num">{value}</span>
      <span className="text-muted">{label}</span>
    </div>
  );
}
