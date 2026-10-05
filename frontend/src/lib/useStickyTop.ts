import { useEffect, useRef, useState } from "react";

/**
 * Sticky offset for a sidebar block. When the block fits the viewport it pins
 * at `top`; when it is taller, it pins by its bottom edge instead, so it first
 * scrolls fully into view and then rides along with the longer column.
 */
export function useStickyTop<T extends HTMLElement>(top: number, margin = 16) {
  const ref = useRef<T>(null);
  const [offset, setOffset] = useState(top);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setOffset(Math.min(top, window.innerHeight - el.offsetHeight - margin));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    window.addEventListener("resize", update);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [top, margin]);
  return { ref, top: offset };
}
